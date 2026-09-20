use rayon::prelude::*;

use crate::error::{EdfError, Result};
use crate::file::EdfFile;
use crate::group::{GroupKind, PadMode, SignalGroup};
use crate::proxy::SignalProxy;

/// Tolerance for onset/gap comparisons, in seconds.
const EPS: f64 = 1e-9;

/// One contiguous piece of an epoch: flat samples `s_start..s_end` land at row column `dest`.
///
/// A window that crosses an EDF+D gap yields one run per side. Each run sits at its true time
/// offset within the row, so the columns that fall in the gap stay empty for the pad policy to
/// fill instead of being closed up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochRun {
    pub s_start: usize,
    pub s_end: usize,
    pub dest: usize,
}

impl EpochRun {
    pub fn len(&self) -> usize {
        self.s_end.saturating_sub(self.s_start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One planned epoch: the event onset, the flat span it touches, and the runs it decodes into.
///
/// `s_start..s_end` is the outer decoded span, the range reported by `epoch_windows`. It is
/// shorter than the nominal window when the window clips a file boundary. For a window that
/// crosses an EDF+D gap it covers both sides, and `runs` says where each side actually belongs.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochWindow {
    pub onset: f64,
    pub s_start: usize,
    pub s_end: usize,
    pub runs: Vec<EpochRun>,
}

impl EpochWindow {
    /// Number of row columns backed by real samples.
    pub fn decoded(&self) -> usize {
        self.runs.iter().map(EpochRun::len).sum()
    }
}

/// A planned set of epochs: one window per event, the per-window validity mask, and the
/// row width every epoch decodes into.
///
/// `n_samples` is the nominal window width `ceil((pre + post) * rate)`. It depends only on the
/// request, never on where the events fall, so rows stay time-aligned with each other and the
/// output shape is stable across files.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochPlan {
    pub windows: Vec<EpochWindow>,
    pub valid: Vec<bool>,
    pub n_samples: usize,
}

/// Edge/gap policy for [`extract_epochs`]. `Drop` is epoch-only; the fill variants reuse the
/// proxy [`PadMode`] so padding semantics cannot drift between the two features.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpochPad {
    /// Omit edge/gap-affected epochs from the output; record their indices as dropped.
    Drop,
    /// Keep every epoch; fill missing samples per `PadMode` and mark the row `valid = false`.
    Fill(PadMode),
}

/// Raise on writer- or spec-level argument problems (`ValueError` on the Python side).
fn epoch_argument(name: &'static str, reason: String) -> EdfError {
    EdfError::InvalidArgument { name, reason }
}

fn validate_pre_post(pre: f64, post: f64) -> Result<()> {
    if !pre.is_finite() || !post.is_finite() || pre < 0.0 || post < 0.0 || pre + post <= 0.0 {
        return Err(epoch_argument(
            "pre/post",
            format!(
                "window [onset - {pre}, onset + {post}) must be finite, non-negative, and non-empty"
            ),
        ));
    }
    Ok(())
}

/// Plan event times into fixed-length sample windows. No data is read.
///
/// `valid[i]` is false when window `i` runs outside `[0, duration)` or straddles an EDF+D
/// gap, independent of the pad policy chosen later. The decoded span is the same
/// `[s_start, s_end)` a `read_page` of that time range would touch.
pub fn plan_epochs(
    file: &EdfFile,
    group: &SignalGroup,
    events: &[f64],
    pre: f64,
    post: f64,
) -> Result<EpochPlan> {
    validate_pre_post(pre, post)?;
    if group.kind() != GroupKind::Rectangular {
        return Err(epoch_argument(
            "group",
            "epoch extraction requires a rectangular (uniform sample rate) signal group; \
             use signal_groups() to discover eligible groups"
                .into(),
        ));
    }
    let sample_rate = group
        .sample_rate()
        .ok_or_else(|| epoch_argument("group", "group has no sample rate".into()))?;
    let spr = group
        .samples_per_record()
        .ok_or_else(|| epoch_argument("group", "group has no samples-per-record".into()))?;
    if sample_rate <= 0.0 || !sample_rate.is_finite() {
        return Err(epoch_argument(
            "group",
            format!("group sample rate must be positive and finite, got {sample_rate}"),
        ));
    }

    let num_records = file.num_records();
    let total_samples = num_records.saturating_mul(spr);

    let is_plus_d = file.header().variant.is_plus_d();
    let record_dur = file.header().record_duration_secs;

    // The onset table, gap intervals, and wall-clock duration all come from the same table for
    // EDF+D. `sampled_duration` is the wall-clock span only when there are no gaps.
    let onsets: Option<Vec<f64>> = is_plus_d.then(|| {
        file.file()
            .with_annotations(|idx| idx.record_onsets.clone())
    });
    let duration = match &onsets {
        Some(o) => o.last().copied().unwrap_or(0.0) + record_dur,
        None => total_samples as f64 / sample_rate,
    };

    // Gap intervals (start, end) sorted ascending by start. A window straddles a gap when the
    // gap overlaps its time span away from the file edges.
    let gaps: Vec<(f64, f64)> = match &onsets {
        Some(o) => {
            let mut g = Vec::new();
            for r in 1..o.len() {
                let prev_end = o[r - 1] + record_dur;
                if o[r] > prev_end + EPS {
                    g.push((prev_end, o[r]));
                }
            }
            g
        }
        None => Vec::new(),
    };
    let gap_starts: Vec<f64> = gaps.iter().map(|&(s, _)| s).collect();
    let gap_ends: Vec<f64> = gaps.iter().map(|&(_, e)| e).collect();

    let proxy = is_plus_d
        .then(|| SignalProxy::new(file.file().clone(), group.indices()[0]))
        .transpose()?;

    // Nominal width, derived from the request alone. Every row is this wide, so column j of
    // any row is always `j / rate - pre` seconds from its event.
    let n_samples = ((pre + post) * sample_rate).ceil() as usize;

    let mut windows = Vec::with_capacity(events.len());
    let mut valid = Vec::with_capacity(events.len());
    for &t in events {
        if !t.is_finite() {
            return Err(epoch_argument(
                "events",
                format!("event time {t} is not finite"),
            ));
        }
        let w_start = t - pre;
        let w_end = t + post;
        let (s_start, s_end) = if let Some(proxy) = &proxy {
            file.file().sample_range_for_time(proxy, w_start, w_end)
        } else {
            let to_index = |s: f64| ((s.max(0.0) * sample_rate).ceil() as usize).min(total_samples);
            (to_index(w_start), to_index(w_end))
        };

        // Place each contiguous piece of the window at its own time offset in the row. For
        // EDF+C the window is one piece; for EDF+D each record contributes its overlap, and
        // adjacent records coalesce, so only real gaps leave a hole.
        let mut runs: Vec<EpochRun> = Vec::new();
        let push_run = |runs: &mut Vec<EpochRun>, s0: usize, s1: usize, dest: f64| {
            if s1 <= s0 {
                return;
            }
            let dest = dest.round().max(0.0) as usize;
            if dest >= n_samples {
                return;
            }
            let len = (s1 - s0).min(n_samples - dest);
            if let Some(last) = runs.last_mut()
                && last.s_end == s0
                && last.dest + last.len() == dest
            {
                last.s_end = s0 + len;
                return;
            }
            runs.push(EpochRun {
                s_start: s0,
                s_end: s0 + len,
                dest,
            });
        };

        match &onsets {
            Some(o) => {
                let first = o.partition_point(|&s| s + record_dur <= w_start + EPS);
                let last = o.partition_point(|&s| s < w_end - EPS);
                for (r, &rec_onset) in o.iter().enumerate().take(last.min(num_records)).skip(first)
                {
                    let seg_start = rec_onset.max(w_start);
                    let seg_end = (rec_onset + record_dur).min(w_end);
                    if seg_end <= seg_start + EPS {
                        continue;
                    }
                    let k0 = ((seg_start - rec_onset) * sample_rate).ceil().max(0.0) as usize;
                    let k1 = (((seg_end - rec_onset) * sample_rate).ceil() as usize).min(spr);
                    if k1 <= k0 {
                        continue;
                    }
                    // Columns are padded only where data is genuinely missing. A window
                    // opening inside this record has nothing missing before it, so its run
                    // starts at column 0 even when the window edge falls between samples.
                    let column = if rec_onset <= w_start + EPS {
                        0.0
                    } else {
                        (rec_onset - w_start) * sample_rate
                    };
                    push_run(&mut runs, r * spr + k0, r * spr + k1, column);
                }
            }
            None => {
                // The only data missing from a contiguous file is what sits before t = 0.
                let column = (-w_start).max(0.0) * sample_rate;
                push_run(&mut runs, s_start, s_end, column);
            }
        }

        // In-file and gap-free. The last gap beginning before `w_end` is the only candidate
        // that can overlap the window, since gap ends ascend with their starts.
        let mut ok = w_start >= -EPS && w_end <= duration + EPS;
        if ok && !gaps.is_empty() {
            let upto = gap_starts.partition_point(|&s| s < w_end - EPS);
            if upto > 0 {
                ok = gap_ends[upto - 1] <= w_start + EPS;
            }
        }

        windows.push(EpochWindow {
            onset: t,
            s_start,
            s_end,
            runs,
        });
        valid.push(ok);
    }
    Ok(EpochPlan {
        windows,
        valid,
        n_samples,
    })
}

/// Decode planned windows into one flat `(n_epochs, n_channels, n_samples)` row-major buffer.
///
/// Returns `(data, valid, dropped)`. Under [`EpochPad::Drop`], invalid windows are omitted,
/// `valid` is all-true for the kept rows, and `dropped` lists the planner indices removed.
/// Under a fill policy every window is kept; the row stays `valid = false` and every column
/// with no sample behind it is filled per [`PadMode`]. Those are the columns that fall off a
/// file boundary and, for a window crossing an EDF+D gap, the columns inside the gap. Each
/// contiguous run lands at its own time offset, so samples from opposite sides of a gap are
/// never closed up against each other. A valid row is decoded in full.
///
/// The row width is `plan.n_samples`, the nominal window width, so every row covers the same
/// time offsets relative to its event and the shape does not depend on where the events fall.
/// Bindings must not re-derive the width from `data.len()`.
pub fn extract_epochs(
    file: &EdfFile,
    group: &SignalGroup,
    plan: &EpochPlan,
    pad: EpochPad,
) -> Result<(Vec<f64>, Vec<bool>, Vec<usize>)> {
    let EpochPlan {
        windows,
        valid,
        n_samples: n,
    } = plan;
    if windows.len() != valid.len() {
        return Err(epoch_argument(
            "windows",
            "windows and valid must have equal length".into(),
        ));
    }
    let n = *n;
    let nc = group.indices().len();
    let total_samples = file.num_records() * group.samples_per_record().unwrap_or(0);

    if matches!(pad, EpochPad::Fill(PadMode::Raise))
        && let Some(i) = valid.iter().position(|&v| !v)
    {
        return Err(EdfError::SampleOutOfRange {
            index: windows[i].s_start,
            count: total_samples,
        });
    }

    let fill = match pad {
        EpochPad::Fill(PadMode::Nan) => f64::NAN,
        EpochPad::Fill(PadMode::Value(v)) => v,
        _ => 0.0,
    };

    let kept: Vec<usize> = match pad {
        EpochPad::Drop => (0..windows.len()).filter(|&i| valid[i]).collect(),
        _ => (0..windows.len()).collect(),
    };
    let dropped: Vec<usize> = match pad {
        EpochPad::Drop => (0..windows.len()).filter(|&i| !valid[i]).collect(),
        _ => Vec::new(),
    };
    let flags: Vec<bool> = match pad {
        EpochPad::Drop => vec![true; kept.len()],
        _ => valid.clone(),
    };

    let row_size = nc * n;
    let mut data = vec![fill; kept.len() * row_size];
    let edge = matches!(pad, EpochPad::Fill(PadMode::Edge));

    // Each task owns one row block (disjoint mutable slice), so the parallel closure never
    // shares `data`. Nothing to fill when no rows survived or the row width collapsed to zero.
    if row_size > 0 {
        data.chunks_mut(row_size)
            .zip(kept.iter())
            .par_bridge()
            .map(|(block, &wi)| -> Result<()> {
                let w = &windows[wi];
                for (c, &idx) in group.indices().iter().enumerate() {
                    let proxy = SignalProxy::new(file.file().clone(), idx)?;
                    let row = &mut block[c * n..(c + 1) * n];
                    for run in &w.runs {
                        let len = run.len();
                        proxy.read_physical(
                            run.s_start,
                            run.s_start + len,
                            &mut row[run.dest..run.dest + len],
                        )?;
                    }
                    if !edge || w.runs.is_empty() {
                        continue;
                    }
                    // Hold the nearest real sample across every hole: the leading run's first
                    // value before the first run, then the preceding run's last value.
                    let lead = row[w.runs[0].dest];
                    row[..w.runs[0].dest].fill(lead);
                    for pair in w.runs.windows(2) {
                        let (prev, next) = (pair[0], pair[1]);
                        let prev_end = prev.dest + prev.len();
                        let held = row[prev_end - 1];
                        row[prev_end..next.dest].fill(held);
                    }
                    let last = w.runs[w.runs.len() - 1];
                    let last_end = last.dest + last.len();
                    let held = row[last_end - 1];
                    row[last_end..].fill(held);
                }
                Ok(())
            })
            .collect::<Result<Vec<()>>>()?;
    }

    Ok((data, flags, dropped))
}

/// Record range covering `[start_sec, end_sec)`.
///
/// EDF+D record onsets are non-uniform, so they must be looked up rather than derived from
/// the record duration.
pub(crate) fn record_range_for_time(
    file: &EdfFile,
    start_sec: f64,
    end_sec: f64,
) -> (usize, usize) {
    let num_records = file.num_records();
    let dur = file.header().record_duration_secs;

    if file.header().variant.is_plus_d() {
        return file.file().with_annotations(|idx| {
            let onsets = &idx.record_onsets;
            if onsets.is_empty() {
                return (0, 0);
            }
            let first = onsets.partition_point(|&o| o + dur <= start_sec);
            let last = onsets.partition_point(|&o| o < end_sec);
            (first.min(num_records), last.min(num_records))
        });
    }

    if dur <= 0.0 {
        return (0, 0);
    }
    let first = ((start_sec.max(0.0) / dur) as usize).min(num_records);
    let last = (((end_sec.max(0.0) / dur).ceil()) as usize).min(num_records);
    (first, last)
}

pub(crate) fn read_samples(proxy: &SignalProxy, s_start: usize, s_end: usize) -> Result<Vec<f64>> {
    if s_start >= proxy.len() || s_start >= s_end {
        return Ok(Vec::new());
    }
    let count = s_end - s_start;
    let mut buf = vec![0.0f64; count];
    proxy.read_physical(s_start, s_end, &mut buf)?;
    Ok(buf)
}

pub(crate) fn read_digital_samples(
    proxy: &SignalProxy,
    s_start: usize,
    s_end: usize,
) -> Result<Vec<i32>> {
    if s_start >= proxy.len() || s_start >= s_end {
        return Ok(Vec::new());
    }
    let count = s_end - s_start;
    let mut buf = vec![0i32; count];
    proxy.read_digital(s_start, s_end, &mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::SignalGroup;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// EDF+C, `num_signals` channels at `rate` Hz, record duration 1s, n = 10 * num_records
    /// samples. Signal i holds value (i + 1) * g for global flat sample g. Physical range
    /// [-32768, 32767] over digital [-32768, 32767] gives gain exactly 1.
    pub(super) fn write_contig(path: &str, num_records: usize, rate: f64, num_signals: usize) {
        let spr = (rate.round().max(1.0)) as usize;
        let total = num_records * spr;
        let mut buf: Vec<u8> = vec![b' '; 256 + 256 * num_signals];
        let mut put = |off: usize, s: &str| {
            let b = s.as_bytes();
            let n = b.len().min(256);
            buf[off..off + n].copy_from_slice(&b[..n]);
        };
        put(0, "0");
        put(8, "X X X X");
        put(88, "Startdate X X X X");
        put(168, "01.01.00");
        put(176, "00.00.00");
        put(184, &(256 + 256 * num_signals).to_string());
        put(192, "EDF+C");
        put(236, &num_records.to_string());
        put(244, "1");
        put(252, &num_signals.to_string());
        // EDF signal headers are stored as one block per field; signal i's field for a field
        // whose block starts at per-signal offset `fo` (stride `w`) lives at
        // `256 + fo * num_signals + w * i`.
        let put_field = |buf: &mut Vec<u8>, ns: usize, fo: usize, w: usize, i: usize, s: &str| {
            let off = 256 + fo * ns + w * i;
            let b = s.as_bytes();
            let n = b.len().min(w);
            buf[off..off + n].copy_from_slice(&b[..n]);
        };
        for i in 0..num_signals {
            put_field(&mut buf, num_signals, 0, 16, i, &format!("Sig{i}"));
            put_field(&mut buf, num_signals, 96, 8, i, "uV");
            put_field(&mut buf, num_signals, 104, 8, i, "-32768");
            put_field(&mut buf, num_signals, 112, 8, i, "32767");
            put_field(&mut buf, num_signals, 120, 8, i, "-32768");
            put_field(&mut buf, num_signals, 128, 8, i, "32767");
            put_field(&mut buf, num_signals, 216, 8, i, &spr.to_string());
        }
        for g in 0..total {
            for i in 0..num_signals {
                let v =
                    (((i + 1) * g) as i64).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    fn group_of(file: &crate::file::EdfFile, indices: &[usize]) -> SignalGroup {
        SignalGroup::from_indices(file.header(), indices).unwrap()
    }

    #[test]
    fn contiguous_window_math() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let plan = plan_epochs(&file, &group, &[1.0, 3.4, 8.7], 0.5, 0.5).unwrap();
        let (windows, flags) = (&plan.windows, &plan.valid);
        assert_eq!(plan.n_samples, 10);
        assert_eq!(windows[0].s_start, 5);
        assert_eq!(windows[0].s_end, 15);
        assert_eq!(windows[1].s_start, 29);
        assert_eq!(windows[1].s_end, 39);
        assert_eq!(windows[2].s_start, 82);
        assert_eq!(windows[2].s_end, 92);
        assert_eq!(flags, &vec![true, true, true]);
    }

    /// EDF+D, 2 ordinary signals at 10 Hz plus one annotation channel carrying only
    /// time-keeping TALs at exactly `onsets` (which must start at 0.0).
    fn write_plus_d(path: &str, onsets: &[f64]) {
        let num_signals = 3usize;
        let spr = 10usize;
        let ann_samples = 58usize;
        let mut buf: Vec<u8> = vec![b' '; 256 + 256 * num_signals];
        let mut put = |off: usize, s: &str| {
            let b = s.as_bytes();
            let n = b.len().min(256);
            buf[off..off + n].copy_from_slice(&b[..n]);
        };
        put(0, "0");
        put(8, "X X X X");
        put(88, "Startdate X X X X");
        put(168, "01.01.00");
        put(176, "00.00.00");
        put(184, &(256 + 256 * num_signals).to_string());
        put(192, "EDF+D");
        put(236, &onsets.len().to_string());
        put(244, "1");
        put(252, &num_signals.to_string());
        let put_field = |buf: &mut Vec<u8>, ns: usize, fo: usize, w: usize, i: usize, s: &str| {
            let off = 256 + fo * ns + w * i;
            let b = s.as_bytes();
            let n = b.len().min(w);
            buf[off..off + n].copy_from_slice(&b[..n]);
        };
        for i in 0..num_signals {
            let label = if i < 2 { "Sig" } else { "EDF Annotations" };
            let samples = if i < 2 { spr } else { ann_samples };
            put_field(&mut buf, num_signals, 0, 16, i, label);
            put_field(&mut buf, num_signals, 96, 8, i, "uV");
            put_field(&mut buf, num_signals, 104, 8, i, "-32768");
            put_field(&mut buf, num_signals, 112, 8, i, "32767");
            put_field(&mut buf, num_signals, 120, 8, i, "-32768");
            put_field(&mut buf, num_signals, 128, 8, i, "32767");
            put_field(&mut buf, num_signals, 216, 8, i, &samples.to_string());
        }
        let ann_bytes = ann_samples * 2;
        for (r, &onset) in onsets.iter().enumerate() {
            for i in 0..2 {
                let v = (((i + 1) * r * 10) as i64).clamp(i64::from(i16::MIN), i64::from(i16::MAX))
                    as i16;
                for s in 0..spr {
                    buf.extend_from_slice(&(v + s as i16).to_le_bytes());
                }
            }
            let tal = format!("+{onset}\u{14}\u{14}\0");
            let mut ann = vec![0u8; ann_bytes];
            ann[..tal.len()].copy_from_slice(tal.as_bytes());
            buf.extend_from_slice(&ann);
        }
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    fn open(path: &std::path::Path) -> crate::file::EdfFile {
        let file = crate::file::EdfFile::open(path).unwrap();
        // Blocks until the (background or deferred) annotation index exists, so the
        // onset table is populated before planning.
        file.wait_for_annotations();
        file
    }

    #[test]
    fn edge_epoch_marked_invalid() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let plan = plan_epochs(&file, &group, &[0.0, 5.0, 9.9], 1.0, 1.0).unwrap();
        assert_eq!(plan.valid, vec![false, true, false]);
    }

    #[test]
    fn plus_d_gap_straddle_marked_invalid() {
        let f = NamedTempFile::new().unwrap();
        write_plus_d(f.path().to_str().unwrap(), &[0.0, 1.0, 5.0, 6.0]);
        let file = open(f.path());
        let group = group_of(&file, &[0, 1]);
        // Window around 1.5 is [1.0, 2.0]: ends exactly where the gap starts: fine.
        // Around 5.2 is [4.7, 5.7]: overlaps the gap [2.0, 5.0): invalid.
        // Around 6.5 is [6.0, 7.0]: inside the last record: fine.
        let plan = plan_epochs(&file, &group, &[1.5, 5.2, 6.5], 0.5, 0.5).unwrap();
        assert_eq!(plan.valid, vec![true, false, true]);
    }

    /// A window covering real data on both sides of a gap must keep each side at its own time
    /// offset, with the gap columns padded. Closing the two runs up against each other would
    /// present samples 3 s apart as neighbours.
    #[test]
    fn gap_spanning_window_keeps_both_sides_in_place() {
        let f = NamedTempFile::new().unwrap();
        // Records at 0,1 then a 3 s gap, then 5,6. 10 Hz.
        write_plus_d(f.path().to_str().unwrap(), &[0.0, 1.0, 5.0, 6.0]);
        let file = open(f.path());
        let group = group_of(&file, &[0, 1]);
        // [1.5, 5.5) spans the gap [2.0, 5.0): 5 real samples, 30 gap columns, 5 real samples.
        let plan = plan_epochs(&file, &group, &[3.5], 2.0, 2.0).unwrap();
        assert_eq!(plan.n_samples, 40);
        assert_eq!(plan.valid, vec![false]);
        let w = &plan.windows[0];
        assert_eq!(w.runs.len(), 2);
        assert_eq!(w.runs[0].dest, 0);
        assert_eq!(w.runs[0].len(), 5);
        assert_eq!(w.runs[1].dest, 35);
        assert_eq!(w.runs[1].len(), 5);

        let (data, _, _) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Nan)).unwrap();
        let row = &data[0..40];
        assert!(row[..5].iter().all(|v| !v.is_nan()));
        assert!(row[5..35].iter().all(|v| v.is_nan()));
        assert!(row[35..].iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn extract_rows_match_direct_reads() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let events = [1.0, 3.4, 8.7];
        let plan = plan_epochs(&file, &group, &events, 0.5, 0.5).unwrap();
        let (data, valid, dropped) = extract_epochs(&file, &group, &plan, EpochPad::Drop).unwrap();
        assert_eq!(valid, vec![true, true, true]);
        assert!(dropped.is_empty());
        let n = plan.n_samples;
        let nc = group.indices().len();
        for (e, w) in plan.windows.iter().enumerate() {
            for (c, &idx) in group.indices().iter().enumerate() {
                let p = file.signal(idx).unwrap();
                let want = p
                    .read_at(w.s_start as f64 / 10.0, w.s_end as f64 / 10.0)
                    .unwrap();
                let row = &data[(e * nc + c) * n..(e * nc + c) * n + n];
                assert_eq!(row, want.as_slice(), "epoch {e} channel {c}");
            }
        }
    }

    #[test]
    fn drop_compacts_and_records_indices() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let plan = plan_epochs(&file, &group, &[0.0, 5.0, 9.9], 2.0, 2.0).unwrap();
        let (data, valid, dropped) = extract_epochs(&file, &group, &plan, EpochPad::Drop).unwrap();
        assert_eq!(dropped, vec![0, 2]);
        assert_eq!(valid, vec![true]);
        assert_eq!(data.len(), 2 * 40); // one kept epoch, 2 channels, window = 4 s * 10 Hz
    }

    /// Row width and pad placement must come from `pre`/`post` alone. When every epoch is
    /// edge-clipped there is no full-width row to infer the width from, and a width taken from
    /// the decoded spans would drop the leading pad and misalign the rows against each other.
    #[test]
    fn all_clipped_rows_keep_nominal_width_and_alignment() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let plan = plan_epochs(&file, &group, &[1.0, 9.0], 2.0, 2.0).unwrap();
        assert_eq!(plan.valid, vec![false, false]);
        assert_eq!(plan.n_samples, 40);
        let (data, _, _) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Nan)).unwrap();
        assert_eq!(data.len(), 2 * 2 * 40);
        // Event 1.0 spans [-1.0, 3.0): the first 1 s of columns is off the file.
        let row0 = &data[0..40];
        assert!(row0[..10].iter().all(|v| v.is_nan()));
        assert!(row0[10..].iter().all(|v| !v.is_nan()));
        // Event 9.0 spans [7.0, 11.0): the last 1 s of columns is off the file.
        let row1 = &data[2 * 40..3 * 40];
        assert!(row1[..30].iter().all(|v| !v.is_nan()));
        assert!(row1[30..].iter().all(|v| v.is_nan()));
    }

    /// With everything dropped the block is empty but still nominally shaped.
    #[test]
    fn drop_everything_keeps_nominal_width() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let plan = plan_epochs(&file, &group, &[1.0], 2.0, 2.0).unwrap();
        let (data, valid, dropped) = extract_epochs(&file, &group, &plan, EpochPad::Drop).unwrap();
        assert_eq!(plan.n_samples, 40);
        assert!(valid.is_empty());
        assert_eq!(dropped, vec![0]);
        assert!(data.is_empty());
    }

    #[test]
    fn nan_fill_marks_invalid_rows() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 2);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0, 1]);
        let plan = plan_epochs(&file, &group, &[0.0, 5.0], 2.0, 2.0).unwrap();
        let (data, valid, dropped) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Nan)).unwrap();
        assert!(dropped.is_empty());
        assert_eq!(valid, vec![false, true]);
        // Nominal window width: 4 s * 10 Hz.
        let n = plan.n_samples;
        assert_eq!(n, 40);
        let e0c0 = &data[0..n];
        // Event 0.0 with pre=2.0: the first 2.0 * rate columns fall off the file start.
        assert!(e0c0[..20].iter().all(|v| v.is_nan()));
        assert!(e0c0[20..].iter().all(|v| !v.is_nan()));
    }
}
