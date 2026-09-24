use rayon::prelude::*;

use crate::error::{EdfError, Result};
use crate::file::EdfFile;
use crate::grid::first_sample_at_or_after;
use crate::group::{GroupKind, PadMode, SignalGroup};
use crate::proxy::SignalProxy;

/// One contiguous piece of an epoch: flat samples `s_start..s_end` go to row column `dest`.
///
/// A window that crosses an EDF+D gap gives one run for each side. Each run sits at its true
/// time offset in the row. Thus the columns in the gap stay empty, and the pad policy fills
/// them. The runs do not close up over the gap.
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
/// crosses an EDF+D gap, it covers both sides, and `runs` gives the correct position of each
/// side. If a window has no samples, `s_start == s_end`. This value is the index of the first
/// sample after the window, or the sample count if the window is past the end of the file.
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

/// A planned set of epochs: one window for each event, a validity mask for the windows, and
/// the row width that every epoch decodes into.
///
/// `n_samples` is the nominal window width `ceil((pre + post) * rate)`. A product within 1e-6
/// of an integer counts as that integer. The width depends only on the request, not on the
/// event times. Thus the rows stay aligned in time with each other, and the output shape does
/// not change between files.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochPlan {
    pub windows: Vec<EpochWindow>,
    pub valid: Vec<bool>,
    pub n_samples: usize,
}

/// Edge and gap policy for [`extract_epochs`]. Only epochs use `Drop`. The fill variants use
/// the proxy [`PadMode`], so the two features always pad in the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpochPad {
    /// Omit epochs that touch an edge or a gap from the output. Record their indices as dropped.
    Drop,
    /// Keep every epoch. Fill missing samples with `PadMode` and mark that row `valid = false`.
    Fill(PadMode),
}

/// Error for an invalid epoch request argument. Python raises it as `ValueError`.
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

/// Append flat samples `s0..s1` at column `dest`. If both the samples and the columns continue
/// the previous run, merge them into that run.
fn push_run(runs: &mut Vec<EpochRun>, s0: usize, s1: usize, dest: usize) {
    if let Some(last) = runs.last_mut()
        && last.s_end == s0
        && last.dest + last.len() == dest
    {
        last.s_end = s1;
        return;
    }
    runs.push(EpochRun {
        s_start: s0,
        s_end: s1,
        dest,
    });
}

/// Plan event times into fixed-length sample windows. This function reads no data.
///
/// Window `i` starts at the first sample at or after `events[i] - pre` and is `n_samples`
/// wide. `valid[i]` is true only when all of those samples exist. Thus it is false when the
/// window goes past either end of the file or touches an EDF+D gap. The pad policy that the
/// caller selects later does not change it.
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

    // EDF+D record onsets are non-uniform, so they come from the annotation index.
    let onsets: Option<Vec<f64>> = file.header().variant.is_plus_d().then(|| {
        file.file()
            .with_annotations(|idx| idx.record_onsets.clone())
    });

    // Every window is placed on one sample grid: grid index `k` is time `k / rate`. A record
    // with onset `o` holds grid indices `grid(o)..grid(o) + spr`. An onset that falls between
    // grid points rounds to the nearest one, which moves that record by under half a sample.
    let record_grid: Option<Vec<i64>> = onsets.as_ref().map(|o| {
        o.iter()
            .take(num_records)
            .map(|&onset| (onset * sample_rate).round() as i64)
            .collect()
    });

    // Nominal width, derived from the request alone. Every row is this wide, and column j of
    // a row is always grid index `first + j`, where `first` is the first sample at or after
    // the window start.
    let n_samples = first_sample_at_or_after((pre + post) * sample_rate).max(1) as usize;
    let n = n_samples as i64;
    let spr_i = spr as i64;

    let mut windows = Vec::with_capacity(events.len());
    let mut valid = Vec::with_capacity(events.len());
    for &t in events {
        if !t.is_finite() {
            return Err(epoch_argument(
                "events",
                format!("event time {t} is not finite"),
            ));
        }
        let first = first_sample_at_or_after((t - pre) * sample_rate);
        let end = first.saturating_add(n);

        // Each record contributes its overlap with the window at its own column. Adjacent
        // records coalesce, so only real gaps and file boundaries leave a hole.
        let mut runs: Vec<EpochRun> = Vec::new();
        // Flat index of the first sample at or after the window, used when no run exists.
        let next_sample;
        match &record_grid {
            Some(grid) => {
                let from = grid.partition_point(|&b| b.saturating_add(spr_i) <= first);
                next_sample = from.saturating_mul(spr).min(total_samples);
                let mut filled = 0i64;
                for (r, &b) in grid.iter().enumerate().skip(from) {
                    if b >= end {
                        break;
                    }
                    // `filled` keeps runs disjoint if two records overlap in time.
                    let lo = first.max(b).max(first + filled);
                    let hi = end.min(b.saturating_add(spr_i));
                    if hi <= lo {
                        continue;
                    }
                    push_run(
                        &mut runs,
                        r * spr + (lo - b) as usize,
                        r * spr + (hi - b) as usize,
                        (lo - first) as usize,
                    );
                    filled = hi - first;
                }
            }
            None => {
                let total = i64::try_from(total_samples).unwrap_or(i64::MAX);
                let lo = first.clamp(0, total);
                let hi = end.clamp(0, total);
                next_sample = lo as usize;
                if hi > lo {
                    push_run(&mut runs, lo as usize, hi as usize, (lo - first) as usize);
                }
            }
        }

        let (s_start, s_end) = match (runs.first(), runs.last()) {
            (Some(a), Some(b)) => (a.s_start, b.s_end),
            _ => (next_sample, next_sample),
        };
        let window = EpochWindow {
            onset: t,
            s_start,
            s_end,
            runs,
        };
        // Valid means every column has a real sample behind it.
        valid.push(window.decoded() == n_samples);
        windows.push(window);
    }
    Ok(EpochPlan {
        windows,
        valid,
        n_samples,
    })
}

/// Decode planned windows into one flat `(n_epochs, n_channels, n_samples)` row-major buffer.
///
/// Returns `(data, valid, dropped)`. With [`EpochPad::Drop`], the function omits invalid
/// windows. `valid` is all true for the kept rows, and `dropped` lists the planner indices that
/// it removed. With a fill policy, the function keeps every window. An invalid row stays
/// `valid = false`, and [`PadMode`] fills every column that has no sample. These are the
/// columns past a file boundary and, for a window that crosses an EDF+D gap, the columns in the
/// gap. Each contiguous run goes to its own time offset. Thus samples from opposite sides of a
/// gap never close up against each other. A valid row is decoded in full.
///
/// The row width is `plan.n_samples`, the nominal window width. Thus every row covers the same
/// time offsets relative to its event, and the shape does not depend on the event times.
/// Bindings must not calculate the width again from `data.len()`.
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
    let edge = matches!(pad, EpochPad::Fill(PadMode::Edge));
    let nc = group.indices().len();
    let spr = group
        .samples_per_record()
        .ok_or_else(|| epoch_argument("group", "group has no samples-per-record".into()))?;
    let total_samples = file.num_records().saturating_mul(spr);
    if edge && total_samples == 0 && windows.iter().any(|w| w.runs.is_empty()) {
        return Err(EdfError::SampleOutOfRange { index: 0, count: 0 });
    }

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

    // Each task owns one row block (disjoint mutable slice), so the parallel closure never
    // shares `data`. Nothing to fill when no rows survived or the row width collapsed to zero.
    if row_size > 0 {
        data.par_chunks_mut(row_size)
            .zip(kept.par_iter())
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
                    if !edge {
                        continue;
                    }
                    // A window with no samples of its own holds the nearest sample before it,
                    // or the first sample of the file when it sits before the start.
                    if w.runs.is_empty() {
                        let at = w.s_start.saturating_sub(1).min(total_samples - 1);
                        let mut held = [0.0f64; 1];
                        proxy.read_physical(at, at + 1, &mut held)?;
                        row.fill(held[0]);
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
/// EDF+D record onsets are not uniform. Thus the function must look them up and cannot
/// calculate them from the record duration.
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

    /// EDF+C, `num_signals` channels at `rate` Hz, record duration 1s, n = round(rate) * num_records
    /// samples. Signal i holds value (i + 1) * g for global flat sample g. Physical range
    /// [-32768, 32767] over digital [-32768, 32767] gives gain exactly 1.
    pub(super) fn write_contig(path: &str, num_records: usize, rate: f64, num_signals: usize) {
        let spr = (rate.round().max(1.0)) as usize;
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
        // A record stores each signal's samples as one block, signal after signal.
        for r in 0..num_records {
            for i in 0..num_signals {
                for g in r * spr..(r + 1) * spr {
                    let v = (((i + 1) * g) as i64).clamp(i64::from(i16::MIN), i64::from(i16::MAX))
                        as i16;
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    pub(super) fn group_of(file: &crate::file::EdfFile, indices: &[usize]) -> SignalGroup {
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

    /// Write an EDF+D file with 2 ordinary signals at 10 Hz and one annotation channel. The
    /// annotation channel holds only timekeeping TALs at exactly `onsets`, which must start at
    /// 0.0.
    fn write_plus_d(path: &str, onsets: &[f64]) {
        write_gapped(path, onsets, 10);
    }

    /// Same as `write_plus_d`, with `spr` samples in each 1 s record. Signal i holds
    /// `(i + 1) * r * spr + s` at sample `s` of record `r`, so signal 0 is the flat index.
    pub(super) fn write_gapped(path: &str, onsets: &[f64], spr: usize) {
        let num_signals = 3usize;
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
                let v = (((i + 1) * r * spr) as i64).clamp(i64::from(i16::MIN), i64::from(i16::MAX))
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

    pub(super) fn open(path: &std::path::Path) -> crate::file::EdfFile {
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
    /// offset, with the gap columns padded. If the two runs close up against each other,
    /// samples 3 s apart appear as neighbors.
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

    /// Row width and pad placement must come from `pre`/`post` alone. If every epoch is clipped
    /// at an edge, no full-width row gives the width. A width taken from the decoded spans
    /// drops the leading pad and misaligns the rows against each other.
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

    /// If all epochs are dropped, the block is empty but still has the nominal shape.
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

    /// A fractional window width must not leave a never-read column in a valid row. Every
    /// valid row holds exactly `n_samples` consecutive real samples, whatever the event phase.
    #[test]
    fn fractional_width_rows_are_fully_read() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 1);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0]);
        // Width 0.25 s * 10 Hz = 2.5, so 3 columns. 9.9 needs sample 100, which does not exist.
        let plan = plan_epochs(&file, &group, &[1.0, 1.04, 1.05, 9.9], 0.125, 0.125).unwrap();
        assert_eq!(plan.n_samples, 3);
        assert_eq!(plan.valid, vec![true, true, true, false]);
        let (data, _, _) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Nan)).unwrap();
        assert_eq!(&data[0..3], &[9.0, 10.0, 11.0]);
        assert_eq!(&data[3..6], &[10.0, 11.0, 12.0]);
        assert_eq!(&data[6..9], &[10.0, 11.0, 12.0]);
        assert_eq!(&data[9..11], &[98.0, 99.0]);
        assert!(data[11].is_nan());
    }

    #[test]
    fn float_noise_does_not_widen_the_row() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 1);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0]);
        let plan = plan_epochs(&file, &group, &[5.0], 0.1, 0.2).unwrap();
        assert_eq!(plan.n_samples, 3);
    }

    /// Edge fill holds the nearest real sample even when the window has no samples of its own.
    #[test]
    fn edge_fill_without_any_samples_holds_nearest() {
        let f = NamedTempFile::new().unwrap();
        write_contig(f.path().to_str().unwrap(), 10, 10.0, 1);
        let file = crate::file::EdfFile::open(f.path()).unwrap();
        let group = group_of(&file, &[0]);
        let plan = plan_epochs(&file, &group, &[-5.0, 20.0], 0.5, 0.5).unwrap();
        assert_eq!(plan.valid, vec![false, false]);
        let (data, _, _) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Edge)).unwrap();
        assert!(data[..10].iter().all(|&v| v == 0.0));
        assert!(data[10..].iter().all(|&v| v == 99.0));

        let g = NamedTempFile::new().unwrap();
        write_plus_d(g.path().to_str().unwrap(), &[0.0, 1.0, 5.0, 6.0]);
        let file = open(g.path());
        let group = group_of(&file, &[0]);
        // [3.0, 4.0) lies inside the gap; the last sample before it is record 1's last, 19.
        let plan = plan_epochs(&file, &group, &[3.5], 0.5, 0.5).unwrap();
        assert!(plan.windows[0].runs.is_empty());
        let (data, _, _) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Edge)).unwrap();
        assert!(data.iter().all(|&v| v == 19.0));
    }

    /// A window that opens between samples inside a record and then crosses a gap keeps one
    /// column grid for both sides, and edge fill over the hole does not panic.
    #[test]
    fn off_grid_window_across_gap_stays_aligned() {
        let f = NamedTempFile::new().unwrap();
        write_plus_d(f.path().to_str().unwrap(), &[0.0, 1.0, 5.0, 6.0]);
        let file = open(f.path());
        let group = group_of(&file, &[0]);
        // Window [1.55, 5.55): first sample at or after 1.55 is grid index 16.
        let plan = plan_epochs(&file, &group, &[3.55], 2.0, 2.0).unwrap();
        let w = &plan.windows[0];
        assert_eq!(plan.n_samples, 40);
        assert_eq!(w.runs.len(), 2);
        assert_eq!((w.runs[0].dest, w.runs[0].len()), (0, 4));
        assert_eq!((w.runs[1].dest, w.runs[1].len()), (34, 6));
        let (data, _, _) =
            extract_epochs(&file, &group, &plan, EpochPad::Fill(PadMode::Edge)).unwrap();
        assert_eq!(&data[..4], &[16.0, 17.0, 18.0, 19.0]);
        assert!(data[4..34].iter().all(|&v| v == 19.0));
        assert_eq!(&data[34..], &[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
    }
}

#[cfg(test)]
mod proptest_model {
    use super::tests::{group_of, open, write_contig, write_gapped};
    use super::*;
    use crate::grid::GRID_EPS;
    use proptest::prelude::*;
    use tempfile::NamedTempFile;

    /// Smallest integer at or above `x`, within the grid tolerance. This is a slow search on
    /// purpose, not a copy of `first_sample_at_or_after`.
    fn first_at_or_after(x: f64) -> i64 {
        let mut k = x.floor() as i64 - 2;
        while (k as f64) < x - GRID_EPS {
            k += 1;
        }
        k
    }

    /// Flat sample behind grid index `k`, if any. `starts[r]` is the first grid index of record r.
    fn sample_at(starts: &[i64], spr: i64, k: i64) -> Option<usize> {
        starts
            .iter()
            .position(|&b| k >= b && k < b + spr)
            .map(|r| r * spr as usize + (k - starts[r]) as usize)
    }

    /// Column-by-column expected rows for channel 0, whose value is its flat sample index.
    fn model_rows(
        starts: &[i64],
        spr: i64,
        events: &[f64],
        pre: f64,
        post: f64,
    ) -> (usize, Vec<Vec<Option<usize>>>) {
        let rate = spr as f64;
        let n = first_at_or_after((pre + post) * rate).max(1) as usize;
        let rows = events
            .iter()
            .map(|&t| {
                let first = first_at_or_after((t - pre) * rate);
                (0..n as i64)
                    .map(|j| sample_at(starts, spr, first + j))
                    .collect()
            })
            .collect();
        (n, rows)
    }

    fn check_against_model(
        file: &EdfFile,
        starts: &[i64],
        spr: usize,
        events: &[f64],
        pre: f64,
        post: f64,
        value: impl Fn(usize, usize) -> f64,
    ) -> std::result::Result<(), TestCaseError> {
        let group = group_of(file, &[0, 1]);
        let (n, rows) = model_rows(starts, spr as i64, events, pre, post);
        let plan = plan_epochs(file, &group, events, pre, post).unwrap();
        prop_assert_eq!(plan.n_samples, n);
        let model_valid: Vec<bool> = rows.iter().map(|r| r.iter().all(Option::is_some)).collect();
        prop_assert_eq!(&plan.valid, &model_valid);

        let total = file.num_records() * spr;
        for w in &plan.windows {
            prop_assert!(w.s_start <= w.s_end && w.s_end <= total);
            let mut col = 0;
            for run in &w.runs {
                prop_assert!(!run.is_empty());
                prop_assert!(run.dest >= col, "runs overlap or are out of order");
                prop_assert!(run.s_start >= w.s_start && run.s_end <= w.s_end);
                col = run.dest + run.len();
            }
            prop_assert!(col <= n);
        }

        let (nan, flags, dropped) =
            extract_epochs(file, &group, &plan, EpochPad::Fill(PadMode::Nan)).unwrap();
        prop_assert!(dropped.is_empty());
        prop_assert_eq!(&flags, &model_valid);
        for (e, row) in rows.iter().enumerate() {
            for c in 0..2 {
                let got = &nan[(e * 2 + c) * n..(e * 2 + c + 1) * n];
                for (j, want) in row.iter().enumerate() {
                    match want {
                        Some(g) => prop_assert_eq!(got[j], value(c, *g)),
                        None => prop_assert!(got[j].is_nan()),
                    }
                }
            }
        }

        let (kept, flags, dropped) = extract_epochs(file, &group, &plan, EpochPad::Drop).unwrap();
        let want_dropped: Vec<usize> = (0..rows.len()).filter(|&i| !model_valid[i]).collect();
        prop_assert_eq!(&dropped, &want_dropped);
        prop_assert!(flags.iter().all(|&v| v));
        let want_kept: Vec<f64> = (0..rows.len())
            .filter(|&i| model_valid[i])
            .flat_map(|i| nan[i * 2 * n..(i + 1) * 2 * n].to_vec())
            .collect();
        prop_assert_eq!(kept, want_kept);

        // Edge fill agrees with NaN fill on every real column and holds the nearest real
        // sample across each hole: the previous real column, or the next one when leading.
        let (edge, _, _) =
            extract_epochs(file, &group, &plan, EpochPad::Fill(PadMode::Edge)).unwrap();
        for (e, row) in rows.iter().enumerate() {
            let got = &edge[e * 2 * n..e * 2 * n + n];
            let first_real = row.iter().position(Option::is_some);
            let mut held = None;
            for (j, want) in row.iter().enumerate() {
                if let Some(g) = want {
                    held = Some(*g as f64);
                }
                let expect = match (held, first_real) {
                    (Some(h), _) => Some(h),
                    (None, Some(f)) => row[f].map(|g| g as f64),
                    (None, None) => None,
                };
                match expect {
                    Some(v) => prop_assert_eq!(got[j], v),
                    None => prop_assert!(got[j].is_finite()),
                }
            }
        }

        let raised = extract_epochs(file, &group, &plan, EpochPad::Fill(PadMode::Raise));
        prop_assert_eq!(raised.is_ok(), model_valid.iter().all(|&v| v));
        Ok(())
    }

    /// Times on a fine lattice around the file, plus points near the grid that differ only by
    /// floating-point error.
    fn event_strategy(span: f64) -> impl Strategy<Value = f64> {
        prop_oneof![
            (-3.0..span + 3.0),
            (-30i32..(span as i32 + 3) * 10).prop_map(|k| f64::from(k) * 0.1),
            (-30i32..(span as i32 + 3) * 10).prop_map(|k| f64::from(k) * 0.1 + 1e-12),
            (-30i32..(span as i32 + 3) * 10).prop_map(|k| f64::from(k) * 0.1 - 1e-12),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn contiguous_matches_model(
            spr in 1usize..=24,
            records in 1usize..=6,
            events in proptest::collection::vec(event_strategy(6.0), 0..8),
            pre in 0.0f64..2.5,
            post in 0.0f64..2.5,
        ) {
            prop_assume!(pre + post > 0.0);
            let f = NamedTempFile::new().unwrap();
            write_contig(f.path().to_str().unwrap(), records, spr as f64, 2);
            let file = crate::file::EdfFile::open(f.path()).unwrap();
            let starts: Vec<i64> = (0..records).map(|r| (r * spr) as i64).collect();
            check_against_model(&file, &starts, spr, &events, pre, post, |c, g| {
                ((c + 1) * g) as f64
            })?;
        }

        #[test]
        fn gapped_matches_model(
            spr in 1usize..=24,
            gaps in proptest::collection::vec(0u32..=3, 0..6),
            events in proptest::collection::vec(event_strategy(20.0), 0..8),
            pre in 0.0f64..4.0,
            post in 0.0f64..4.0,
        ) {
            prop_assume!(pre + post > 0.0);
            let mut onsets = vec![0.0f64];
            for g in &gaps {
                onsets.push(onsets[onsets.len() - 1] + 1.0 + f64::from(*g));
            }
            let f = NamedTempFile::new().unwrap();
            write_gapped(f.path().to_str().unwrap(), &onsets, spr);
            let file = open(f.path());
            let starts: Vec<i64> = onsets.iter().map(|o| *o as i64 * spr as i64).collect();
            check_against_model(&file, &starts, spr, &events, pre, post, |c, g| {
                ((c + 1) * (g / spr) * spr + g % spr) as f64
            })?;
        }

        /// A time range resolves to the samples whose time lies in `[start, end)`, on both
        /// the uniform path and the onset-table path.
        #[test]
        fn sample_range_matches_model(
            spr in 1usize..=24,
            gaps in proptest::collection::vec(0u32..=3, 0..6),
            a in event_strategy(20.0),
            b in event_strategy(20.0),
        ) {
            let mut onsets = vec![0.0f64];
            for g in &gaps {
                onsets.push(onsets[onsets.len() - 1] + 1.0 + f64::from(*g));
            }
            let f = NamedTempFile::new().unwrap();
            write_gapped(f.path().to_str().unwrap(), &onsets, spr);
            let file = open(f.path());
            let proxy = file.signal(0).unwrap();
            let rate = spr as f64;
            let (lo, hi) = (first_at_or_after(a.max(0.0) * rate), first_at_or_after(b.max(0.0) * rate));
            let inside: Vec<usize> = onsets
                .iter()
                .enumerate()
                .flat_map(|(r, o)| (0..spr).map(move |s| (r * spr + s, *o as i64 * spr as i64 + s as i64)))
                .filter(|&(_, k)| k >= lo && k < hi)
                .map(|(g, _)| g)
                .collect();
            let (s_start, s_end) = proxy.sample_range_for_time(a, b);
            match (inside.first(), inside.last()) {
                (Some(&first), Some(&last)) => prop_assert_eq!((s_start, s_end), (first, last + 1)),
                _ => prop_assert!(s_start >= s_end, "expected empty, got {}..{}", s_start, s_end),
            }

            let g = NamedTempFile::new().unwrap();
            write_contig(g.path().to_str().unwrap(), onsets.len(), rate, 1);
            let contig = crate::file::EdfFile::open(g.path()).unwrap();
            let total = (onsets.len() * spr) as i64;
            let (u_start, u_end) = contig.signal(0).unwrap().sample_range_for_time(a, b);
            prop_assert_eq!(u_start as i64, lo.clamp(0, total));
            prop_assert_eq!(u_end as i64, hi.clamp(0, total));
        }
    }
}
