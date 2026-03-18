use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::annotation::AnnotationIndex;
use crate::error::{EdfError, Result};
use crate::header::EdfHeader;
use crate::record::RecordLayout;

/// A memory-mapped EDF file with parsed header, record layout, and annotation index.
///
/// On open, the file is mapped into memory and scanned sequentially to parse
/// the header and build the annotation index. The mmap remains alive for the
/// lifetime of this struct, backing all data access through the proxy layer.
pub struct MappedFile {
    mmap: Mmap,
    pub header: EdfHeader,
    pub layout: RecordLayout,
    pub annotations: AnnotationIndex,
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.mmap.len())
            .field("header", &self.header)
            .field("layout", &self.layout)
            .finish()
    }
}

impl MappedFile {
    /// Open and parse an EDF/EDF+ file.
    ///
    /// This performs a sequential scan of the entire file to build the annotation
    /// index and record-to-time map. On platforms that support it, madvise hints
    /// are used to optimize I/O: sequential during the scan, then random for
    /// subsequent access.
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        let file = std::fs::File::open(path).map_err(|e| EdfError::FileOpen {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| EdfError::MmapFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential).ok();

        let header = EdfHeader::parse(&mmap)?;
        let layout = RecordLayout::from_header(&header);
        let annotations = AnnotationIndex::build(&mmap, &header, &layout)?;

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random).ok();

        Ok(Arc::new(MappedFile {
            mmap,
            header,
            layout,
            annotations,
        }))
    }

    /// Raw bytes of the entire file (the mmap backing).
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Extract the raw bytes of a single data record.
    pub fn record_bytes(&self, rec_idx: usize) -> Result<&[u8]> {
        let num_records = self.header.num_records.max(0) as usize;
        if rec_idx >= num_records {
            return Err(EdfError::RecordOutOfRange {
                index: rec_idx,
                count: num_records,
            });
        }
        let start = self.header.data_offset() + rec_idx * self.layout.record_size;
        let end = start + self.layout.record_size;
        self.mmap.get(start..end).ok_or(EdfError::RecordOutOfRange {
            index: rec_idx,
            count: num_records,
        })
    }

    /// Hint to the OS that we'll soon need the bytes for the given record range.
    #[cfg(unix)]
    pub fn advise_willneed(&self, start_record: usize, end_record: usize) {
        let data_offset = self.header.data_offset();
        let byte_start = data_offset + start_record * self.layout.record_size;
        let byte_end = data_offset + end_record * self.layout.record_size;
        let byte_end = byte_end.min(self.mmap.len());
        // Mmap::advise_range is not available on all versions;
        // fall back to advising the whole region is acceptable.
        if byte_start < byte_end
            && let Some(slice) = self.mmap.get(byte_start..byte_end)
        {
            let _ = unsafe {
                libc::madvise(
                    slice.as_ptr() as *mut libc::c_void,
                    byte_end - byte_start,
                    libc::MADV_WILLNEED,
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn build_test_file() -> (NamedTempFile, usize, usize) {
        let num_signals = 1;
        let header_bytes = 256 + 256 * num_signals;
        let num_records = 3;
        let samples_per_record = 4;
        let record_size = samples_per_record * 2;

        let mut buf = vec![b' '; header_bytes];
        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, "");
        write_field(&mut buf, 236, 8, &num_records.to_string());
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, &num_signals.to_string());

        let sig_data = &mut buf[256..];
        write_sig(sig_data, 0, 1, 0, 16, "EEG");
        write_sig(sig_data, 0, 1, 16, 80, "");
        write_sig(sig_data, 0, 1, 96, 8, "uV");
        write_sig(sig_data, 0, 1, 104, 8, "-3200");
        write_sig(sig_data, 0, 1, 112, 8, "3200");
        write_sig(sig_data, 0, 1, 120, 8, "-32768");
        write_sig(sig_data, 0, 1, 128, 8, "32767");
        write_sig(sig_data, 0, 1, 136, 80, "");
        write_sig(sig_data, 0, 1, 216, 8, &samples_per_record.to_string());
        write_sig(sig_data, 0, 1, 224, 32, "");

        for rec in 0..num_records {
            for sample in 0..samples_per_record {
                let val = (rec * samples_per_record + sample) as i16;
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        (file, samples_per_record, record_size)
    }

    #[test]
    fn open_and_read_records() {
        let (file, _, record_size) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();

        assert_eq!(mapped.header.num_signals, 1);
        assert_eq!(mapped.header.num_records, 3);
        assert_eq!(mapped.layout.record_size, record_size);

        let rec0 = mapped.record_bytes(0).unwrap();
        assert_eq!(rec0.len(), record_size);

        let rec2 = mapped.record_bytes(2).unwrap();
        let first_sample = i16::from_le_bytes([rec2[0], rec2[1]]);
        assert_eq!(first_sample, 8);
    }

    #[test]
    fn record_out_of_range() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        assert!(mapped.record_bytes(3).is_err());
    }

    #[test]
    fn open_nonexistent_file() {
        let result = MappedFile::open(Path::new("/tmp/nonexistent_edf_file.edf"));
        assert!(matches!(result.unwrap_err(), EdfError::FileOpen { .. }));
    }

    fn write_field(buf: &mut [u8], offset: usize, size: usize, value: &str) {
        let bytes = value.as_bytes();
        let len = bytes.len().min(size);
        buf[offset..offset + len].copy_from_slice(&bytes[..len]);
    }

    fn write_sig(
        data: &mut [u8],
        index: usize,
        num_signals: usize,
        field_offset: usize,
        field_size: usize,
        value: &str,
    ) {
        let start = field_offset * num_signals + field_size * index;
        let bytes = value.as_bytes();
        let len = bytes.len().min(field_size);
        data[start..start + len].copy_from_slice(&bytes[..len]);
    }
}
