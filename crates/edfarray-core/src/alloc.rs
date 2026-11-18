use crate::error::{EdfError, Result};

/// Fill a `Vec` with `len` copies of `value`, or return [`EdfError::MemoryRequest`] naming
/// `name` if the allocator refuses.
pub(crate) fn filled_vec<T: Clone>(len: usize, value: T, name: &'static str) -> Result<Vec<T>> {
    let bytes = len as u128 * std::mem::size_of::<T>() as u128;
    let mut buf: Vec<T> = Vec::new();
    if buf.try_reserve_exact(len).is_err() {
        return Err(EdfError::MemoryRequest {
            name,
            elements: len,
            bytes,
        });
    }
    buf.resize(len, value);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_allocation_succeeds() {
        let buf = filled_vec(4, 1.0f64, "test").unwrap();
        assert_eq!(buf, vec![1.0; 4]);
    }

    #[test]
    fn empty_allocation_succeeds() {
        assert!(filled_vec(0, 1.0f64, "test").unwrap().is_empty());
    }
}
