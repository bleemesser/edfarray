"""Page-cache eviction for cold-read benchmarking.

Warm-cache numbers hide the access pattern entirely: an mmap read that touches every page of a
file looks identical to one that touches a hundredth of them, once the file is resident. These
helpers evict a single file so first-touch cost is visible.

Both platform paths are unprivileged and target one file, so they do not disturb the rest of
the machine.
"""

import ctypes
import ctypes.util
import os
import platform

MS_INVALIDATE = 2
PROT_READ = 1
MAP_SHARED = 1


def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def evict_linux(path):
    """Drop a file's page-cache pages via posix_fadvise."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def evict_darwin(path):
    """Drop a file's pages by invalidating a shared mapping of it.

    `purge` needs root, but msync(MS_INVALIDATE) over a MAP_SHARED mapping drops the mapped
    file's clean pages and needs no privileges.
    """
    libc = _libc()
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
    ]
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if not addr:
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            if libc.msync(ctypes.c_void_p(addr), ctypes.c_size_t(size), MS_INVALIDATE) != 0:
                raise OSError(ctypes.get_errno(), "msync(MS_INVALIDATE) failed")
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def residency(path, samples=64):
    """Fraction of a file's pages currently resident, sampled. For verifying eviction."""
    libc = _libc()
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
    ]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        page = os.sysconf("SC_PAGESIZE")
        pages = max(1, size // page)
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if not addr:
            return 1.0
        try:
            vec = (ctypes.c_char * 1)()
            resident = checked = 0
            for i in range(0, pages, max(1, pages // samples)):
                if libc.mincore(ctypes.c_void_p(addr + i * page), page, vec) == 0:
                    resident += vec[0][0] & 1
                    checked += 1
            return resident / checked if checked else 1.0
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def evict(path):
    """Evict `path` from the page cache. Returns the method used."""
    system = platform.system()
    if system == "Linux":
        evict_linux(path)
        return "posix_fadvise"
    if system == "Darwin":
        evict_darwin(path)
        return "msync(MS_INVALIDATE)"
    raise NotImplementedError(f"no eviction method for {system}")
