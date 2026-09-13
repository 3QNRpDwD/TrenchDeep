//! Requested allocation bytes, not RSS. Counters never allocate or log.
use serde::Serialize;
#[cfg(feature = "benchmarkAlloc")]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

#[derive(Clone, Copy, Default, Serialize)]
pub struct Snapshot {
    pub allocations: usize,
    pub reallocations: usize,
    pub deallocations: usize,
    pub requested_bytes: usize,
    pub live_bytes: usize,
    pub peak_live_bytes: usize,
}
#[cfg(feature = "benchmarkAlloc")]
static ALLOC: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "benchmarkAlloc")]
static REALLOC: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "benchmarkAlloc")]
static DEALLOC: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "benchmarkAlloc")]
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "benchmarkAlloc")]
static LIVE: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "benchmarkAlloc")]
static PEAK: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "benchmarkAlloc")]
pub struct CountingAllocator;
#[cfg(feature = "benchmarkAlloc")]
fn acquired(size: usize) {
    REQUESTED.fetch_add(size, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}
#[cfg(feature = "benchmarkAlloc")]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forward the caller's layout unchanged to System.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOC.fetch_add(1, Relaxed);
            acquired(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same allocator/layout contract as alloc.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            ALLOC.fetch_add(1, Relaxed);
            acquired(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout are the original allocation supplied by caller.
        unsafe { System.dealloc(ptr, layout) };
        DEALLOC.fetch_add(1, Relaxed);
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: System owns ptr; preserve its layout and requested new size.
        let new_ptr = unsafe { System.realloc(ptr, layout, size) };
        if !new_ptr.is_null() {
            REALLOC.fetch_add(1, Relaxed);
            LIVE.fetch_sub(layout.size(), Relaxed);
            acquired(size);
        }
        // A failed realloc leaves the old allocation and live bytes unchanged.
        new_ptr
    }
}
pub fn snapshot() -> Snapshot {
    #[cfg(feature = "benchmarkAlloc")]
    return Snapshot {
        allocations: ALLOC.load(Relaxed),
        reallocations: REALLOC.load(Relaxed),
        deallocations: DEALLOC.load(Relaxed),
        requested_bytes: REQUESTED.load(Relaxed),
        live_bytes: LIVE.load(Relaxed),
        peak_live_bytes: PEAK.load(Relaxed),
    };
    #[cfg(not(feature = "benchmarkAlloc"))]
    Snapshot::default()
}
pub fn begin() -> Snapshot {
    #[cfg(feature = "benchmarkAlloc")]
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    snapshot()
}
pub fn delta(start: Snapshot) -> Snapshot {
    let end = snapshot();
    Snapshot {
        allocations: end.allocations - start.allocations,
        reallocations: end.reallocations - start.reallocations,
        deallocations: end.deallocations - start.deallocations,
        requested_bytes: end.requested_bytes - start.requested_bytes,
        live_bytes: end.live_bytes,
        peak_live_bytes: end.peak_live_bytes,
    }
}
