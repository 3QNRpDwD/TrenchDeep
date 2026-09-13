#![cfg(feature = "benchmarkAlloc")]
#[path = "../benches/support/counting_allocator.rs"]
mod allocation;

#[test]
fn allocator_tracks_zeroed_resize_and_release() {
    use std::alloc::{GlobalAlloc, Layout};
    let allocator = allocation::CountingAllocator;
    let start = allocation::begin();
    let small = Layout::from_size_align(16, 8).unwrap();
    // SAFETY: all pointers are obtained from this allocator, accessed within
    // their allocated lengths, and released exactly once with matching layout.
    unsafe {
        let ptr = allocator.alloc_zeroed(small);
        assert!(!ptr.is_null());
        assert!(std::slice::from_raw_parts(ptr, 16).iter().all(|&v| v == 0));
        ptr.write(42);
        let grown = allocator.realloc(ptr, small, 64);
        assert!(!grown.is_null());
        assert_eq!(grown.read(), 42);
        let mid = allocation::delta(start);
        assert_eq!(mid.allocations, 1);
        assert_eq!(mid.reallocations, 1);
        assert_eq!(mid.requested_bytes, 80);
        assert_eq!(mid.live_bytes, start.live_bytes + 64);
        let shrunk = allocator.realloc(grown, Layout::from_size_align(64, 8).unwrap(), 8);
        assert!(!shrunk.is_null());
        assert_eq!(shrunk.read(), 42);
        allocator.dealloc(shrunk, Layout::from_size_align(8, 8).unwrap());
    }
    let end = allocation::delta(start);
    assert_eq!(end.live_bytes, start.live_bytes);
    assert_eq!(end.peak_live_bytes, start.live_bytes + 64);
    assert_eq!(end.requested_bytes, 88);
    assert_eq!(end.deallocations, 1);
}
