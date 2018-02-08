use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::Backend;

use relevant::Relevant;

/// Trait for types that represent a block (`Range`) of `Memory`.
pub trait Block<B: Backend>: Send + Sync + Debug {
    /// `Memory` instance of the block.
    fn memory(&self) -> &B::Memory;

    /// `Range` of the memory this block occupy.
    fn range(&self) -> Range<u64>;

    /// Get size of the block.
    #[inline]
    fn size(&self) -> u64 {
        self.range().end - self.range().start
    }

    /// Check if a block is a child of this block.
    ///
    /// ### Parameters:
    ///
    /// - `other`: potential child block
    ///
    /// ### Type parameters:
    ///
    /// - `T`: tag of potential child block
    fn contains<T>(&self, other: &T) -> bool
    where
        T: Block<B>,
    {
        use std::ptr::eq;
        eq(self.memory(), other.memory()) && self.range().start <= other.range().start
            && self.range().end >= other.range().end
    }
}

/// Tagged block of memory.
///
/// A `RawBlock` must never be silently dropped, that will result in a panic.
/// The block must be freed by returning it to the same allocator it came from.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `T`: tag type, used by allocators to track allocations
#[derive(Debug)]
pub struct RawBlock<B: Backend> {
    relevant: Relevant,
    range: Range<u64>,
    memory: *const B::Memory,
}

unsafe impl<B> Send for RawBlock<B>
where
    B: Backend,
{
}

unsafe impl<B> Sync for RawBlock<B>
where
    B: Backend,
{
}

impl<B> RawBlock<B>
where
    B: Backend,
{
    /// Construct a tagged block from `Memory` and `Range`.
    /// The given `Memory` must not be freed if there are `Block`s allocated from it that are still
    /// in use.
    ///
    /// ### Parameters:
    ///
    /// - `memory`: pointer to the actual memory for the block
    /// - `range`: range of the `memory` used by the block
    pub(crate) fn new(memory: *const B::Memory, range: Range<u64>) -> Self {
        assert!(range.start <= range.end);
        RawBlock {
            relevant: Relevant,
            memory,
            range,
        }
    }

    #[doc(hidden)]
    /// Dispose of this block.
    ///
    /// This is unsafe because the caller must ensure that the memory of the block is not used
    /// again. This will typically entail dropping some kind of resource (`Buffer` or `Image` to
    /// give some examples) that occupy this memory.
    ///
    /// ### Returns
    ///
    /// Tag value of the block
    pub unsafe fn dispose(self) {
        self.relevant.dispose();
    }
}

impl<B> Block<B> for RawBlock<B>
where
    B: Backend,
{
    /// Get memory of the block.
    #[inline]
    fn memory(&self) -> &B::Memory {
        // Has to be valid
        unsafe { &*self.memory }
    }

    /// Get memory range of the block.
    #[inline]
    fn range(&self) -> Range<u64> {
        self.range.clone()
    }
}

impl<B, T, Y> Block<B> for (T, Y)
where
    B: Backend,
    T: Block<B>,
    Y: Send + Sync + Debug,
{
    /// Get memory of the block.
    #[inline(always)]
    fn memory(&self) -> &B::Memory {
        // Has to be valid
        self.0.memory()
    }

    /// Get memory range of the block.
    #[inline(always)]
    fn range(&self) -> Range<u64> {
        self.0.range()
    }
}
