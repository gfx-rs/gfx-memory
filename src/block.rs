use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;

use relevant::Relevant;

/// Trait for types that represent a block (`Range`) of `Memory`.
pub trait Block: Send + Sync + Debug {
    /// Memory type
    type Memory: Debug + Any;

    /// `Memory` instance of the block.
    fn memory(&self) -> &Self::Memory;

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
    /// - `T`: block with same memory type.
    fn contains<T>(&self, other: &T) -> bool
    where
        T: Block<Memory = Self::Memory>,
    {
        use std::ptr::eq;
        eq(self.memory(), other.memory())
            && self.range().start <= other.range().start
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
/// - `M`: hal memory type.
#[derive(Debug)]
pub struct RawBlock<M> {
    relevant: Relevant,
    range: Range<u64>,
    memory: *const M,
}

unsafe impl<M> Send for RawBlock<M> {}

unsafe impl<M> Sync for RawBlock<M> {}

impl<M> RawBlock<M> {
    /// Construct a tagged block from `Memory` and `Range`.
    /// The given `Memory` must not be freed if there are `Block`s allocated from it that are still
    /// in use.
    ///
    /// ### Parameters:
    ///
    /// - `memory`: pointer to the actual memory for the block
    /// - `range`: range of the `memory` used by the block
    pub(crate) fn new(memory: *const M, range: Range<u64>) -> Self {
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

impl<M> Block for RawBlock<M>
where
    M: Debug + Any,
{
    type Memory = M;

    #[inline]
    fn memory(&self) -> &M {
        // Has to be valid
        unsafe { &*self.memory }
    }

    #[inline]
    fn range(&self) -> Range<u64> {
        self.range.clone()
    }
}

impl<T, Y> Block for (T, Y)
where
    T: Block,
    Y: Send + Sync + Debug,
{
    type Memory = T::Memory;

    #[inline(always)]
    fn memory(&self) -> &T::Memory {
        self.0.memory()
    }

    #[inline(always)]
    fn range(&self) -> Range<u64> {
        self.0.range()
    }
}
