use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::Backend;

use MemoryAllocator;
use relevant::Relevant;

/// Trait for types that represent a block (`Range`) of `Memory`.
pub trait Block<B: Backend> {
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
/// A `TaggedBlock` must never be silently dropped, that will result in a panic.
/// The block must be freed by returning it to the same allocator it came from.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `T`: tag type, used by allocators to track allocations
#[derive(Debug)]
pub struct TaggedBlock<B: Backend, T> {
    relevant: Relevant,
    range: Range<u64>,
    memory: *const B::Memory,
    tag: T,
}

unsafe impl<B, T> Send for TaggedBlock<B, T>
where
    B: Backend,
    T: Send,
{
}

unsafe impl<B, T> Sync for TaggedBlock<B, T>
where
    B: Backend,
    T: Sync,
{
}

impl<B> TaggedBlock<B, ()>
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
        TaggedBlock {
            relevant: Relevant,
            tag: (),
            memory,
            range,
        }
    }
}

impl<B, T> Block<B> for TaggedBlock<B, T>
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

impl<B, T> TaggedBlock<B, T>
where
    B: Backend,
{
    /// Free this block by returning it to the origin.
    ///
    /// ### Parameters:
    ///
    /// - `origin`: allocator the block was allocated from
    /// - `device`: device the memory was allocated from
    ///
    /// ### Type parameters:
    ///
    /// - `A`: allocator type
    pub fn free<A>(self, origin: &mut A, device: &B::Device)
    where
        A: MemoryAllocator<B, Block = Self>,
        T: Debug + Copy + Send + Sync,
    {
        origin.free(device, self);
    }

    /// Push an additional tag value to this block.
    ///
    /// Tags form a stack.
    ///
    /// ### Parameters:
    ///
    /// - `value`: new tag to push onto the tag stack
    ///
    /// ### Type parameters:
    ///
    /// - `Y`: type of the new tag
    pub fn push_tag<Y>(self, value: Y) -> TaggedBlock<B, (Y, T)> {
        TaggedBlock {
            relevant: self.relevant,
            memory: self.memory,
            tag: (value, self.tag),
            range: self.range,
        }
    }

    /// Replace tag value using the given function.
    ///
    /// Tags form a stack.
    ///
    /// ### Parameters:
    ///
    /// - `f`: function that will take the current tag and output a new tag for the block
    ///
    /// ### Type parameters:
    ///
    /// - `Y`: type of the new tag
    pub fn convert_tag<F, Y>(self, f: F) -> TaggedBlock<B, Y>
    where
        F: FnOnce(T) -> Y,
    {
        TaggedBlock {
            relevant: self.relevant,
            memory: self.memory,
            tag: f(self.tag),
            range: self.range,
        }
    }

    /// Replace the tag value of this block
    ///
    /// ### Parameters:
    ///
    /// - `value`: new tag for the block
    ///
    /// ### Type parameters:
    ///
    /// - `Y`: type of the new tag
    ///
    /// ### Returns
    ///
    /// The block and the old tag value
    pub fn replace_tag<Y>(self, value: Y) -> (TaggedBlock<B, Y>, T) {
        (
            TaggedBlock {
                relevant: self.relevant,
                memory: self.memory,
                tag: value,
                range: self.range,
            },
            self.tag,
        )
    }

    /// Take the tag value of this block, leaving `()` in its place.
    ///
    /// ### Returns
    ///
    /// The block and the old tag value
    pub fn take_tag(self) -> (TaggedBlock<B, ()>, T) {
        self.replace_tag(())
    }

    /// Set the tag value of this block.
    ///
    /// The old tag will be dropped.
    ///
    /// ### Parameters:
    ///
    /// - `value`: new tag for the block
    ///
    /// ### Type parameters:
    ///
    /// - `Y`: type of the new tag
    pub fn set_tag<Y>(self, value: Y) -> TaggedBlock<B, Y> {
        TaggedBlock {
            relevant: self.relevant,
            memory: self.memory,
            tag: value,
            range: self.range,
        }
    }

    /// Dispose of this block.
    ///
    /// This is unsafe because the caller must ensure that the memory of the block is not used
    /// again. This will typically entail dropping some kind of resource (`Buffer` or `Image` to
    /// give some examples) that occupy this memory.
    ///
    /// ### Returns
    ///
    /// Tag value of the block
    pub unsafe fn dispose(self) -> T {
        self.relevant.dispose();
        self.tag
    }
}

impl<B, T, Y> TaggedBlock<B, (Y, T)>
where
    B: Backend,
{
    /// Pop top tag value from this block
    ///
    /// Tags form a stack.
    ///
    /// ### Returns
    ///
    /// The block and the old top tag value
    pub fn pop_tag(self) -> (TaggedBlock<B, T>, Y) {
        (
            TaggedBlock {
                relevant: self.relevant,
                memory: self.memory,
                tag: self.tag.1,
                range: self.range,
            },
            self.tag.0,
        )
    }
}
