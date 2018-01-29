use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::Backend;

use MemoryAllocator;
use relevant::Relevant;

/// Trait for types that represents or bound to range of `Memory`.
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

    /// Helper method to check if `other` block is sub-block of `self`
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
/// It is relevant type and can't be silently dropped.
/// User must return this block to the same allocator it came from.
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
    /// Construct untagged block from `Memory` and `Range`.
    /// Pointed `Memory` shouldn't be freed until at least one `TaggedBlock`s of it
    /// still exists.
    pub fn new(memory: *const B::Memory, range: Range<u64>) -> Self {
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
    /// Free this block returning it to the origin.
    /// It must be the allocator this block was allocated from.
    pub fn free<A>(self, origin: &mut A, device: &B::Device)
    where
        A: MemoryAllocator<B, Tag = T>,
        T: Debug + Copy + Send + Sync,
    {
        origin.free(device, self);
    }

    /// Push additional tag value to this block.
    /// Tags form a stack - e.g. LIFO
    pub fn push_tag<Y>(self, value: Y) -> TaggedBlock<B, (Y, T)> {
        TaggedBlock {
            relevant: self.relevant,
            memory: self.memory,
            tag: (value, self.tag),
            range: self.range,
        }
    }

    /// Convert tag value using specified function.
    /// Tags form a stack - e.g. LIFO
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

    /// Replace tag attached to this block
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

    /// Take tag attached to this block leaving `()` in its place.
    pub fn take_tag(self) -> (TaggedBlock<B, ()>, T) {
        self.replace_tag(())
    }

    /// Set tag to this block.
    /// Drops old tag.
    pub fn set_tag<Y>(self, value: Y) -> TaggedBlock<B, Y> {
        TaggedBlock {
            relevant: self.relevant,
            memory: self.memory,
            tag: value,
            range: self.range,
        }
    }

    /// Dispose of this block.
    /// Returns tag value.
    /// This is unsafe as the caller must ensure that
    /// the memory of the block won't be used.
    /// Typically by dropping resource (`Buffer` or `Image`) that occupy this memory.
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
    /// Tags form a stack - e.g. LIFO
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
