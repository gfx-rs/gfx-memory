use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::{Backend, MemoryTypeId};
use gfx_hal::memory::Requirements;

use {MemoryAllocator, MemoryError, MemorySubAllocator};
use arena::{ArenaAllocator, ArenaBlock};
use block::{Block, RawBlock};
use chunked::{ChunkedAllocator, ChunkedBlock};
use root::RootAllocator;

/// Controls what sub allocator is used for an allocation by `CombinedAllocator`
#[derive(Clone, Copy, Debug)]
pub enum Type {
    /// For short-lived objects, such as staging buffers.
    ShortLived,

    /// General purpose.
    General,
}

/// Allocator with support for both short-lived and long-lived allocations.
///
/// This allocator allocates blocks using either an `ArenaAllocator` or a `ChunkedAllocator`
/// depending on which kind of allocation is requested.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
#[derive(Debug)]
pub struct CombinedAllocator<B>
where
    B: Backend,
{
    root: RootAllocator<B>,
    root_used: u64,
    arenas: ArenaAllocator<RawBlock<B::Memory>>,
    chunks: ChunkedAllocator<RawBlock<B::Memory>>,
    allocations: usize,
}

impl<B> CombinedAllocator<B>
where
    B: Backend,
{
    /// Create a combined allocator.
    ///
    /// ### Parameters:
    ///
    /// - `memory_type_id`: ID of the memory type this allocator allocates from.
    /// - `arena_chunk_size`: see `ArenaAllocator`
    /// - `blocks_per_chunk`: see `ChunkedAllocator`
    /// - `min_block_size`: see `ChunkedAllocator`
    /// - `max_chunk_size`: see `ChunkedAllocator`
    pub fn new(
        memory_type_id: MemoryTypeId,
        arena_chunk_size: u64,
        blocks_per_chunk: usize,
        min_block_size: u64,
        max_chunk_size: u64,
    ) -> Self {
        CombinedAllocator {
            root: RootAllocator::new(memory_type_id),
            root_used: 0,
            arenas: ArenaAllocator::new(memory_type_id, arena_chunk_size),
            chunks: ChunkedAllocator::new(
                memory_type_id,
                blocks_per_chunk,
                min_block_size,
                max_chunk_size,
            ),
            allocations: 0,
        }
    }

    /// Get memory type id
    pub fn memory_type(&self) -> MemoryTypeId {
        self.root.memory_type()
    }

    /// Get the total size of all blocks allocated by this allocator.
    pub fn used(&self) -> u64 {
        self.root_used + self.arenas.used() + self.chunks.used()
    }

    /// Get the total size of all chunks allocated by this allocator.
    pub fn allocated(&self) -> u64 {
        self.root_used + self.arenas.allocated() + self.chunks.allocated()
    }
}

impl<B> MemoryAllocator<B> for CombinedAllocator<B>
where
    B: Backend,
{
    type Request = Type;
    type Block = CombinedBlock<B::Memory>;

    fn alloc(
        &mut self,
        device: &B::Device,
        request: Type,
        reqs: Requirements,
    ) -> Result<CombinedBlock<B::Memory>, MemoryError> {
        let block = match request {
            Type::ShortLived => self.arenas
                .alloc(&mut self.root, device, (), reqs)
                .map(|ArenaBlock(block, tag)| CombinedBlock(block, CombinedTag::Arena(tag)))?,
            Type::General => {
                if reqs.size > self.chunks.max_chunk_size() / 2 {
                    let block = self.root
                        .alloc(device, (), reqs)
                        .map(|block| CombinedBlock(block, CombinedTag::Root))?;
                    self.root_used += block.size();
                    block
                } else {
                    self.chunks.alloc(&mut self.root, device, (), reqs).map(
                        |ChunkedBlock(block, tag)| CombinedBlock(block, CombinedTag::Chunked(tag)),
                    )?
                }
            }
        };
        self.allocations += 1;
        Ok(block)
    }

    fn free(&mut self, device: &B::Device, block: CombinedBlock<B::Memory>) {
        match block.1 {
            CombinedTag::Arena(tag) => {
                self.arenas.free(&mut self.root, device, ArenaBlock(block.0, tag))
            }
            CombinedTag::Chunked(tag) => {
                self.chunks.free(&mut self.root, device, ChunkedBlock(block.0, tag))
            }
            CombinedTag::Root => {
                self.root_used -= block.size();
                self.root.free(device, block.0)
            },
        }
        self.allocations -= 1;
    }

    fn is_used(&self) -> bool {
        if self.allocations == 0 {
            debug_assert!(!self.arenas.is_used() && !self.chunks.is_used());
            false
        } else {
            true
        }
    }

    fn dispose(mut self, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            return Err(self);
        }
        self.arenas.dispose(&mut self.root, device).unwrap();
        self.chunks.dispose(&mut self.root, device).unwrap();
        self.root.dispose(device).unwrap();
        Ok(())
    }
}

/// `Block` type returned by `CombinedAllocator`.
#[derive(Debug)]
pub struct CombinedBlock<M>(pub(crate) RawBlock<M>, pub(crate) CombinedTag);

#[derive(Debug)]
pub(crate) enum CombinedTag {
    Arena(u64),
    Chunked(usize),
    Root,
}

impl<M> Block for CombinedBlock<M>
where
    M: Debug + Any,
{
    type Memory = M;

    #[inline(always)]
    fn memory(&self) -> &M {
        self.0.memory()
    }

    #[inline(always)]
    fn range(&self) -> Range<u64> {
        self.0.range()
    }
}

#[test]
#[allow(dead_code)]
fn test_send_sync() {
    fn foo<T: Send + Sync>() {}
    fn bar<B: Backend>() {
        foo::<CombinedAllocator<B>>()
    }
}
