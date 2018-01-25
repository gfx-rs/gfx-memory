use gfx_hal::{Backend, MemoryTypeId};
use gfx_hal::memory::Requirements;

use {Block, MemoryError, MemoryAllocator, MemorySubAllocator};
use arena::ArenaAllocator;
use chunked::ChunkedAllocator;
use root::RootAllocator;

/// Type of sub-allocator used.
#[derive(Clone, Copy, Debug)]
pub enum Type {
    /// For short-living objects.
    /// Such as staging buffers.
    ShortLive,

    /// General purpose.
    General,
}

/// Combines arena and chunked sub-allocators.
/// Uses root memory allocator as their super-allocator.
/// Allows to choose which allocator to use.
#[derive(Debug)]
pub struct CombinedAllocator<B>
where
    B: Backend,
{
    root: RootAllocator<B>,
    arenas: ArenaAllocator<B, RootAllocator<B>>,
    chunks: ChunkedAllocator<B, RootAllocator<B>>,
}

impl<B> CombinedAllocator<B>
where
    B: Backend,
{
    /// Create combined allocator with paramaters for sub-allocators.
    pub fn new(
        memory_type_id: MemoryTypeId,
        arena_size: u64,
        chunks_per_block: usize,
        min_chunk_size: u64,
        max_chunk_size: u64,
    ) -> Self {
        CombinedAllocator {
            root: RootAllocator::new(memory_type_id),
            arenas: ArenaAllocator::new(arena_size, memory_type_id),
            chunks: ChunkedAllocator::new(chunks_per_block, min_chunk_size, max_chunk_size, memory_type_id),
        }
    }

    /// Get memory type id
    pub fn memory_type(&self) -> MemoryTypeId {
        self.root.memory_type()
    }
}

impl<B> MemoryAllocator<B> for CombinedAllocator<B>
where
    B: Backend,
{
    type Info = Type;
    type Tag = (Type, usize);

    fn alloc(
        &mut self,
        device: &B::Device,
        info: Type,
        reqs: Requirements,
    ) -> Result<Block<B, (Type, usize)>, MemoryError> {
        match info {
            Type::ShortLive => self.arenas
                .alloc(&mut self.root, device, (), reqs),
            Type::General => {
                if reqs.size > self.chunks.max_chunk_size() {
                    self.root
                        .alloc(device, (), reqs)
                        .map(|block| block.set_tag(0))
                } else {
                    self.chunks
                        .alloc(&mut self.root, device, (), reqs)
                }
            }
        }
        .map(|block| block.push_tag(info))
    }

    fn free(&mut self, device: &B::Device, block: Block<B, (Type, usize)>) {
        let (block, ty) = block.pop_tag();
        match ty {
            Type::ShortLive => self.arenas.free(&mut self.root, device, block),
            Type::General => {
                if block.size() > self.chunks.max_chunk_size() {
                    self.root.free(device, block.set_tag(()));
                } else {
                    self.chunks.free(&mut self.root, device, block);
                }
            }
        }
    }

    fn is_unused(&self) -> bool {
        let unused = self.arenas.is_unused() && self.chunks.is_unused();
        assert_eq!(unused, self.root.is_unused());
        unused
    }

    fn dispose(mut self, device: &B::Device) -> Result<(), Self> {
        let memory_type_id = self.root.memory_type();
        let arena_size = self.arenas.arena_size();
        let chunks_per_block = self.chunks.chunks_per_block();
        let min_chunk_size = self.chunks.min_chunk_size();
        let max_chunk_size = self.chunks.max_chunk_size();

        let arenas = self.arenas.dispose(&mut self.root, device);
        let chunks = self.chunks.dispose(&mut self.root, device);

        if arenas.is_err() || chunks.is_err() {
            let arenas = arenas.err().unwrap_or_else(|| ArenaAllocator::new(arena_size, memory_type_id));
            let chunks = chunks.err().unwrap_or_else(|| ChunkedAllocator::new(chunks_per_block, min_chunk_size, max_chunk_size, memory_type_id));

            Err(CombinedAllocator {
                root: self.root,
                arenas,
                chunks,
            })
        } else {
            self.root.dispose(device).unwrap();
            Ok(())
        }
    }
}
