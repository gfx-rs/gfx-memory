use std::any::Any;
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::{Backend, MemoryTypeId};
use gfx_hal::memory::Requirements;

use {alignment_shift, MemoryAllocator, MemoryError, MemorySubAllocator};
use block::{Block, RawBlock};

/// Chunks are super-allocator blocks,
/// which are then divided into smaller 'blocks'
#[derive(Debug)]
struct FreeBlock {
    /// Index of chunk (big block from super-allocator)
    chunk_index: usize,
    /// Block index inside the chunk
    block_index: u64,
}

#[derive(Debug)]
struct ChunkedNode<T> {
    id: MemoryTypeId,
    /// Size of chunks - big blocks this allocator takes from super-allocator.
    chunk_size: u64,
    /// Size of small blocks
    block_size: u64,
    /// List of free blocks
    free: VecDeque<FreeBlock>,
    /// List of allocated chunks
    chunks: Vec<T>,
}

impl<T> ChunkedNode<T> {
    fn new(chunk_size: u64, block_size: u64, id: MemoryTypeId) -> Self {
        ChunkedNode {
            id,
            chunk_size,
            block_size,
            free: VecDeque::new(),
            chunks: Vec::new(),
        }
    }

    fn is_used(&self) -> bool {
        // All blocks are free
        self.count() != self.free.len()
    }

    fn count(&self) -> usize {
        // Blocks count is chunk count multiplied by blocks per chunk
        self.chunks.len() * self.blocks_per_chunk()
    }

    fn blocks_per_chunk(&self) -> usize {
        // How many blocks there are in a chunk
        (self.chunk_size / self.block_size) as usize
    }

    fn grow<B, A>(
        &mut self,
        owner: &mut A,
        device: &B::Device,
        request: A::Request,
    ) -> Result<(), MemoryError>
    where
        B: Backend,
        T: Block<Memory = B::Memory>,
        A: MemoryAllocator<B, Block = T>,
    {
        let reqs = Requirements {
            type_mask: 1 << self.id.0,
            size: self.chunk_size,
            alignment: self.block_size,
        };
        // Get a new chunk
        let chunk = owner.alloc(device, request, reqs)?;
        assert_eq!(0, alignment_shift(reqs.alignment, chunk.range().start));
        assert!(chunk.size() >= self.chunk_size);

        let blocks_per_chunk = self.blocks_per_chunk();

        // `len()` will return the next index to use
        let chunk_index = self.chunks.len();

        // Fill the free list with new blocks
        self.free.extend((0..blocks_per_chunk).map(|i| FreeBlock {
            chunk_index,
            block_index: i as u64,
        }));

        // Place the new chunk in the list
        self.chunks.push(chunk);

        Ok(())
    }

    fn alloc_no_grow<M>(&mut self) -> Option<ChunkedBlock<M>>
    where
        M: Debug + Any,
        T: Block<Memory = M>,
    {
        // Find a free block
        self.free.pop_front().map(|free_block| {
            // Memory offset is block index times block size
            // plus chunk memory offset
            let offset = free_block.block_index * self.block_size
                + self.chunks[free_block.chunk_index].range().start;
            let block = RawBlock::new(
                self.chunks[free_block.chunk_index].memory(),
                offset..self.block_size + offset,
            );
            // Remember what chunk the block came from
            ChunkedBlock(block, free_block.chunk_index)
        })
    }
}

impl<B, O, T> MemorySubAllocator<B, O> for ChunkedNode<T>
where
    B: Backend,
    T: Block<Memory = B::Memory>,
    O: MemoryAllocator<B, Block = T>,
{
    type Request = O::Request;
    type Block = ChunkedBlock<B::Memory>;

    fn alloc(
        &mut self,
        owner: &mut O,
        device: &B::Device,
        request: O::Request,
        reqs: Requirements,
    ) -> Result<ChunkedBlock<B::Memory>, MemoryError> {
        // Check memory type
        if (1 << self.id.0) & reqs.type_mask == 0 {
            return Err(MemoryError::NoCompatibleMemoryType);
        }

        // Try to allocate a block
        let block = match self.alloc_no_grow() {
            Some(block) => block,
            None => {
                // Grow from super-allocator
                self.grow(owner, device, request)?;
                self.alloc_no_grow().expect("Just growed")
            }
        };

        // Check that block meets the requirements.
        assert!(block.size() >= reqs.size);
        assert_eq!(block.range().start & (reqs.alignment - 1), 0);
        Ok(block)
    }

    fn free(&mut self, _owner: &mut O, _device: &B::Device, block: ChunkedBlock<B::Memory>) {
        assert_eq!(block.range().start % self.block_size, 0);
        assert_eq!(block.size(), self.block_size);
        let offset = block.range().start;
        let block_memory: *const B::Memory = block.memory();

        // Dispose block retreiving chunk index
        let chunk_index = unsafe {
            block.0.dispose();
            block.1
        };

        // Confirm the chunk index
        assert!(::std::ptr::eq(
            self.chunks[chunk_index].memory(),
            block_memory
        ));

        // Calculate the block index inside the chunk
        let block_index = (offset - self.chunks[chunk_index].range().start) / self.block_size;

        // Push the block back into the 'free blocks' list
        self.free.push_front(FreeBlock {
            block_index,
            chunk_index,
        });
    }

    fn dispose(mut self, owner: &mut O, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            for chunk in self.chunks.drain(..) {
                owner.free(device, chunk);
            }
            Ok(())
        }
    }
}

/// Sub-allocator that can be used for long-lived objects.
///
/// This allocator allocates memory in chunks containing `blocks_per_chunk` equally sized blocks
/// from the underlying allocator, up to a maximum chunk size of `max_chunk_size` bytes. It rounds
/// up the requested allocation size to the closest power of two and returns a single block from a
/// chunk.
///
/// This allocator can only allocate memory `max_chunk_size` bytes in size or less.
///
/// ### Type parameters:
///
/// - `T`: type of bigger blocks this allocator sub-allocates from.
#[derive(Debug)]
pub struct ChunkedAllocator<T> {
    id: MemoryTypeId,
    blocks_per_chunk: usize,
    min_block_size: u64,
    max_chunk_size: u64,
    nodes: Vec<ChunkedNode<T>>,
}

impl<T> ChunkedAllocator<T> {
    /// Create a new chunked allocator.
    ///
    /// ### Parameters:
    ///
    /// - `blocks_per_chunk`: The number of blocks in each chunk allocated from the underlying
    ///                       allocator.
    /// - `min_block_size`: The minimum block size used by this allocator in bytes. Allocations
    ///                     significantly than this may incur much larger overhead.
    /// - `max_chunk_size`: The maximum size of chunks allocated from the underlying allocator
    ///                     in bytes. Blocks larger than this cannot be allocated.
    /// - `id`: ID of the memory type this allocator allocates from.
    ///
    /// ### Panics
    ///
    /// Panics if `min_block_size` or `max_chunk_size` are not a power of two.
    pub fn new(
        blocks_per_chunk: usize,
        min_block_size: u64,
        max_chunk_size: u64,
        id: MemoryTypeId,
    ) -> Self {
        assert!(min_block_size.is_power_of_two());
        assert!(max_chunk_size.is_power_of_two());
        ChunkedAllocator {
            id,
            blocks_per_chunk,
            min_block_size,
            max_chunk_size,
            nodes: Vec::new(),
        }
    }

    /// Check if any of the blocks allocated by this allocator are still in use.
    /// If this function returns `false`, the allocator can be `dispose`d.
    pub fn is_used(&self) -> bool {
        self.nodes.iter().any(ChunkedNode::is_used)
    }

    /// Get memory type of the allocator
    pub fn memory_type(&self) -> MemoryTypeId {
        self.id
    }

    /// Get minimum block size
    pub fn min_block_size(&self) -> u64 {
        self.min_block_size
    }

    /// Get maximum chunk size
    pub fn max_chunk_size(&self) -> u64 {
        self.max_chunk_size
    }

    /// Get the number of chunks per block
    pub fn blocks_per_chunk(&self) -> usize {
        self.blocks_per_chunk
    }

    /// Retrieves the block backing an allocation.
    pub fn underlying_block<M: Debug + Any>(&self, block: &ChunkedBlock<M>) -> &T {
        let index = self.pick_node(block.size());
        &self.nodes[index as usize].chunks[block.1]
    }

    fn block_size(&self, index: u8) -> u64 {
        self.min_block_size * (1u64 << (index as u8))
    }

    fn chunk_size(&self, index: u8) -> u64 {
        min(
            self.block_size(index) * self.blocks_per_chunk as u64,
            self.max_chunk_size,
        )
    }

    fn pick_node(&self, size: u64) -> u8 {
        // blocks can't be larger than max_chunk_size
        debug_assert!(size <= self.max_chunk_size);
        let bits = ::std::mem::size_of::<usize>() * 8;
        assert_ne!(size, 0);
        let node = (bits - ((size - 1) / self.min_block_size).leading_zeros() as usize) as u8;
        debug_assert!(size <= self.block_size(node));
        debug_assert!(node == 0 || size > self.block_size(node - 1));
        node
    }

    fn grow(&mut self, index: u8) {
        assert!(self.chunk_size(index) <= self.max_chunk_size);
        let len = self.nodes.len() as u8;
        let id = self.id;

        let range = len..index + 1;
        self.nodes.reserve(range.len());
        for index in range {
            let node = ChunkedNode::new(self.chunk_size(index), self.block_size(index), id);
            self.nodes.push(node);
        }
    }
}

impl<B, O, T> MemorySubAllocator<B, O> for ChunkedAllocator<T>
where
    B: Backend,
    T: Block<Memory = B::Memory>,
    O: MemoryAllocator<B, Block = T>,
{
    type Request = O::Request;
    type Block = ChunkedBlock<B::Memory>;

    fn alloc(
        &mut self,
        owner: &mut O,
        device: &B::Device,
        request: O::Request,
        reqs: Requirements,
    ) -> Result<ChunkedBlock<B::Memory>, MemoryError> {
        if max(reqs.size, reqs.alignment) > self.max_chunk_size {
            return Err(MemoryError::OutOfMemory);
        }
        let index = self.pick_node(max(reqs.size, reqs.alignment));
        self.grow(index);
        self.nodes[index as usize].alloc(owner, device, request, reqs)
    }

    fn free(&mut self, owner: &mut O, device: &B::Device, block: ChunkedBlock<B::Memory>) {
        let index = self.pick_node(block.size());
        self.nodes[index as usize].free(owner, device, block);
    }

    fn dispose(mut self, owner: &mut O, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            for node in self.nodes.drain(..) {
                node.dispose(owner, device).unwrap();
            }
            Ok(())
        }
    }
}

/// `Block` type returned by `ChunkedAllocator`.
#[derive(Debug)]
pub struct ChunkedBlock<M>(pub(crate) RawBlock<M>, pub(crate) usize);

impl<M> Block for ChunkedBlock<M>
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
    fn bar<M: Send + Sync>() {
        foo::<ChunkedAllocator<M>>()
    }
}
