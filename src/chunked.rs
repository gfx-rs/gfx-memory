use std::cmp::{max, min};
use std::collections::VecDeque;
use std::ops::Range;

use gfx_hal::{Backend, MemoryTypeId};
use gfx_hal::memory::Requirements;

use {alignment_shift, MemoryAllocator, MemoryError, MemorySubAllocator};
use block::{Block, RawBlock};

#[derive(Debug)]
struct FreeBlock {
    block_index: usize,
    chunk_index: u64,
}

#[derive(Debug)]
struct ChunkedNode<T> {
    id: MemoryTypeId,
    block_size: u64,
    chunk_size: u64,
    free: VecDeque<FreeBlock>,
    blocks: Vec<T>,
}

impl<T> ChunkedNode<T> {
    fn new(chunk_size: u64, block_size: u64, id: MemoryTypeId) -> Self {
        ChunkedNode {
            id,
            chunk_size,
            block_size,
            free: VecDeque::new(),
            blocks: Vec::new(),
        }
    }

    fn is_used(&self) -> bool {
        self.count() != self.free.len()
    }

    fn count(&self) -> usize {
        self.blocks.len() * self.blocks_per_chunk()
    }

    fn blocks_per_chunk(&self) -> usize {
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
        T: Block<B>,
        A: MemoryAllocator<B, Block = T>,
    {
        let reqs = Requirements {
            type_mask: 1 << self.id.0,
            size: self.chunk_size,
            alignment: self.chunk_size,
        };
        let block = owner.alloc(device, request, reqs)?;
        assert_eq!(0, alignment_shift(reqs.alignment, block.range().start));
        assert!(block.size() >= self.chunk_size);

        let blocks_per_chunk = self.blocks_per_chunk();
        let block_index = self.blocks.len();
        self.free.extend((0..blocks_per_chunk).map(|i| FreeBlock {
            block_index,
            chunk_index: i as u64,
        }));
        self.blocks.push(block);

        Ok(())
    }

    fn alloc_no_grow<B>(&mut self) -> Option<ChunkedBlock<B>>
    where
        B: Backend,
        T: Block<B>,
    {
        self.free.pop_front().map(|free_block| {
            let offset = free_block.chunk_index * self.chunk_size;
            let block = RawBlock::new(
                self.blocks[free_block.block_index].memory(),
                offset..self.chunk_size + offset,
            );
            ChunkedBlock(block, free_block.block_index)
        })
    }
}

impl<B, O, T> MemorySubAllocator<B, O> for ChunkedNode<T>
where
    B: Backend,
    T: Block<B>,
    O: MemoryAllocator<B, Block = T>,
{
    type Request = O::Request;
    type Block = ChunkedBlock<B>;

    fn alloc(
        &mut self,
        owner: &mut O,
        device: &B::Device,
        request: O::Request,
        reqs: Requirements,
    ) -> Result<ChunkedBlock<B>, MemoryError> {
        if (1 << self.id.0) & reqs.type_mask == 0 {
            return Err(MemoryError::NoCompatibleMemoryType);
        }
        Ok(match self.alloc_no_grow() {
            Some(block) => {
                assert!(block.size() >= reqs.size);
                assert_eq!(block.range().start & (reqs.alignment - 1), 0);
                block
            }
            None => {
                self.grow(owner, device, request)?;
                self.alloc_no_grow().expect("Just growed")
            }
        })
    }

    fn free(&mut self, _owner: &mut O, _device: &B::Device, block: ChunkedBlock<B>) {
        assert_eq!(block.range().start % self.chunk_size, 0);
        assert_eq!(block.size(), self.chunk_size);
        let offset = block.range().start;
        let block_memory: *const B::Memory = block.memory();
        let block_index = unsafe {
            block.0.dispose();
            block.1
        };
        assert!(::std::ptr::eq(
            self.blocks[block_index].memory(),
            block_memory
        ));
        let chunk_index = offset / self.chunk_size;
        self.free.push_front(FreeBlock {
            block_index,
            chunk_index,
        });
    }

    fn dispose(mut self, owner: &mut O, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            for block in self.blocks.drain(..) {
                owner.free(device, block);
            }
            Ok(())
        }
    }
}

/// Allocator that rounds up the requested size to the closest power of two and returns a block
/// from a list of equal sized chunks.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `A`: allocator used to allocate bigger blocks of memory
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
    /// - `blocks_per_chunk`: used for calculating size of memory blocks to request from the
    ///                       underlying allocator
    /// - `min_block_size`: ?
    /// - `max_chunk_size`: ?
    /// - `id`: hal memory type
    ///
    /// ### Panics
    ///
    /// Panics if `chunk_size` or `min_block_size` are not a power of two.
    pub fn new(
        blocks_per_chunk: usize,
        min_block_size: u64,
        max_chunk_size: u64,
        id: MemoryTypeId,
    ) -> Self {
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

    /// Get minimal block size
    pub fn min_block_size(&self) -> u64 {
        self.min_block_size
    }

    /// Get maximal chunk size
    pub fn max_chunk_size(&self) -> u64 {
        self.max_chunk_size
    }

    /// Get chunks per block count
    pub fn blocks_per_chunk(&self) -> usize {
        self.blocks_per_chunk
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
    T: Block<B>,
    O: MemoryAllocator<B, Block = T>,
{
    type Request = O::Request;
    type Block = ChunkedBlock<B>;

    fn alloc(
        &mut self,
        owner: &mut O,
        device: &B::Device,
        request: O::Request,
        reqs: Requirements,
    ) -> Result<ChunkedBlock<B>, MemoryError> {
        if reqs.size > self.max_chunk_size {
            return Err(MemoryError::OutOfMemory);
        }
        let index = self.pick_node(max(reqs.size, reqs.alignment));
        self.grow(index);
        self.nodes[index as usize].alloc(owner, device, request, reqs)
    }

    fn free(&mut self, owner: &mut O, device: &B::Device, block: ChunkedBlock<B>) {
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

/// Opaque type for `Block` tag used by the `ChunkedAllocator`.
///
/// `ChunkedAllocator` places this tag on the memory blocks, and then use it in
/// `free` to find the memory node the block was allocated from.
#[derive(Debug)]
pub struct ChunkedBlock<B: Backend>(pub(crate) RawBlock<B>, pub(crate) usize);

impl<B> Block<B> for ChunkedBlock<B>
where
    B: Backend,
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
