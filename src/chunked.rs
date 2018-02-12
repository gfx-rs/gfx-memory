use std::any::Any;
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::fmt::Debug;
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
        T: Block<Memory = B::Memory>,
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

    fn alloc_no_grow<M>(&mut self) -> Option<ChunkedBlock<M>>
    where
        M: Debug + Any,
        T: Block<Memory = M>,
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

    fn free(&mut self, _owner: &mut O, _device: &B::Device, block: ChunkedBlock<B::Memory>) {
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
/// - `T`: type of bigger blocks this allcator sub-allocates from.
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
        if reqs.size > self.max_chunk_size {
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
