use std::any::Any;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::mem::replace;
use std::ops::Range;

use gfx_hal::{Backend, MemoryTypeId};
use gfx_hal::memory::Requirements;

use {alignment_shift, MemoryAllocator, MemoryError, MemorySubAllocator};
use block::{Block, RawBlock};

/// Linear allocator that can be used for short-lived objects.
///
/// ### Type parameters:
///
/// - `T`: type of bigger blocks this allcator sub-allocates from.
#[derive(Debug)]
pub struct ArenaAllocator<T> {
    id: MemoryTypeId,
    arena_size: u64,
    freed: u64,
    hot: Option<ArenaNode<T>>,
    nodes: VecDeque<ArenaNode<T>>,
}

impl<T> ArenaAllocator<T> {
    /// Create a new arena allocator
    ///
    /// ### Parameters:
    ///
    /// - `arena_size`: size in bytes of the arena
    /// - `id`: hal memory type
    pub fn new(arena_size: u64, id: MemoryTypeId) -> Self {
        ArenaAllocator {
            id,
            arena_size,
            freed: 0,
            hot: None,
            nodes: VecDeque::new(),
        }
    }

    /// Check if any of the blocks allocated by this allocator are still in use.
    /// If this function returns `false`, the allocator can be `dispose`d.
    pub fn is_used(&self) -> bool {
        !self.nodes.is_empty()
            || self.hot
                .as_ref()
                .map(|node| node.is_used())
                .unwrap_or(false)
    }

    /// Get memory type of the allocator
    pub fn memory_type(&self) -> MemoryTypeId {
        self.id
    }

    /// Get size of the arena
    pub fn arena_size(&self) -> u64 {
        self.arena_size
    }

    fn cleanup<B, A>(&mut self, owner: &mut A, device: &B::Device)
    where
        B: Backend,
        T: Block<Memory = B::Memory>,
        A: MemoryAllocator<B, Block = T>,
    {
        while self.nodes
            .front()
            .map(|node| !node.is_used())
            .unwrap_or(false)
        {
            if let Some(node) = self.nodes.pop_front() {
                if let Some(ref mut hot) = self.hot {
                    if hot.is_used() {
                        self.nodes.push_back(replace(hot, node));
                    } else {
                        // No need to replace.
                        node.dispose(owner, device).unwrap();
                    }
                }
            }
            self.freed += 1;
        }
    }

    fn allocate_node<B, A>(
        &mut self,
        owner: &mut A,
        device: &B::Device,
        request: A::Request,
        reqs: Requirements,
    ) -> Result<ArenaNode<T>, MemoryError>
    where
        B: Backend,
        T: Block<Memory = B::Memory>,
        A: MemoryAllocator<B, Block = T>,
    {
        let arena_size = ((reqs.size - 1) / self.arena_size + 1) * self.arena_size;
        let arena_requirements = Requirements {
            type_mask: 1 << self.id.0,
            size: arena_size,
            alignment: reqs.alignment,
        };
        let arena_block = owner.alloc(device, request, arena_requirements)?;
        Ok(ArenaNode::new(arena_block))
    }
}

impl<B, O, T> MemorySubAllocator<B, O> for ArenaAllocator<T>
where
    B: Backend,
    T: Block<Memory = B::Memory>,
    O: MemoryAllocator<B, Block = T>,
{
    type Request = O::Request;
    type Block = ArenaBlock<B::Memory>;

    fn alloc(
        &mut self,
        owner: &mut O,
        device: &B::Device,
        request: O::Request,
        reqs: Requirements,
    ) -> Result<ArenaBlock<B::Memory>, MemoryError> {
        if (1 << self.id.0) & reqs.type_mask == 0 {
            return Err(MemoryError::NoCompatibleMemoryType);
        }
        let index = self.freed + self.nodes.len() as u64;
        if let Some(ref mut hot) = self.hot.as_mut() {
            match hot.alloc(reqs) {
                Some(block) => return Ok(ArenaBlock(block, index)),
                None => {}
            }
        };

        let mut node = self.allocate_node(owner, device, request, reqs)?;
        let block = node.alloc(reqs).unwrap();
        if let Some(hot) = replace(&mut self.hot, Some(node)) {
            match hot.dispose(owner, device) {
                Ok(()) => {}
                Err(hot) => self.nodes.push_back(hot),
            }
        };
        let index = self.freed + self.nodes.len() as u64;
        Ok(ArenaBlock(block, index))
    }

    fn free(&mut self, owner: &mut O, device: &B::Device, block: ArenaBlock<B::Memory>) {
        let ArenaBlock(block, index) = block;
        let index = (index - self.freed) as usize;

        match self.nodes.len() {
            len if len == index => {
                self.hot.as_mut().unwrap().free(block);
            }
            len if len > index => {
                self.nodes[index].free(block);
                self.cleanup(owner, device);
            }
            _ => unreachable!(),
        }
    }

    fn dispose(mut self, owner: &mut O, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            if let Some(hot) = self.hot.take() {
                hot.dispose(owner, device).expect("Already checked");
            }
            Ok(())
        }
    }
}

#[derive(Debug)]
struct ArenaNode<T> {
    used: u64,
    freed: u64,
    block: T,
}

impl<T> ArenaNode<T> {
    fn new(block: T) -> Self {
        ArenaNode {
            used: 0,
            freed: 0,
            block,
        }
    }

    fn alloc<M>(&mut self, reqs: Requirements) -> Option<RawBlock<M>>
    where
        M: Debug + Any,
        T: Block<Memory = M>,
    {
        let offset = self.block.range().start + self.used;
        let total_size = reqs.size + alignment_shift(reqs.alignment, offset);

        if self.block.size() - self.used < total_size {
            None
        } else {
            self.used += total_size;
            Some(RawBlock::new(
                self.block.memory(),
                offset..total_size + offset,
            ))
        }
    }

    fn free<M>(&mut self, block: RawBlock<M>)
    where
        M: Debug + Any,
        T: Block<Memory = M>,
    {
        assert!(self.block.contains(&block));
        self.freed += block.size();
        unsafe { block.dispose() }
    }

    fn is_used(&self) -> bool {
        self.freed != self.used
    }

    fn dispose<B, A>(self, owner: &mut A, device: &B::Device) -> Result<(), Self>
    where
        B: Backend,
        T: Block<Memory = B::Memory>,
        A: MemoryAllocator<B, Block = T>,
    {
        if self.is_used() {
            Err(self)
        } else {
            owner.free(device, self.block);
            Ok(())
        }
    }
}

/// `Block` type returned by `ArenaAllocator`.
#[derive(Debug)]
pub struct ArenaBlock<M>(pub(crate) RawBlock<M>, pub(crate) u64);

impl<M> Block for ArenaBlock<M>
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
        foo::<ArenaAllocator<M>>()
    }
}
