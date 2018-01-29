use std::collections::VecDeque;
use std::mem::replace;

use gfx_hal::{Backend, MemoryTypeId};
use gfx_hal::memory::Requirements;

use block::{Block, TaggedBlock};
use {alignment_shift, MemoryAllocator, MemoryError, MemorySubAllocator};

#[derive(Debug)]
struct ArenaNode<B: Backend, A: MemoryAllocator<B>> {
    used: u64,
    freed: u64,
    block: A::Block,
}

impl<B, A> ArenaNode<B, A>
where
    B: Backend,
    A: MemoryAllocator<B>,
{
    fn new(block: A::Block) -> Self {
        ArenaNode {
            used: 0,
            freed: 0,
            block,
        }
    }

    fn alloc(&mut self, reqs: Requirements) -> Option<TaggedBlock<B, ()>> {
        let offset = self.block.range().start + self.used;
        let total_size = reqs.size + alignment_shift(reqs.alignment, offset);

        if self.block.size() - self.used < total_size {
            None
        } else {
            self.used += total_size;
            Some(TaggedBlock::new(self.block.memory(), offset..total_size + offset))
        }
    }

    fn free(&mut self, block: TaggedBlock<B, ()>) {
        assert!(self.block.contains(&block));
        self.freed += block.size();
        unsafe { block.dispose() }
    }

    fn is_used(&self) -> bool {
        self.freed != self.used
    }

    fn dispose(self, owner: &mut A, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            owner.free(device, self.block);
            Ok(())
        }
    }
}

/// Linear allocator for short-lived objects
#[derive(Debug)]
pub struct ArenaAllocator<B: Backend, A: MemoryAllocator<B>> {
    id: MemoryTypeId,
    arena_size: u64,
    freed: u64,
    hot: Option<ArenaNode<B, A>>,
    nodes: VecDeque<ArenaNode<B, A>>,
}

impl<B, A> ArenaAllocator<B, A>
where
    B: Backend,
    A: MemoryAllocator<B>,
{
    /// Construct allocator.
    pub fn new(arena_size: u64, id: MemoryTypeId) -> Self {
        ArenaAllocator {
            id,
            arena_size,
            freed: 0,
            hot: None,
            nodes: VecDeque::new(),
        }
    }

    /// Get memory type of the allocator
    pub fn memory_type(&self) -> MemoryTypeId {
        self.id
    }

    /// Get size of the arena
    pub fn arena_size(&self) -> u64 {
        self.arena_size
    }

    fn cleanup(&mut self, owner: &mut A, device: &B::Device) {
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

    fn allocate_node(
        &mut self,
        owner: &mut A,
        device: &B::Device,
        request: A::Request,
        reqs: Requirements,
    ) -> Result<ArenaNode<B, A>, MemoryError> {
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

impl<B, A> MemorySubAllocator<B> for ArenaAllocator<B, A>
where
    B: Backend,
    A: MemoryAllocator<B>,
{
    type Owner = A;
    type Request = A::Request;
    type Block = TaggedBlock<B, Tag>;

    fn alloc(
        &mut self,
        owner: &mut A,
        device: &B::Device,
        request: A::Request,
        reqs: Requirements,
    ) -> Result<TaggedBlock<B, Tag>, MemoryError> {
        if (1 << self.id.0) & reqs.type_mask == 0 {
            return Err(MemoryError::NoCompatibleMemoryType);
        }
        let count = self.freed + self.nodes.len() as u64;
        if let Some(ref mut hot) = self.hot.as_mut() {
            match hot.alloc(reqs) {
                Some(block) => return Ok(block.set_tag(Tag(count))),
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
        let count = self.freed + self.nodes.len() as u64;
        Ok(block.set_tag(Tag(count)))
    }

    fn free(&mut self, owner: &mut A, device: &B::Device, block: TaggedBlock<B, Tag>) {
        let (block, Tag(tag)) = block.replace_tag(());
        let index = (tag - self.freed) as usize;

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

    fn is_used(&self) -> bool {
        self.nodes.is_empty()
            && self.hot
                .as_ref()
                .map(|node| node.is_used())
                .unwrap_or(false)
    }

    fn dispose(mut self, owner: &mut A, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            if let Some(hot) = self.hot.take() {
                hot.dispose(owner, device).unwrap();
            }
            Ok(())
        }
    }
}

/// Opaque type for `TaggedBlock` tag.
/// `ArenaAllocator` places this tag and than uses it in `MemorySubAllocator::free` method.
#[derive(Debug, Clone, Copy)]
pub struct Tag(u64);
