use std::marker::PhantomData;

use gfx_hal::{Backend, Device, MemoryTypeId};
use gfx_hal::memory::Requirements;
use {Block, MemoryError, MemoryAllocator};
use relevant::Relevant;

/// Allocator that allocates memory directly from device.
#[derive(Debug)]
pub struct RootAllocator<B> {
    relevant: Relevant,
    id: MemoryTypeId,
    allocations: usize,
    pd: PhantomData<fn() -> B>,
}

impl<B> RootAllocator<B> {
    /// Create new allocator that will allocate memory of specified type.
    pub fn new(id: MemoryTypeId) -> Self {
        RootAllocator {
            relevant: Relevant,
            id,
            allocations: 0,
            pd: PhantomData,
        }
    }

    /// Get memory type this allocator allocates.
    pub fn memory_type(&self) -> MemoryTypeId {
        self.id
    }
}

impl<B> MemoryAllocator<B> for RootAllocator<B>
where
    B: Backend,
{
    type Tag = ();
    type Request = ();

    fn alloc(
        &mut self,
        device: &B::Device,
        _: (),
        reqs: Requirements,
    ) -> Result<Block<B, ()>, MemoryError> {
        let memory = device.allocate_memory(self.id, reqs.size)?;
        let memory = Box::into_raw(Box::new(memory)); // Suboptimal
        self.allocations += 1;
        Ok(Block::new(memory, 0..reqs.size))
    }

    fn free(&mut self, device: &B::Device, block: Block<B, ()>) {
        assert_eq!(block.range().start, 0);
        device.free_memory(*unsafe { Box::from_raw(block.memory() as *const _ as *mut _) });
        unsafe { block.dispose() };
        self.allocations -= 1;
    }

    fn is_used(&self) -> bool {
        self.allocations != 0
    }

    fn dispose(self, _: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            self.relevant.dispose();
            Ok(())
        }
    }
}
