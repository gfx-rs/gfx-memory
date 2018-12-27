use std::marker::PhantomData;

use gfx_hal::memory::Requirements;
use gfx_hal::{Backend, Device, MemoryTypeId};

use block::{Block, RawBlock};
use relevant::Relevant;
use {MemoryAllocator, MemoryError};

/// Allocator that allocates memory directly from device.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
#[derive(Debug)]
pub struct RootAllocator<B> {
    relevant: Relevant,
    id: MemoryTypeId,
    used: u64,
    pd: PhantomData<fn() -> B>,
}

impl<B> RootAllocator<B> {
    /// Create new allocator that will allocate memory of specified type.
    ///
    /// ### Parameters:
    ///
    /// - `id`: ID of the memory type this allocator allocates from.
    pub fn new(id: MemoryTypeId) -> Self {
        RootAllocator {
            relevant: Relevant,
            id,
            used: 0,
            pd: PhantomData,
        }
    }

    /// Get memory type this allocator allocates.
    pub fn memory_type(&self) -> MemoryTypeId {
        self.id
    }

    /// Get the total size of all blocks allocated by this allocator.
    pub fn used(&self) -> u64 {
        self.used
    }
}

impl<B> MemoryAllocator<B> for RootAllocator<B>
where
    B: Backend,
{
    type Request = ();
    type Block = RawBlock<B::Memory>;

    unsafe fn alloc(
        &mut self,
        device: &B::Device,
        _: (),
        reqs: Requirements,
    ) -> Result<RawBlock<B::Memory>, MemoryError> {
        let memory = device.allocate_memory(self.id, reqs.size)?;
        let memory = Box::into_raw(Box::new(memory)); // Suboptimal
        self.used += reqs.size;
        Ok(RawBlock::new(memory, 0..reqs.size))
    }

    unsafe fn free(&mut self, device: &B::Device, block: RawBlock<B::Memory>) {
        let size = block.size();
        assert_eq!(block.range().start, 0);
        device.free_memory(*Box::from_raw(block.memory() as *const _ as *mut _));
        block.dispose();
        self.used -= size;
    }

    fn is_used(&self) -> bool {
        self.used != 0
    }

    unsafe fn dispose(self, _: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            self.relevant.dispose();
            Ok(())
        }
    }
}

#[test]
#[allow(dead_code)]
fn test_send_sync() {
    fn foo<T: Send + Sync>() {}
    fn bar<B: Backend>() {
        foo::<RootAllocator<B>>()
    }
}
