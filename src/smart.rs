use std::marker::PhantomData;
use std::ops::Range;

use gfx_hal::{Backend, MemoryProperties, MemoryType, MemoryTypeId};
use gfx_hal::memory::{Properties, Requirements};

use {MemoryAllocator, MemoryError};
use block::Block;
use combined::{CombinedAllocator, CombinedBlock};

/// Allocator that can choose memory type based on requirements, and keeps track of allocators
/// for all given memory types.
///
/// Allocates memory blocks from the least used memory type from those which satisfy requirements.
#[derive(Debug)]
pub struct GenericSmartAllocator<B: Backend, A: MemoryAllocator<B>> {
    allocators: Vec<(MemoryType, A)>,
    heaps: Vec<Heap>,
    _phantom: PhantomData<fn(B)>,
}

impl <B, A> GenericSmartAllocator<B, A>
where
    B: Backend,
    A: MemoryAllocator<B>,
{
    /// Create a new smart allocator from `MemoryProperties` given by a device, backed by a
    /// custom allocator.
    ///
    /// ### Parameters:
    ///
    /// - `memory_properties`: memory properties describing the memory available on a device
    /// - `new_allocator`: the function used to create an allocator for each memory type
    pub fn new<F: FnMut(MemoryTypeId) -> A>(
        memory_properties: MemoryProperties,
        mut new_allocator: F,
    ) -> Self {
        GenericSmartAllocator {
            allocators: memory_properties
                .memory_types
                .into_iter()
                .enumerate()
                .map(|(index, memory_type)| (memory_type, new_allocator(MemoryTypeId(index))))
                .collect(),
            heaps: memory_properties
                .memory_heaps
                .into_iter()
                .map(|size| Heap { size, used: 0 })
                .collect(),
            _phantom: PhantomData,
        }
    }

    /// Get properties of the block
    pub fn properties(&self, block: &GenericSmartBlock<A::Block>) -> Properties {
        self.allocators[block.1].0.properties
    }
}

impl<B, A> MemoryAllocator<B> for GenericSmartAllocator<B, A>
where
    B: Backend,
    A: MemoryAllocator<B>,
{
    type Request = (A::Request, Properties);
    type Block = GenericSmartBlock<A::Block>;

    fn alloc(
        &mut self,
        device: &B::Device,
        (backing_req, prop): (A::Request, Properties),
        reqs: Requirements,
    ) -> Result<GenericSmartBlock<A::Block>, MemoryError> {
        let mut compatible = false;
        let mut candidate = None;

        // Find compatible memory type with least used heap with enough available memory
        for index in 0..self.allocators.len() {
            let memory_type = self.allocators[index].0;
            // filter out non-compatible
            if ((1 << index) & reqs.type_mask) != (1 << index)
                || !memory_type.properties.contains(prop)
            {
                continue;
            }
            compatible = true;
            // filter out if heap has not enough memory available
            if self.heaps[memory_type.heap_index].available() < (reqs.size + reqs.alignment) {
                continue;
            }
            // Compare with candidate. Replace if this one is less used.
            let this_usage = self.heaps[memory_type.heap_index].usage();
            match candidate {
                Some((ref mut candidate, ref mut usage)) if *usage > this_usage => {
                    *candidate = index;
                    *usage = this_usage;
                }
                ref mut candidate @ None => *candidate = Some((index, this_usage)),
                _ => {}
            }
        }

        match candidate {
            Some((chosen, _)) => {
                // Allocate from final candidate
                let block = self.allocators[chosen].1.alloc(device, backing_req, reqs)?;
                self.heaps[self.allocators[chosen].0.heap_index].alloc(block.size());
                Ok(GenericSmartBlock(block, chosen))
            }
            None => {
                // No candidates
                Err(if !compatible {
                    MemoryError::NoCompatibleMemoryType
                } else {
                    MemoryError::OutOfMemory
                })
            }
        }
    }

    fn free(&mut self, device: &B::Device, block: GenericSmartBlock<A::Block>) {
        let GenericSmartBlock(block, index) = block;
        self.heaps[self.allocators[index].0.heap_index].free(block.size());
        self.allocators[index].1.free(device, block);
    }

    fn is_used(&self) -> bool {
        self.allocators
            .iter()
            .any(|&(_, ref allocator)| allocator.is_used())
    }

    fn dispose(mut self, device: &B::Device) -> Result<(), Self> {
        if self.is_used() {
            Err(self)
        } else {
            for (_, allocator) in self.allocators.drain(..) {
                allocator.dispose(device).unwrap();
            }
            Ok(())
        }
    }
}

#[derive(Debug)]
struct Heap {
    size: u64,
    used: u64,
}

impl Heap {
    fn available(&self) -> u64 {
        self.size - self.used
    }

    fn alloc(&mut self, size: u64) {
        self.used += size;
    }

    fn free(&mut self, size: u64) {
        self.used -= size;
    }

    fn usage(&self) -> f32 {
        self.used as f32 / self.size as f32
    }
}

/// `Block` type returned by `GenericSmartAllocator`.
#[derive(Debug)]
pub struct GenericSmartBlock<B: Block>(B, usize);

impl<B> Block for GenericSmartBlock<B>
where
    B: Block,
{
    type Memory = B::Memory;

    #[inline(always)]
    fn memory(&self) -> &B::Memory {
        self.0.memory()
    }

    #[inline(always)]
    fn range(&self) -> Range<u64> {
        self.0.range()
    }
}

/// A `GenericSmartAllocator` based on a `CombinedAllocator`
pub type SmartAllocator<B> = GenericSmartAllocator<B, CombinedAllocator<B>>;

/// A `GenericSmartBlock` based on a `CombinedBlock`
pub type SmartBlock<B> = GenericSmartBlock<CombinedBlock<B>>;

#[test]
#[allow(dead_code)]
fn test_send_sync() {
    fn foo<T: Send + Sync>() {}
    fn bar<B: Backend>() {
        foo::<SmartAllocator<B>>()
    }
}
