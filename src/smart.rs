use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::{Backend, MemoryProperties, MemoryType, MemoryTypeId};
use gfx_hal::memory::{Properties, Requirements};

use {MemoryAllocator, MemoryError};
use block::Block;
use combined::{CombinedAllocator, CombinedBlock, Type};

/// Allocator that can choose memory type based on requirements, and keeps track of allocators
/// for all given memory types.
///
/// Allocates memory blocks from the least used memory type from those which satisfy requirements.
#[derive(Debug)]
pub struct SmartAllocator<B: Backend> {
    allocators: Vec<(MemoryType, CombinedAllocator<B>)>,
    heaps: Vec<Heap>,
}

impl<B> SmartAllocator<B>
where
    B: Backend,
{
    /// Create a new smart allocator from `MemoryProperties` given by a device.
    ///
    /// ### Parameters:
    ///
    /// - `memory_properties`: memory properties describing the memory available on a device
    /// - `arena_size`: see `ArenaAllocator`
    /// - `blocks_per_chunk`: see `ChunkedAllocator`
    /// - `min_block_size`: see `ChunkedAllocator`
    /// - `max_chunk_size`: see `ChunkedAllocator`
    pub fn new(
        memory_properties: MemoryProperties,
        arena_size: u64,
        blocks_per_chunk: usize,
        min_block_size: u64,
        max_chunk_size: u64,
    ) -> Self {
        SmartAllocator {
            allocators: memory_properties
                .memory_types
                .into_iter()
                .enumerate()
                .map(|(index, memory_type)| {
                    (
                        memory_type,
                        CombinedAllocator::new(
                            MemoryTypeId(index),
                            arena_size,
                            blocks_per_chunk,
                            min_block_size,
                            max_chunk_size,
                        ),
                    )
                })
                .collect(),
            heaps: memory_properties
                .memory_heaps
                .into_iter()
                .map(|size| Heap { size, used: 0 })
                .collect(),
        }
    }

    /// Get properties of the block
    pub fn properties(&self, block: &SmartBlock<B::Memory>) -> Properties {
        self.allocators[block.1].0.properties
    }
}

impl<B> MemoryAllocator<B> for SmartAllocator<B>
where
    B: Backend,
{
    type Request = (Type, Properties);
    type Block = SmartBlock<B::Memory>;

    fn alloc(
        &mut self,
        device: &B::Device,
        (ty, prop): (Type, Properties),
        reqs: Requirements,
    ) -> Result<SmartBlock<B::Memory>, MemoryError> {
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
                let block = self.allocators[chosen].1.alloc(device, ty, reqs)?;
                self.heaps[self.allocators[chosen].0.heap_index].alloc(block.size());
                Ok(SmartBlock(block, chosen))
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

    fn free(&mut self, device: &B::Device, block: SmartBlock<B::Memory>) {
        let SmartBlock(block, index) = block;
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

/// `Block` type returned by `SmartAllocator`.
#[derive(Debug)]
pub struct SmartBlock<M>(CombinedBlock<M>, usize);

impl<M> Block for SmartBlock<M>
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
        foo::<SmartAllocator<B>>()
    }
}
