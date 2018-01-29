use gfx_hal::{Backend, MemoryProperties, MemoryType, MemoryTypeId};
use gfx_hal::memory::{Properties, Requirements};

use block::{Block, TaggedBlock};
use combined::{CombinedAllocator, Type};
use {MemoryAllocator, MemoryError};

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
}

/// Allocator that may choose memory type based on requirements.
/// It allocates from least used memory type from those which satisfy requirements.
#[derive(Debug)]
pub struct SmartAllocator<B: Backend> {
    allocators: Vec<(MemoryType, CombinedAllocator<B>)>,
    heaps: Vec<Heap>,
}

impl<B> SmartAllocator<B>
where
    B: Backend,
{
    /// Create new smart allocator from `MemoryProperties` given by adapter
    /// and paramters for sub-allocators.
    pub fn new(
        memory_properties: MemoryProperties,
        arena_size: u64,
        chunks_per_block: usize,
        min_chunk_size: u64,
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
                            chunks_per_block,
                            min_chunk_size,
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
}

impl<B> MemoryAllocator<B> for SmartAllocator<B>
where
    B: Backend,
{
    type Request = (Type, Properties);
    type Block = TaggedBlock<B, Tag>;

    fn alloc(
        &mut self,
        device: &B::Device,
        (ty, prop): (Type, Properties),
        reqs: Requirements,
    ) -> Result<TaggedBlock<B, Tag>, MemoryError> {
        let ref mut heaps = self.heaps;
        let allocators = self.allocators.iter_mut().enumerate();

        let mut compatible_count = 0;
        let (index, &mut (memory_type, ref mut allocator)) = allocators
            .filter(|&(index, &mut (ref memory_type, _))| {
                ((1 << index) & reqs.type_mask) == (1 << index)
                    && memory_type.properties.contains(prop)
            })
            .filter(|&(_, &mut (ref memory_type, _))| {
                compatible_count += 1;
                heaps[memory_type.heap_index].available() >= (reqs.size + reqs.alignment)
            })
            .next()
            .ok_or(MemoryError::from(if compatible_count == 0 {
                MemoryError::NoCompatibleMemoryType
            } else {
                MemoryError::OutOfMemory
            }))?;

        let block = allocator.alloc(device, ty, reqs)?;
        heaps[memory_type.heap_index].alloc(block.size());

        Ok(block.convert_tag(|tag| Tag(index, tag)))
    }

    fn free(&mut self, device: &B::Device, block: TaggedBlock<B, Tag>) {
        let (block, Tag(index, tag)) = block.take_tag();
        self.heaps[self.allocators[index].0.heap_index].free(block.size());
        self.allocators[index].1.free(device, block.set_tag(tag));
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

/// Opaque type for `TaggedBlock` tag.
/// `ChunkedAllocator` places this tag and than uses it in `MemorySubAllocator::free` method.
#[derive(Debug, Clone, Copy)]
pub struct Tag(usize, ::combined::Tag);
