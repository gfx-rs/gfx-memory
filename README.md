
# gfx_mem - graphics memory management for gfx_hal.

This crate provides tools to manage GPU memory provided by `gfx_hal`.
The main tool is `MemoryAllocator` trait. Which can be used to allocate `Block`s of memory.
Most notable `MemoryAllocator` implementation is `SmartAllocator` which can be used as-is in application.
Other allocators in this crate are used internally in `SmartAllocator`.
They are exposed for users who would want to create their own implementations
in case `SmartAllocator` doesn't satisfy their needs.

Simple example of using `SmartAllocator` to create a `Buffer`:

```rust
extern crate gfx_hal;
extern crate gfx_mem;

use gfx_hal::{Backend, Device};
use gfx_hal::buffer::Usage;
use gfx_hal::memory::Properties;
use gfx_mem::{SmartAllocator, Type};

fn make_vertex_buffer<B: Backend>(device: &B::Device, allocator: &mut Allocator, size: u64) -> Result<B::Buffer, Box<Error>> {
    // Create unbounded buffer object. It has no memory assigned.
    let ubuf: B::UnboundBuffer = device.create_buffer(size, Usage::VERTEX).map_err(Box::new)?;
    // Ger memory requirements for the buffer.
    let reqs = device.get_buffer_requirements(&ubuf);
    // Allocate block of device-local memory that satisfy requirements for buffer.
    let block = allocator.alloc(device, (Type::General, Properties::DEVICE_LOCAL), reqs).map_err(Box::new)?;
    // Bind memory block to the buffer.
    device.bind_buffer_memory(block.memory(), block.range().start, ubuf).map_err(Box::new)
}
```

This crate is mid-level as it requires user to follow simple rules.
* Free memory block by returning it to the allocator it was allocated from.
* Use same instance of `Device` for allocating and deallocating blocks.

Violating those rules may cause an undefined behaviour.