
# gfx-memory - graphics memory management for gfx-hal.
[![Build Status](https://travis-ci.org/gfx-rs/gfx-memory.svg)](https://travis-ci.org/gfx-rs/gfx-memory)
[![Docs](https://docs.rs/gfx-memory/badge.svg)](https://docs.rs/gfx-memory)
[![Crates.io](https://img.shields.io/crates/v/gfx-memory.svg?maxAge=2592000)](https://crates.io/crates/gfx-memory)

This crate provides tools to manage GPU memory provided by `gfx-hal`.

The main tool is the `MemoryAllocator` trait, which can be used to allocate `Block`s of memory.
The most notable `MemoryAllocator` implementation is `SmartAllocator` which can be used as-is.
All other allocators in this crate are used internally in `SmartAllocator`, but are also exposed 
for users who want to create their own implementations in case `SmartAllocator` don't satisfy their needs.

A `Factory` is also provided, that wraps the allocation logic in this crate, along with creation of memory resources
on a `Device` (such as `Buffer` or `Image`). For most use cases, the `Factory` provides all capabilities needed to 
manage memory based resources on a `gfx_hal::Device`.

### Example 

Simple example of using `SmartAllocator` to create a vertex `Buffer`:

```rust
extern crate gfx_hal;
extern crate gfx_memory;

use std::error::Error;

use gfx_hal::{Backend, Device};
use gfx_hal::buffer::Usage;
use gfx_hal::memory::Properties;
use gfx_memory::{MemoryAllocator, SmartAllocator, Type, Block};

type SmartBlock<B> = <SmartAllocator<B> as MemoryAllocator<B>>::Block;

fn make_vertex_buffer<B: Backend>(
    device: &B::Device,
    allocator: &mut SmartAllocator<B>,
    size: u64,
) -> Result<(SmartBlock<B>, B::Buffer), Box<Error>> {
    // Create unbounded buffer object. It has no memory assigned.
    let ubuf: B::UnboundBuffer = device.create_buffer(size, Usage::VERTEX).map_err(Box::new)?;
    // Get memory requirements for the buffer.
    let reqs = device.get_buffer_requirements(&ubuf);
    // Allocate block of device-local memory that satisfy requirements for buffer.
    let block = allocator
        .alloc(device, (Type::General, Properties::DEVICE_LOCAL), reqs)
        .map_err(Box::new)?;
    // Bind memory block to the buffer.
    Ok(device
        .bind_buffer_memory(block.memory(), block.range().start, ubuf)
        .map(|buffer| (block, buffer))
        .map_err(Box::new)?)
}

```

This crate is mid-level and it requires the user to follow a few simple rules:

* When memory blocks are to be freed, they must be returned to the allocator they were allocated from.
* The same instance of `Device` must be used for allocating and freeing blocks.

Violating those rules may cause undefined behaviour.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

We are a community project that welcomes contribution from anyone. If you're interested in helping out, you can contact 
us either through GitHub, or via [`gitter`](https://gitter.im/gfx-rs/gfx).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
