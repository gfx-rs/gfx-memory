//! Memory management for gfx_hal.
//!

#![deny(dead_code)]
#![deny(missing_docs)]
#![deny(unused_imports)]
#![deny(unused_must_use)]

extern crate gfx_hal;
extern crate relevant;

use std::cmp::Eq;
use std::error::Error;
use std::fmt::{self, Debug};
use std::ops::{Add, Rem, Sub};

use gfx_hal::Backend;
use gfx_hal::device::OutOfMemory;
use gfx_hal::memory::Requirements;

mod arena;
mod block;
mod chunked;
mod combined;
mod factory;
mod root;
mod smart;

pub use arena::ArenaAllocator;
pub use block::{Block, TaggedBlock};
pub use chunked::ChunkedAllocator;
pub use combined::{CombinedAllocator, Type};
pub use root::RootAllocator;
pub use smart::SmartAllocator;

/// Possible errors that may be returned from allocators.
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Allocator doesn't have compatible memory types.
    NoCompatibleMemoryType,

    /// All compatible memory is exhausted.
    OutOfMemory,
}

impl From<OutOfMemory> for MemoryError {
    fn from(_: OutOfMemory) -> Self {
        MemoryError::OutOfMemory
    }
}

impl fmt::Display for MemoryError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(self.description())
    }
}

impl Error for MemoryError {
    fn description(&self) -> &str {
        match *self {
            MemoryError::NoCompatibleMemoryType => "No compatible memory",
            MemoryError::OutOfMemory => "Out of memory",
        }
    }
}

/// Trait that manage memory allocations from `Device`.
pub trait MemoryAllocator<B: Backend>: Debug {
    /// Information required to allocate block.
    type Request;

    /// Allocator will allocate blocks of this type.
    type Block: Block<B> + Debug + Send + Sync;

    /// Allocate block of memory.
    fn alloc(
        &mut self,
        device: &B::Device,
        request: Self::Request,
        reqs: Requirements,
    ) -> Result<Self::Block, MemoryError>;

    /// Free block of memory.
    /// TaggedBlock must be allocated from this allocator.
    /// Device must be the same that was used during allocation.
    fn free(&mut self, device: &B::Device, block: Self::Block);

    /// Check if not all blocks allocated from this allocator are freed.
    /// If this function returns `false` than subsequent call to `dispose` must return `Ok(())`
    fn is_used(&self) -> bool;

    /// Try to dispose of this allocator.
    /// It will result in `Err(self)` if is in use.
    /// Allocators have to be disposed, dropping them might result in a panic.
    fn dispose(self, device: &B::Device) -> Result<(), Self>
    where
        Self: Sized;
}

/// Trait that allows to sub-allocate memory blocks from another allocator.
pub trait MemorySubAllocator<B: Backend> {
    /// Type of allocator for this sub-allocator.
    type Owner;

    /// Information required to allocate block.
    type Request;

    /// Allocator will allocate blocks of this type.
    type Block: Block<B> + Debug + Send + Sync;

    /// Allocate block of memory from this allocator.
    /// This allocator will use `owner` to get memory in bigger chunks.
    /// `owner` must always be the same for an instance.
    /// Memory of allocated blocks will satisfy requirements.
    /// `request` may contain additional requirements and/or hints for allocation.
    fn alloc(
        &mut self,
        owner: &mut Self::Owner,
        device: &B::Device,
        request: Self::Request,
        reqs: Requirements,
    ) -> Result<Self::Block, MemoryError>;

    /// Free block of memory.
    /// TaggedBlock must be allocated from this allocator.
    /// Device must be the same that was used during allocation.
    /// It may choose to free inner block allocated from `owner`.
    fn free(&mut self, owner: &mut Self::Owner, device: &B::Device, block: Self::Block);

    /// Check if not all blocks allocated from this allocator are freed.
    /// If this function returns `false` than subsequent call to `dispose` must return `Ok(())`
    fn is_used(&self) -> bool;

    /// Try to dispose of this allocator.
    /// It will result in `Err(self)` if is in use.
    /// Allocators usually will panic on drop.
    fn dispose(self, owner: &mut Self::Owner, device: &B::Device) -> Result<(), Self>
    where
        Self: Sized;
}

/// Calculate shift from specified offset required to satisfy alignment.
pub fn alignment_shift<T>(alignment: T, offset: T) -> T
where
    T: From<u8> + Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Eq + Copy,
{
    if offset == 0.into() {
        0.into()
    } else {
        alignment - (offset - 1.into()) % alignment - 1.into()
    }
}

/// Shift from specified offset in order to to satisfy alignment.
pub fn shift_for_alignment<T>(alignment: T, offset: T) -> T
where
    T: From<u8> + Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Eq + Copy,
{
    offset + alignment_shift(alignment, offset)
}
