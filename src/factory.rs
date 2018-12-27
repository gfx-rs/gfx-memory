use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::buffer::{CreationError as BufferCreationError, Usage as BufferUsage};
use gfx_hal::device::BindError;
use gfx_hal::format::Format;
use gfx_hal::image::{
    CreationError as ImageCreationError, Kind, Level, Tiling, Usage as ImageUsage, ViewCapabilities,
};
use gfx_hal::{Backend, Device};

use block::Block;

use {MemoryAllocator, MemoryError};

/// Factory trait used to create buffers and images and manage the memory for them.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
pub trait Factory<B: Backend> {
    /// Type of buffers this factory produce.
    /// The user can borrow the raw buffer.
    type Buffer: BorrowMut<B::Buffer> + Block<Memory = B::Memory>;

    /// Type of images this factory produce.
    /// The user can borrow the raw image.
    type Image: BorrowMut<B::Image> + Block<Memory = B::Memory>;

    /// Information required to produce a buffer.
    type BufferRequest;

    /// Information required to produce an image.
    type ImageRequest;

    /// Error type this factory can yield.
    type Error;

    /// Create a buffer with the specified size and usage.
    ///
    /// ### Parameters
    ///
    /// - `device`: device to create the buffer on
    /// - `request`: information needed by the `MemoryAllocator` to allocate a block of memory for
    ///              the buffer
    /// - `size`: size in bytes of the buffer
    /// - `usage`: hal buffer `Usage`
    unsafe fn create_buffer(
        &mut self,
        device: &B::Device,
        request: Self::BufferRequest,
        size: u64,
        usage: BufferUsage,
    ) -> Result<Self::Buffer, Self::Error>;

    /// Create an image with the specified kind, level, format and usage.
    ///
    /// ### Parameters:
    ///
    /// - `device`: device to create the image on
    /// - `request`: information needed by the `MemoryAllocator` to allocate a block of memory for
    ///              the image
    /// - `kind`: `Kind` of texture storage to allocate
    /// - `level`: mipmap level
    /// - `format`: texture format
    /// - `usage`: hal image usage
    unsafe fn create_image(
        &mut self,
        device: &B::Device,
        request: Self::ImageRequest,
        kind: Kind,
        level: Level,
        format: Format,
        tiling: Tiling,
        usage: ImageUsage,
        view_caps: ViewCapabilities,
    ) -> Result<Self::Image, Self::Error>;

    /// Destroy a buffer created by this factory.
    ///
    /// ### Parameters:
    ///
    /// - `device`: device the buffer was created on
    /// - `buffer`: the buffer to destroy
    unsafe fn destroy_buffer(&mut self, device: &B::Device, buffer: Self::Buffer);

    /// Destroy image created by this factory.
    ///
    /// ### Parameters:
    ///
    /// - `device`: device the image was created on
    /// - `image`: the image to destroy
    unsafe fn destroy_image(&mut self, device: &B::Device, image: Self::Image);
}

/// Memory resource produced by the blanket `MemoryAllocator` as `Factory` implementation.
///
/// ### Type parameters:
///
/// - `I`: Item type produced by the `Factory` (hal `Buffer` or `Image`)
/// - `T`: Memory block type (see `Block`)
#[derive(Debug)]
pub struct Item<I, T> {
    raw: I,
    block: T,
}

impl<I, T> Item<I, T> {
    /// Get raw item.
    pub fn raw(&self) -> &I {
        &self.raw
    }
}

impl<I, T> Item<I, T> {
    /// Get block of the item.
    pub fn block(&self) -> &T {
        &self.block
    }
}

impl<I, T> Borrow<I> for Item<I, T> {
    fn borrow(&self) -> &I {
        &self.raw
    }
}

impl<I, T> BorrowMut<I> for Item<I, T> {
    fn borrow_mut(&mut self) -> &mut I {
        &mut self.raw
    }
}

impl<I, T> Block for Item<I, T>
where
    I: Debug + Send + Sync,
    T: Block,
{
    type Memory = T::Memory;

    fn memory(&self) -> &T::Memory {
        self.block.memory()
    }

    fn range(&self) -> Range<u64> {
        self.block.range()
    }
}

/// Possible errors that may be returned from the blanket `MemoryAllocator` as `Factory`
/// implementation.
#[derive(Debug, Clone, Fail)]
pub enum FactoryError {
    /// Memory error.
    #[fail(display = "Memory error")]
    MemoryError(#[cause] MemoryError),

    /// Binding of a buffer or image object failed.
    #[fail(display = "Bind error")]
    BindError(#[cause] BindError),

    /// Buffer creation error.
    #[fail(display = "Failed to create buffer")]
    BufferCreationError(#[cause] BufferCreationError),

    /// Image creation error.
    #[fail(display = "Failed to create image")]
    ImageCreationError(#[cause] ImageCreationError),
}

impl From<MemoryError> for FactoryError {
    fn from(error: MemoryError) -> Self {
        FactoryError::MemoryError(error)
    }
}

impl From<BufferCreationError> for FactoryError {
    fn from(error: BufferCreationError) -> Self {
        FactoryError::BufferCreationError(error)
    }
}

impl From<ImageCreationError> for FactoryError {
    fn from(error: ImageCreationError) -> Self {
        FactoryError::ImageCreationError(error)
    }
}

impl From<BindError> for FactoryError {
    fn from(error: BindError) -> Self {
        FactoryError::BindError(error)
    }
}

impl<B, A> Factory<B> for A
where
    B: Backend,
    A: MemoryAllocator<B>,
{
    type Buffer = Item<B::Buffer, A::Block>;
    type Image = Item<B::Image, A::Block>;
    type BufferRequest = A::Request;
    type ImageRequest = A::Request;
    type Error = FactoryError;

    unsafe fn create_buffer(
        &mut self,
        device: &B::Device,
        request: A::Request,
        size: u64,
        usage: BufferUsage,
    ) -> Result<Item<B::Buffer, A::Block>, FactoryError> {
        let mut buf = device.create_buffer(size, usage)?;
        let reqs = device.get_buffer_requirements(&buf);
        let block = self.alloc(device, request, reqs)?;
        device.bind_buffer_memory(block.memory(), block.range().start, &mut buf)?;
        Ok(Item { raw: buf, block })
    }

    unsafe fn create_image(
        &mut self,
        device: &B::Device,
        request: A::Request,
        kind: Kind,
        level: Level,
        format: Format,
        tiling: Tiling,
        usage: ImageUsage,
        view_caps: ViewCapabilities,
    ) -> Result<Item<B::Image, A::Block>, FactoryError> {
        let mut img = device.create_image(kind, level, format, tiling, usage, view_caps)?;
        let reqs = device.get_image_requirements(&img);
        let block = self.alloc(device, request, reqs)?;
        device.bind_image_memory(block.memory(), block.range().start, &mut img)?;
        Ok(Item { raw: img, block })
    }

    unsafe fn destroy_buffer(&mut self, device: &B::Device, buffer: Self::Buffer) {
        device.destroy_buffer(buffer.raw);
        self.free(device, buffer.block);
    }

    unsafe fn destroy_image(&mut self, device: &B::Device, image: Self::Image) {
        device.destroy_image(image.raw);
        self.free(device, image.block);
    }
}
