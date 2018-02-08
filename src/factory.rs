use std::borrow::{Borrow, BorrowMut};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::buffer::{CreationError as BufferCreationError, Usage as BufferUsage};
use gfx_hal::format::Format;
use gfx_hal::image::{CreationError as ImageCreationError, Kind, Level, Usage as ImageUsage};

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
    type Buffer: BorrowMut<B::Buffer> + Block<B>;

    /// Type of images this factory produce.
    /// The user can borrow the raw image.
    type Image: BorrowMut<B::Image> + Block<B>;

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
    fn create_buffer(
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
    fn create_image(
        &mut self,
        device: &B::Device,
        request: Self::ImageRequest,
        kind: Kind,
        level: Level,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Self::Image, Self::Error>;

    /// Destroy a buffer created by this factory.
    ///
    /// ### Parameters:
    ///
    /// - `device`: device the buffer was created on
    /// - `buffer`: the buffer to destroy
    fn destroy_buffer(&mut self, device: &B::Device, buffer: Self::Buffer);

    /// Destroy image created by this factory.
    ///
    /// ### Parameters:
    ///
    /// - `device`: device the image was created on
    /// - `image`: the image to destroy
    fn destroy_image(&mut self, device: &B::Device, image: Self::Image);
}

/// Memory resource produced by the blanket `MemoryAllocator` as `Factory` implementation.
///
/// ### Type parameters:
///
/// - `I`: Item type produced by the `Factory` (hal `Buffer` or `Image`)
/// - `B`: Memory block type (see `Block`)
#[derive(Debug)]
pub struct Item<I, T> {
    raw: I,
    size: u64,
    block: T,
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

impl<B, I, T> Block<B> for Item<I, T>
where
    B: Backend,
    T: Block<B>,
    I: Debug + Send + Sync,
{
    fn memory(&self) -> &B::Memory {
        self.block.memory()
    }

    fn range(&self) -> Range<u64> {
        let offset = self.block.range().start;
        offset..offset + self.size
    }
}

/// Possible errors that may be returned from the blanket `MemoryAllocator` as `Factory`
/// implementation.
#[derive(Debug)]
pub enum FactoryError {
    /// Memory error.
    MemoryError(MemoryError),

    /// Buffer creation error.
    BufferCreationError(BufferCreationError),

    /// Image creation error.
    ImageCreationError(ImageCreationError),
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

impl Display for FactoryError {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        match *self {
            FactoryError::MemoryError(ref error) => write!(fmt, "{}", error),
            FactoryError::BufferCreationError(ref error) => write!(fmt, "{}", error),
            FactoryError::ImageCreationError(ref error) => write!(fmt, "{}", error),
        }
    }
}

impl Error for FactoryError {
    fn description(&self) -> &str {
        match *self {
            FactoryError::MemoryError(_) => "Memory error in factory",
            FactoryError::BufferCreationError(_) => "Buffer creation error in factory",
            FactoryError::ImageCreationError(_) => "Image creation error in factory",
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            FactoryError::MemoryError(ref error) => Some(error),
            FactoryError::BufferCreationError(ref error) => Some(error),
            FactoryError::ImageCreationError(ref error) => Some(error),
        }
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

    fn create_buffer(
        &mut self,
        device: &B::Device,
        request: A::Request,
        size: u64,
        usage: BufferUsage,
    ) -> Result<Item<B::Buffer, A::Block>, FactoryError> {
        let ubuf = device.create_buffer(size, usage)?;
        let reqs = device.get_buffer_requirements(&ubuf);
        let block = self.alloc(device, request, reqs)?;
        let buf = device
            .bind_buffer_memory(block.memory(), block.range().start, ubuf)
            .unwrap();
        Ok(Item {
            raw: buf,
            block,
            size,
        })
    }

    fn create_image(
        &mut self,
        device: &B::Device,
        request: A::Request,
        kind: Kind,
        level: Level,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Item<B::Image, A::Block>, FactoryError> {
        let uimg = device.create_image(kind, level, format, usage)?;
        let reqs = device.get_image_requirements(&uimg);
        let block = self.alloc(device, request, reqs)?;
        let img = device
            .bind_image_memory(block.memory(), block.range().start, uimg)
            .unwrap();
        Ok(Item {
            raw: img,
            block,
            size: reqs.size,
        })
    }

    fn destroy_buffer(&mut self, device: &B::Device, buffer: Self::Buffer) {
        device.destroy_buffer(buffer.raw);
        self.free(device, buffer.block);
    }

    fn destroy_image(&mut self, device: &B::Device, image: Self::Image) {
        device.destroy_image(image.raw);
        self.free(device, image.block);
    }
}
