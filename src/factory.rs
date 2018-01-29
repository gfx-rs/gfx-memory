use std::borrow::{Borrow, BorrowMut};
use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::buffer::{Usage as BufferUsage, CreationError as BufferCreationError};
use gfx_hal::format::Format;
use gfx_hal::image::{Kind, Level, Usage as ImageUsage, CreationError as ImageCreationError};

use block::Block;

use {MemoryAllocator, MemoryError};

/// Trait that allows to create buffers and images and manages memory for them.
pub trait Factory<B: Backend> {
    /// Type of buffers this factory produces.
    /// User may borrow raw buffer from it.
    type Buffer: BorrowMut<B::Buffer> + Block<B>;

    /// Type of images this factory produces.
    /// User may borrow raw image from it.
    type Image: BorrowMut<B::Image> + Block<B>;

    /// Information required to produce buffer.
    type BufferRequest;

    /// Information required to produce image.
    type ImageRequest;

    /// Error type this factory can yield.
    type Error;

    /// Create buffer with specified size and usage.
    fn create_buffer(
        &mut self,
        device: &B::Device,
        request: Self::BufferRequest,
        size: u64,
        usage: BufferUsage,
    ) -> Result<Self::Buffer, Self::Error>;

    /// Create image with specified kind, level, format and usage.
    fn create_image(
        &mut self,
        device: &B::Device,
        request: Self::ImageRequest,
        kind: Kind,
        level: Level,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Self::Image, Self::Error>;

    fn destroy_buffer(
        &mut self,
        device: &B::Device,
        buffer: Self::Buffer,
    );

    fn destroy_image(
        &mut self,
        device: &B::Device,
        image: Self::Image,
    );
}

#[derive(Debug)]
pub struct Item<I, B> {
    raw: I,
    block: B,
}

impl<I, B> Borrow<I> for Item<I, B> {
    fn borrow(&self) -> &I {
        &self.raw
    }
}

impl<I, B> BorrowMut<I> for Item<I, B> {
    fn borrow_mut(&mut self) -> &mut I {
        &mut self.raw
    }
}

impl<X, I, B> Block<X> for Item<I, B>
where
    X: Backend,
    B: Block<X>,
{
    fn memory(&self) -> &X::Memory {
        self.block.memory()
    }

    fn range(&self) -> Range<u64> {
        self.block.range()
    }
}


/// Possible errors that may be returned from allocator-as-factory
pub enum FactoryError {
    MemoryError(MemoryError),
    BufferCreationError(BufferCreationError),
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
    ) -> Result<Item<B::Buffer, A::Block>, FactoryError>
    {
        let ubuf = device.create_buffer(size, usage)?;
        let reqs = device.get_buffer_requirements(&ubuf);
        let block = self.alloc(device, request, reqs)?;
        let buf = device.bind_buffer_memory(block.memory(), block.range().start, ubuf).unwrap();
        Ok(Item {
            raw: buf,
            block,
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
        let img = device.bind_image_memory(block.memory(), block.range().start, uimg).unwrap();
        Ok(Item {
            raw: img,
            block,
        })
    }

    fn destroy_buffer(
        &mut self,
        device: &B::Device,
        buffer: Self::Buffer,
    )
    {
        device.destroy_buffer(buffer.raw);
        self.free(device, buffer.block);
    }

    fn destroy_image(
        &mut self,
        device: &B::Device,
        image: Self::Image,
    )
    {
        device.destroy_image(image.raw);
        self.free(device, image.block);
    }
}