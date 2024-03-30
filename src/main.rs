use std::ops::Deref;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAlloc, StandardCommandBufferAllocator,
    StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::VulkanLibrary;
use vulkano::sync::{self, GpuFuture};

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL found");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..InstanceCreateInfo::default()
        },
    )
    .expect("Failed to create an instance");

    let device_extensions = DeviceExtensions {
        // TODO get swapchain working.
        // khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                    // && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|_| p)
        })
        .min_by_key(|p| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("No suitable physical device could be found.");

    println!(
        "Using device: {} (type: {:?}, driver: {})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
        physical_device.properties().driver_name.as_ref().unwrap(),
    );

    // We need to find a family of queues that support graphical operations.
    // We need a queue family in order to create a device. The create of a devices
    // returns both the created device, and a list of queues in that family we chose.
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            // provide the desired queue family by index.
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create device.");

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));


    // Any struct deriving from AnyBitPattern from bytemuck library
    // can be put in a buffer. Vulkano provides its own BufferContents macro
    // that does this.
    #[derive(BufferContents, Vertex, Debug)]
    // Any data sent through an FFI boundary should use repr(C).
    // Makes order, size and allignment of values match that of C/C++.
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.25, -0.1],
        },
    ];
    let destination_vertices: [Vertex; 3] = [
        Vertex { position: [0.0, 0.0] },
        Vertex { position: [0.0, 0.0] },
        Vertex { position: [0.0, 0.0] },
    ];
    let vertex_buffer_source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                // We are using Buffer::from_data to upload data to the buffer so require
                // that the host can accesss the buffer to upload it. Else we will need
                // to use a proxy buffer that the data is copied from.
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .expect("Failed to create buffer!");

    let vertex_buffer_destination = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                // We are using Buffer::from_data to upload data to the buffer so require
                // that the host can accesss the buffer to upload it. Else we will need
                // to use a proxy buffer that the data is copied from.
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        destination_vertices,
    )
    .expect("Failed to create buffer!");

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer(CopyBufferInfo::buffers(
            vertex_buffer_source.clone(),
            vertex_buffer_destination.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .flush()
        .unwrap();

    let src_result = vertex_buffer_source.read().unwrap();
    println!("{:?}", src_result.iter());
    let dst_result = vertex_buffer_destination.read().unwrap();
    println!("{:?}", dst_result.iter());
}
