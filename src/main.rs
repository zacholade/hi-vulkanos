use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::{swapchain, VulkanLibrary};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

fn main() {
    let event_loop = EventLoop::new();

    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL found");

    let required_extensions = Surface::required_extensions(&event_loop);

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..InstanceCreateInfo::default()
        },
    )
    .expect("Failed to create an instance");

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
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

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                // Some drivers report an `min_image_count` of 1, but fullscreen mode requires at
                // least 2. Therefore we must ensure the count is at least 2, otherwise the program
                // would crash when entering fullscreen mode on those drivers.
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // Any struct deriving from AnyBitPattern from bytemuck library
    // can be put in a buffer. Vulkano provides its own BufferContents macro
    // that does this.
    #[derive(BufferContents, Vertex, Debug, PartialEq)]
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
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
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
        // vertices,
        vertices,
    )
    .expect("Failed to create buffer!");

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            "
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            "
        }
    }
    // let command_buffer_allocator = StandardCommandBufferAllocator::new(
    //     device.clone(),
    //     StandardCommandBufferAllocatorCreateInfo::default(),
    // );
}
