use std::ops::Deref;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::VulkanLibrary;

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

    // let data: i32 = 12;

    #[derive(BufferContents)]
    #[repr(C)]
    struct MyBufferStruct {
        a: u32,
        b: u32,
    }
    let data = MyBufferStruct{a: 1, b: 2};
    let buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
    )
    .expect("Failed to create buffer!");
}
