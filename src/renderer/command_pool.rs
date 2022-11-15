use crate as app;
use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

pub unsafe fn create_command_pools(
    instance: &app::Instance,
    device: &app::Device,
    data: &mut app::AppData,
) -> Result<()> {
    data.command_pool = create_command_pool(instance, device, data)?;

    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}

pub unsafe fn create_command_pool(
    instance: &app::Instance,
    device: &app::Device,
    data: &mut app::AppData,
) -> Result<vk::CommandPool> {
    let indices = app::QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}