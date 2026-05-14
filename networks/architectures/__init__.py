from .unet import UNet, UNetOccupancy

architectures = {
    'unet': UNet,
    'unet_occ': UNetOccupancy,
}