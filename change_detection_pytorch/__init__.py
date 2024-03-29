from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .manet import MAnet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pan import PAN
from .stanet import STANet
from .upernet import UPerNet

from . import encoders
from . import utils
from . import losses
from . import datasets

from .__version__ import __version__

from typing import Optional
import torch


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 4,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """

    archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN, STANet, UPerNet]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
            arch, list(archs_dict.keys()),
        ))
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
