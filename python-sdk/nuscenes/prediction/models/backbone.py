# nuScenes dev-kit.
# Code written by Freddy Boulton 2020.
from typing import Tuple

import os
import torch
import pretrainedmodels
from torch import nn
from torchvision.models import (mobilenet_v2, resnet18, resnet34, resnet50,
                                resnet101, resnet152)


def set_linear_layer_ss(model, output_dim):
    """
    Set the final linear for models loaded from
    facebookresearch/semi-supervised-ImageNet1K-models
    """
    dim_feats = model.fc.in_features
    model.fc = nn.Linear(dim_feats, output_dim)
    return model


def load_ss_imagenet_models(model_key):
    return torch.hub.load(
        'facebookresearch/semi-supervised-ImageNet1K-models',
        model_key)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def set_classification_layer_simclr(model, output_dim):
    dim_feats = model.l1.in_features
    model.l1 = nn.Linear(dim_feats, output_dim)
    model.l2 = Identity()
    return model


def get_simclr_model():
    from nuscenes.prediction.models.simclr_google import resnet_wider
    model = resnet_wider.resnet50x4()
    sd = torch.load(os.path.join(
        os.path.dirname(resnet_wider.__file__),
        'resnet50-4x.pth'), map_location='cpu')
    model.load_state_dict(sd['state_dict'])
    return model


def freeze_bottom_resnet(model):
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.relu.parameters():
        param.requires_grad = False
    for param in model.maxpool.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False


def freeze_bottom_simclr(model):
    for param in model.features[0].parameters():
        param.requires_grad = False
    for param in model.features[1].parameters():
        param.requires_grad = False
    for param in model.features[2].parameters():
        param.requires_grad = False
    for param in model.features[3].parameters():
        param.requires_grad = False
    for param in model.features[4].parameters():
        param.requires_grad = False
    for param in model.features[5].parameters():
        param.requires_grad = False
    for param in model.features[6].parameters():
        param.requires_grad = False
    for param in model.features[7][0].parameters():
        param.requires_grad = False


def freeze_resnext_supervised(model):
    for param in model.features[:6].parameters():
        param.requires_grad = False
    for param in model.features[6][:16].parameters():
        param.requires_grad = False


def freeze_bottom_noisy_student_efficientnet(model):
    """
    Freeze the bottom 5 of 7 blocks, and anything before the blocks
    """
    for param in model.conv_stem.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for block in model.blocks[:5]:
        for param in block.parameters():
            param.requires_grad = False


def print_frozen_params(model):
    """For debugging, print the frozen parameters"""
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    frozen_weights = len([p for p in model.parameters() if not p.requires_grad])
    total_weights = len([p for p in model.parameters()])

    print(f"Froze {frozen_params/total_params:.2f} of params, "
          f"{frozen_weights/total_weights:.2f} of weights.")


def get_pretrained_model(model_key,
                         freeze_bottom=True):
    """
    Load pretrained model for image layers.
    """

    # Models from Cadene
    if model_key in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext101_32x4d']:
        model = pretrainedmodels.__dict__[model_key](
            num_classes=1000, pretrained='imagenet')  # original
        if freeze_bottom and model_key == 'resnext101_32x4d':
            freeze_resnext_supervised(model)
        elif freeze_bottom:
            freeze_bottom_resnet(model)
    elif model_key in ['resnext101_32x4d_swsl', 'resnext101_32x4d_ssl', 'resnet50_swsl', 'resnet50_ssl']:
        model = load_ss_imagenet_models(model_key)
        if freeze_bottom:
            freeze_bottom_resnet(model)
    elif model_key == 'simclr':
        model = get_simclr_model()
        if freeze_bottom:
            freeze_bottom_resnet(model)
    else:
        model = None

    if model is not None:
        print_frozen_params(model)
    return model


def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])


def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """ Helper to calculate the shape of the fully-connected regression layer. """
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]


class Backbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x4d_ssl, resnext101_32x4d_swsl, simclr.
    """

    def __init__(self, version: str, freeze_bottom: bool=True):
        """
        Inits ResNetBackbone
        :param version: resnet version to use.
        """
        super().__init__()

        self.backbone = trim_network_at_index(get_pretrained_model(version, freeze_bottom), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048].
        """
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)


class MobileNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: mobilenet_v2.
    """

    def __init__(self, version: str):
        """
        Inits MobileNetBackbone.
        :param version: mobilenet version to use.
        """
        super().__init__()

        if version != 'mobilenet_v2':
            raise NotImplementedError(f'Only mobilenet_v2 has been implemented. Received {version}.')

        self.backbone = trim_network_at_index(mobilenet_v2(), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For mobilenet_v2,
            the shape is [batch_size, 1280].
        """
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])
