import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import (ResNet, BasicBlock, Bottleneck, 
                                       ResNet34_Weights, ResNet50_Weights,
                                       conv1x1, conv3x3)
from typing import Optional, Callable, Type, Union
from torch import Tensor

import math


########################## LoRA

class LoRALayer:
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, # input
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling, # weight
                self.conv.bias # bias
            )
        return self.conv(x)


########################## Adapter

class AdapterLayer(nn.Module):
    def __init__(self, dim: int, r: int):
        super().__init__()
        self.down_proj = nn.Conv2d(dim, dim // r, kernel_size=1)
        self.gelu = nn.GELU()
        self.up_proj = nn.Conv2d(dim // r, dim, kernel_size=1)
        
    def forward(self, x):
        out = self.down_proj(x)
        out = self.gelu(out)
        out = self.up_proj(out)

        return out + x
    

########################## ResNet Blocks

class BasicBlockPEFT(BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lora_r: int = 0,
        adapter_r: int = 0,
    ) -> None:
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        if lora_r > 0:
            self.conv1 = ConvLoRA(
                nn.Conv2d, inplanes, planes, kernel_size=3, r=lora_r, padding=1, stride=stride
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if lora_r > 0:
            self.conv2 = ConvLoRA(
                nn.Conv2d, planes, planes, kernel_size=3, r=lora_r, padding=1
            )
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        if adapter_r > 0:
            self.adapter = AdapterLayer(planes * self.expansion, adapter_r)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if hasattr(self, 'adapter'):
            out = self.adapter(out)
        out += identity
        out = self.relu(out)

        return out

class BottleneckPEFT(Bottleneck):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lora_r: int = 0,
        adapter_r: int = 0,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if lora_r > 0:
            self.conv1 = ConvLoRA(nn.Conv2d, inplanes, width, kernel_size=1, r=lora_r)
        else:
            self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if lora_r > 0:
            self.conv2 = ConvLoRA(nn.Conv2d, width, width, kernel_size=3, r=lora_r, padding=1, stride=stride, groups=groups, dilation=dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        if lora_r > 0:
            self.conv3 = ConvLoRA(nn.Conv2d, width, planes * self.expansion, kernel_size=1, r=lora_r,) 
        else:
            self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if adapter_r > 0:
            self.adapter = AdapterLayer(planes * self.expansion, adapter_r)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if hasattr(self, 'adapter'):
            out = self.adapter(out)
        out += identity
        out = self.relu(out)

        return out
    

class ResNetPEFT(ResNet):
    def __init__(self, *args, lora_r: int = 0, adapter_r: int = 0, **kwargs
    ) -> None:
        self.lora_r = lora_r
        self.adapter_r = adapter_r
        super().__init__(*args, **kwargs)
        

    def _make_layer(
        self,
        block: Type[Union[BasicBlockPEFT, BottleneckPEFT]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                lora_r=self.lora_r, adapter_r=self.adapter_r
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    lora_r=self.lora_r, 
                    adapter_r=self.adapter_r
                )
            )

        return nn.Sequential(*layers)

#### model return method

def resnet34_peft(lora_r, adapter_r):
    weights = ResNet34_Weights.DEFAULT
    model = ResNetPEFT(BasicBlockPEFT,
                    [3, 4, 6, 3],
                    num_classes=1000,
                    lora_r=lora_r,
                    adapter_r=adapter_r,
                    )
    model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
    return model

def resnet50_peft(lora_r, adapter_r):
    weights = ResNet50_Weights.DEFAULT
    model = ResNetPEFT(BottleneckPEFT,
                   [3, 4, 6, 3],
                   num_classes=1000,
                   lora_r=lora_r,
                   adapter_r=adapter_r,
                   )
    model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
    return model