import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

# https://github.com/kenshohara/3D-ResNets-PyTorchより


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class GlobalAveragePooling3D(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return F.avg_pool3d(x, kernel_size=x.size()[2:]).view(batch_size, -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_classes=400,
                 in_channels=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = GlobalAveragePooling3D()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_until_avgpool(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward_until_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], shortcut_type, num_classes, in_channels)
    return model


def resnet18(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], shortcut_type, num_classes, in_channels)
    return model


def resnet34(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], shortcut_type, num_classes, in_channels)
    return model


def resnet50(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], shortcut_type, num_classes, in_channels)
    return model


def resnet101(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], shortcut_type, num_classes, in_channels)
    return model


def resnet152(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], shortcut_type, num_classes, in_channels)
    return model


def resnet200(shortcut_type, num_classes, in_channels):
    """Constructs a ResNet-200 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], shortcut_type, num_classes, in_channels)
    return model



class Flatten4D(nn.Module):
    """
    4D tensorを1Dにflattenする
    """
    def forward(self, x):
        return x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))


class Flatten3D(nn.Module):
    """
    3D tensorを1Dにflattenする
    """
    def forward(self, x):
        return x.view(-1, x.size(1) * x.size(2) * x.size(3))


class SmallAlexNet(nn.Module):
    def __init__(self, hidden=128, num_classes=1, in_channels=1):
        super(SmallAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(2, 4, 4)), # (L/2, H/4, W/4)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(2, 4, 4)), # (L/4, H/16, W/16)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2)), # (L/8, H/32, W/32)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
        )
        self.flatten = Flatten4D()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 3 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SmallAlexNet2(nn.Module):
    """
    conv layerの数を3から5に増やす代わりに、strideやkernel sizeを小さくする
    """
    def __init__(self, hidden=128, num_classes=1, in_channels=1):
        super(SmallAlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2)), # (L/2, H/2, W/2)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2)), # (L/4, H/4, W/4)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2)), # (L/8, H/8, W/8)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)), # (L/16, H/16, W/16)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)), # (L/32, H/32, W/32)
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True),
        )
        self.flatten = Flatten4D()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 3 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# 以下、https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.pyより
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    """
    オリジナルソースコードからの変更点
    * __init__内のBottleNeckのinitializeの処理を削除
        * resnet10, 18ではBottleNeckを使わないため
        * それに伴い、zero_init_residualという引数も削除
    * in_channelsを指定できるようにした
        * もとは「3」とハードコーディングされていた
    * 最後に活性化関数としてsigmoidを施せるオプションを追加した


    留意事項
    * groups=1, width_per_group=64となっているがここはBasicBlockをつかっている分には使わない引数である。
    * replace_stride_with_dilationも、BasicBlockを使っている分には使わない(というより、デフォルトの値以外を受け付けない)引数である
    """
    def __init__(self, block, layers, num_classes=1000, in_channels=24, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 ngf=64, norm_layer=None, last_sigmoid=False):
        super(ResNet2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.last_sigmoid = last_sigmoid
        if self.last_sigmoid:
            warnings.warn("モデルの出力にsigmoid関数を施します")
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, ngf, layers[0])
        self.layer2 = self._make_layer(block, ngf * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, ngf * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, ngf * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ngf * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_until_avgpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.last_sigmoid:
            x = F.sigmoid(x)

        return x

    def forward_until_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet10_2D(num_classes, in_channels, ngf, last_sigmoid=False):
    """Constructs a ResNet-10 model.
    """
    model = ResNet2D(BasicBlock2D, [1, 1, 1, 1], num_classes, in_channels, ngf=ngf, last_sigmoid=last_sigmoid)
    return model


def resnet18_2D(num_classes, in_channels, ngf, last_sigmoid=False):
    """Constructs a ResNet-18 model.
    """
    model = ResNet2D(BasicBlock2D, [2, 2, 2, 2], num_classes, in_channels, ngf=ngf, last_sigmoid=last_sigmoid)
    return model



# 以下、2D-CNN
class SmallAlexNet_2D(nn.Module):
    def __init__(self, hidden=128, num_classes=1, in_channels=24):
        super(SmallAlexNet_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(7, 7), padding=(3, 3), stride=(4, 4)), # (H/4, W/4)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(7, 7), padding=(3, 3), stride=(4, 4)), # (H/16, W/16)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), # (H/32, W/32)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SmallAlexNet2_2D(nn.Module):
    """
    conv layerの数を3から5に増やす代わりに、strideやkernel sizeを小さくする
    """
    def __init__(self, hidden=128, num_classes=1, in_channels=24):
        super(SmallAlexNet2_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), # (H/2, W/2)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), # (H/4, W/4)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), # (H/8, W/8)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), # (H/16, W/16)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), # (H/32, W/32)
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

