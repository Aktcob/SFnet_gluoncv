import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, HybridBlock
from gluoncv.model_zoo.resnetv1b import resnet18_v1b
# convlution // batch normalization // Relu
class CBR(HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad, 
                                    norm_layer=nn.BatchNorm, norm_kwargs=None, 
                                    is_bn=True, is_relu=True, is_bias=False):
        super(CBR, self).__init__()
        self.is_bn = is_bn
        self.is_relu = is_relu
        self.conv = nn.Conv2D(in_channels=in_channels, channels=out_channels, 
                kernel_size=kernel_size, strides=stride, padding=pad, use_bias=is_bias)
        if self.is_bn:
            self.bn = norm_layer(in_channels=out_channels, **({} if norm_kwargs is None else norm_kwargs))
        if self.is_relu:
            # self.relu = nn.Activation('relu')
            self.relu = nn.LeakyReLU(alpha=0.1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_relu:
            x = self.relu(x)
        return x

class PSP(HybridBlock):
    def __init__(self, in_channels=512, out_channels=128, height=60, width=60):
        super(PSP, self).__init__()
        # out_channels = int(in_channels/4)
        self._up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1 = CBR(in_channels, out_channels, 1, 1, 0)
            self.conv2 = CBR(in_channels, out_channels, 1, 1, 0)
            self.conv3 = CBR(in_channels, out_channels, 1, 1, 0)
            self.conv4 = CBR(in_channels, out_channels, 1, 1, 0)
            self.block = nn.HybridSequential()
            self.block.add(CBR(4*out_channels+in_channels, out_channels, 1, 1, 0))
            self.block.add(nn.Dropout(0.1))

    def pool(self, F, x, size):
        return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)

    def upsample(self, F, x):
        return F.contrib.BilinearResize2D(x, **self._up_kwargs)

    def hybrid_forward(self, F, x):
        feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)))
        feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)))
        feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)))
        feat4 = self.upsample(F, self.conv4(self.pool(F, x, 6)))
        return self.block(F.concat(x, feat1, feat2, feat3, feat4, dim=1))


class FAM(HybridBlock):
    def __init__(self, in_channels=128, out_channels=128, height=320, width=320):
        super(FAM, self).__init__()
        self.low_conv = CBR(in_channels, out_channels, 1, 1, 0, is_bn=False, is_relu=False)
        self.high_conv = CBR(in_channels, out_channels, 1, 1, 0, is_bn=False, is_relu=False)
        self.flow_conv = CBR(out_channels*2, 2, 3, 1, 1, is_bn=False, is_relu=False)
        # mesh grid
        grid_x = np.arange(height)
        grid_y = np.arange(width)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # stack to (n, n, 2)
        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        # expand dims to (1, n, n, 2) so it's easier for broadcasting
        offsets = np.array(np.expand_dims(offsets, axis=0)).astype(np.float32)
        self.offsets = self.params.get_constant('offsets', offsets)
        self.height = height
        self.width = width
        self._up_kwargs = {'height': height, 'width': width}

    def hybrid_forward(self, F, x_low, x_high, offsets):
        B, C, H, W = F.shape_array(x_low)
        x_high_ori = x_high
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x_high = F.contrib.BilinearResize2D(x_high, **self._up_kwargs)
        fusion_layer = F.concat(x_low, x_high, dim=1)
        flow = self.flow_conv(fusion_layer).transpose(axes=(0, 2, 3, 1)) # B * H * W * 2
        offsets = F.slice_like(offsets, flow * 0, axes=(1, 2))
        offsets = F.stop_gradient(offsets)
        flow = flow + offsets
        vgrid_x = 2.0 * flow.slice_axis(axis=-1, begin=0, end=1) / max(self.height-1, 1) - 1.0
        vgrid_y = 2.0 * flow.slice_axis(axis=-1, begin=1, end=None) / max(self.width-1, 1) - 1.0
        vgrid = F.concat(vgrid_x, vgrid_y, dim=-1).transpose(axes=(0, 3, 1, 2)) 
        output = F.BilinearSampler(x_high_ori, vgrid)
        return output

class AlignHead(HybridBlock):
    def __init__(self, in_channels=256, fpn_channels=128, height=320, width=320):
        super(AlignHead, self).__init__()
        fpn_in_channels = [in_channels//4, in_channels//2, in_channels]
        heights = [height//16, height//8, height//4]
        widths = [width//16, width//8, width//4]
        self._up_kwargs = {'height': height, 'width': width}
        self.fpn_in_stages = nn.HybridSequential()
        for fpn_in in fpn_in_channels[::-1]:
            self.fpn_in_stages.add(CBR(fpn_in, fpn_channels, 1, 1, 0))
        self.fpn_out_stages = nn.HybridSequential()
        self.fpn_out_aligh_stages = nn.HybridSequential()
        for i in range(len(fpn_in_channels)):
            self.fpn_out_stages.add(CBR(fpn_channels, fpn_channels, 3, 1, 1))
            self.fpn_out_aligh_stages.add(FAM(fpn_channels, fpn_channels//2, heights[i], widths[i]))

    def hybrid_forward(self, F, top_layer, c3, c2, c1):
        c3 = self.fpn_in_stages[0](c3)
        c3_ = self.fpn_out_aligh_stages[0](c3, top_layer)
        c3_ = c3 + c3_
        c3_feat = F.contrib.BilinearResize2D(c3_, **self._up_kwargs)
        c2 = self.fpn_in_stages[1](c2)
        c2_ = self.fpn_out_aligh_stages[1](c2, c3_)
        c2_ = c2 + c2_
        c2_feat = F.contrib.BilinearResize2D(c2_, **self._up_kwargs)
        c1 = self.fpn_in_stages[2](c1)
        c1_ = self.fpn_out_aligh_stages[2](c1, c2_)
        c1_ = c1 + c1_
        c1_feat = F.contrib.BilinearResize2D(c1_, **self._up_kwargs)
        top_layer_feat = F.contrib.BilinearResize2D(top_layer, **self._up_kwargs)
        return F.concat(top_layer_feat, c3_feat, c2_feat, c1_feat, dim=1)

class SFnet(HybridBlock):
    def __init__(self, nclass=19, aux=True, backbone='resnet50', height=320, width=320,
                 base_size=520, crop_size=480, pretrained_base=True, **kwargs):
        super(SFnet, self).__init__()
        self.height = height
        self.width = width
        self._up_kwargs = {'height': height, 'width': width}
        pretrained = resnet18_v1b()
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

        self.PPM = PSP(512, 128, self.height//32, self.width//32)
        self.head = AlignHead()
        self.conv_last = nn.HybridSequential()
        self.conv_last.add(CBR(4*128, 128, 3, 1, 1))
        self.conv_last.add(CBR(128, nclass, 1, 1, 0, is_relu=False, is_bn=False, is_bias=True))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        top_layer = self.PPM(c4)
        fusion_layer = self.head(top_layer, c3, c2, c1)
        return self.conv_last(fusion_layer)

if __name__ == '__main__':
    net = SFnet()
    net.initialize()
    img1 = nd.ones([2,3,320,320])
    img2 = nd.ones([1,3,10,10])
    print (net(img1).shape)
    # print (net)





