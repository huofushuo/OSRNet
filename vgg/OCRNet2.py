import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Softmax

from vgg.vgg import B2_VGG

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.out = BasicConv2d(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        feat1 = _upsample_like(self.conv1(self.pool(x, 1)), x)
        feat2 = _upsample_like(self.conv2(self.pool(x, 2)), x)
        feat3 = _upsample_like(self.conv3(self.pool(x, 3)), x)
        feat4 = _upsample_like(self.conv4(self.pool(x, 6)), x)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class PAM_Module(nn.Module):
    """ Position attention module"""
    #paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1).contiguous()
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).contiguous()
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=9, dilation=9)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)*x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = max_out
        y = self.conv1(y)
        return self.sigmoid(y)*x


class decoder(nn.Module):
    def __init__(self, channel):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5_rgb = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5_rgb = nn.Sequential(
            nn.Conv2d(channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.decoder4_rgb = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4_rgb = nn.Sequential(
            nn.Conv2d(channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.decoder3_rgb = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3_rgb = nn.Sequential(
            nn.Conv2d(channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.decoder2_rgb = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2_rgb = nn.Sequential(
            nn.Conv2d(channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.decoder1_rgb = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            # TransBasicConv2d(64, 64, kernel_size=2, stride=2,
            #                  padding=0, dilation=1, bias=False)
        )
        self.S1_rgb = nn.Sequential(
            nn.Conv2d(channel, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.ppm_r5 = PyramidPooling(512, channel)
        # self.ppm_r4 = PyramidPooling(512, channel)
        self.aspp_r5 = ASPP(64, 32)
        self.aspp_r4 = ASPP(64, 32)
        self.ch_r5 = BasicConv2d(512, 64, 3, padding=1)
        self.ch_r4 = BasicConv2d(512, 32, 3, padding=1)
        self.ch_r3 = BasicConv2d(256, 32, 3, padding=1)
        self.ch_r2 = BasicConv2d(128, 32, 3, padding=1)
        self.ch_r1 = BasicConv2d(64, 32, 3, padding=1)

        self.sa5 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.ca5 = ChannelAttention(64)
        self.ca4 = ChannelAttention(64)

        # self.uf5_rgb = ASPP(512, 512)
        # self.uf4_rgb = ASPP(1024, 4*channel)
        self.uf3_rgb = RSU(2*channel, 16, channel)
        self.uf2_rgb = RSU(2*channel, 16, channel)
        self.uf1_rgb = RSU(2*channel, 16, channel)


        # for m in self.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         m.weight.data.normal_(mean=0, std=0.01)
        #         m.bias.data.fill_(0.0)

    def forward(self, x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb):

        x1_rgb = self.ch_r1(x1_rgb)
        x2_rgb = self.ch_r2(x2_rgb)
        x3_rgb = self.ch_r3(x3_rgb)
        x4_rgb = self.ch_r4(x4_rgb)
        x5_rgb = self.ch_r5(x5_rgb)

        x5_rgb = self.ca5(x5_rgb)
        x5_rgb = self.sa5(x5_rgb)

        x5_rgb = self.aspp_r5(x5_rgb)
        x5_up_rgb = self.decoder5_rgb(x5_rgb)
        # print('x5_up size {} '.format(x5_up.shape))
        s5_rgb = self.S5_rgb(x5_up_rgb)

        # x4_rgb = self.pam4(torch.cat((x4_rgb, x5_up_rgb), 1)+torch.cat((x4_rgb, x5_up_rgb), 1)*s5_rgb) + self.cam4(torch.cat((x4_rgb, x5_up_rgb), 1)+torch.cat((x4_rgb, x5_up_rgb), 1)*s5_rgb)
        x4_rgb = torch.cat((x4_rgb, x5_up_rgb), 1)
        x4_rgb = self.ca4(x4_rgb)
        x4_rgb = self.sa4(x4_rgb)

        x4_rgb = self.aspp_r4(x4_rgb)
        x4_up_rgb = self.decoder4_rgb(x4_rgb)
        # print('x4_up size {} '.format(x4_up.shape))
        s4_rgb = self.S4_rgb(x4_up_rgb)

        s4_rgb3 = _upsample_like(s4_rgb, s4_rgb)
        x3_rgb = self.uf3_rgb(torch.cat((x3_rgb, x4_up_rgb), 1)+torch.cat((x3_rgb, x4_up_rgb), 1)*s4_rgb3)
        x3_up_rgb = self.decoder3_rgb(x3_rgb)
        # print('x3_up size {} '.format(x3_up.shape))
        s3_rgb = self.S3_rgb(x3_up_rgb)

        s4_rgb2 = _upsample_like(s4_rgb, s3_rgb)
        x2_rgb = self.uf2_rgb(torch.cat((x2_rgb, x3_up_rgb), 1)+torch.cat((x2_rgb, x3_up_rgb), 1)*s4_rgb2)
        x2_up_rgb = self.decoder2_rgb(x2_rgb)
        # print('x2_up size {} '.format(x2_up.shape))
        s2_rgb = self.S2_rgb(x2_up_rgb)

        s4_rgb1 = _upsample_like(s4_rgb, s2_rgb)
        x1_rgb = self.uf1_rgb(torch.cat((x1_rgb, x2_up_rgb), 1)+torch.cat((x1_rgb, x2_up_rgb), 1)*s4_rgb1)
        x1_up_rgb = self.decoder1_rgb(x1_rgb)
        # print('x1_up size {} '.format(x1_up.shape))
        s1_rgb = self.S1_rgb(x1_up_rgb)
        # print('s1 size {} '.format(s1.shape))


        return s1_rgb, s2_rgb, s3_rgb, s4_rgb, s5_rgb

class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class RSU(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin


class SpatialAttention_sig(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_sig, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return x



class OCRNet(nn.Module):
    def __init__(self, channel=32):
        super(OCRNet, self).__init__()
        #Backbone model

        self.vgg = B2_VGG('single')
        # self.vgg_dep = B2_VGG('dep')

        # self.agg2_rgbd = aggregation(channel)
        self.decoder_rgbd = decoder(32)

        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x_rgb):
        x1_rgb = self.vgg.conv1(x_rgb)
        x2_rgb = self.vgg.conv2(x1_rgb)
        x3_rgb = self.vgg.conv3(x2_rgb)
        x4_rgb = self.vgg.conv4(x3_rgb)
        x5_rgb = self.vgg.conv5(x4_rgb)

        s1, s2, s3, s4, s5 = self.decoder_rgbd(x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb)

        s3 = self.upsample2(s3)
        s4 = self.upsample4(s4)
        s5 = self.upsample8(s5)

        return s1, s2, s3, s4, s5
