import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# from ops.dcn.deform_conv import ModulatedDeformConv
from deform_conv import ModulatedDeformConv
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.utils import get_logger

class TDAM(nn.Module):
    def __init__(self, in_nc, nf=64, base_ks=3, deform_ks=3,spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'):
    # def __init__(self, in_nc, nf=64, base_ks=3, deform_ks=3,spynet_pretrained=None):   # if restart the model 
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(TDAM, self).__init__()
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        layers = []
        layers.append(nn.Conv2d(in_channels=15, out_channels=nf, kernel_size=base_ks, padding=base_ks//2, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=base_ks, padding=base_ks//2, bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_offset_extra = nn.Sequential(*layers)

        self.conv_flow = nn.Conv2d(
            24, in_nc*2*self.size_dk, base_ks, padding=base_ks//2
            )
        self.conv_offset = nn.Conv2d(
            nf, in_nc*2*self.size_dk, base_ks, padding=base_ks//2
            )
        self.conv_mask = nn.Conv2d(
            nf, in_nc*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, nf, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )
    
    def check_if_mirror_extended(self, inputs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            inputs (tensor): Input images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if inputs.size(1) % 2 == 0:
            inputs_1, inputs_2 = torch.chunk(inputs, 2, dim=1)
            if torch.norm(inputs_1 - inputs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, inputs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            inputs (tensor): Input images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = inputs.size()
        inputs_1 = inputs[:, :-1, :, :, :].reshape(-1, c, h, w)
        inputs_2 = inputs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(inputs_1, inputs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(inputs_2, inputs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, inputs):

        n, t, c, h, w = inputs.size()
        self.check_if_mirror_extended(inputs)
        flows_forward, flows_backward = self.compute_flow(inputs)

        # backward-time propgation
        warped_outputs = []
        feat_prop = inputs.new_zeros(n, 1, h, w)  # new a feat
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(inputs[:, i, :, :, :], flow.permute(0, 2, 3, 1))
            warped_outputs.append(feat_prop)

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                    feat_prop = flow_warp(inputs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                else:
                    flow = flows_backward[:, -i, :, :, :]
                    feat_prop = flow_warp(inputs[:, -i, :, :, :], flow.permute(0, 2, 3, 1))
            warped_outputs.append(feat_prop)
        
        warped_outputs = warped_outputs[::-1]
        ii = t//2   
        lq_frame = inputs[:, t//2, :, :, :]
        warped_outputs.append(lq_frame)

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        flows = torch.cat([flows_forward,flows_backward],dim=1)
        flow_feature = self.conv_flow(flows.view(n,-1,h,w))
        warped_outputs = torch.stack(warped_outputs, dim=1)
        conv_feature = self.conv_offset_extra(warped_outputs.view(n,-1,h,w))
        # conv_feature = self.conv_offset_extra(warped_outputs)
        off_residual = self.conv_offset(conv_feature)
        off = flow_feature + off_residual
        msk = torch.sigmoid(self.conv_mask(conv_feature))

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs.squeeze(2).contiguous(), off, msk), 
            inplace=True
            )

        return fused_feat

class ResBlock(nn.Module):
    def __init__(self, nf=64, base_ks=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, base_ks, stride, padding)
        self.conv2 = nn.Conv2d(nf, nf, base_ks, stride, padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea):
        temp = fea
        temp = self.conv1(temp)
        temp = self.lrelu(temp)
        temp = self.conv2(temp)
        out = temp + fea
        return out

class PSFM(nn.Module):
    def __init__(self, nf=64, base_ks=3):
        """
        Pyrimidal Spatial Fusion Module
        Args:
            in_nc: num of input channels.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PSFM, self).__init__()
        
        # for stage 0 
        u0 = []
        for i in range(4):
            u0.append(ResBlock())

        self.s0_u0_res = nn.Sequential(*u0)
        self.s0_u1 = nn.Sequential(
            nn.Conv2d(nf*2, nf, base_ks, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

        # for stage 1
        self.d1_conv1 = nn.Conv2d(nf, nf//2, 3,1,1)
        self.d1_str2 = nn.Sequential(
            nn.Conv2d(nf//2, nf//2, base_ks, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        self.d1_Res = ResBlock(nf*2)
        self.u1_deconv = nn.ConvTranspose2d(nf*2,  nf, 4, stride=2, padding=1)
        self.u1_conv = nn.Sequential(
            nn.Conv2d(nf*4, nf*2, base_ks, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

        # for stage2
        self.d2_conv1 = nn.Conv2d(nf*2, nf, 3,1,1)
        self.d2_str2 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        self.d2_Res = ResBlock(nf*4)
        self.u2_deconv = nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1)

        tail_conv_lst = []
        for i in range(5):
            tail_conv_lst.append(ResBlock())
        self.hid_conv = nn.Sequential(*tail_conv_lst)

    def forward(self, inputs):
        s0_fea = self.s0_u0_res(inputs)

        # for stage down1
        d1_fea1 = self.d1_conv1(s0_fea)
        d1_fea_bic = F.interpolate(d1_fea1, scale_factor=0.5, mode='bicubic')
        d1_fea_avg = F.avg_pool2d(d1_fea1, kernel_size=2, stride=2)
        d1_fea_max = F.max_pool2d(d1_fea1, kernel_size=2, stride=2)
        d1_fea_s2 = self.d1_str2(d1_fea1)
        d1_fea = self.d1_Res(torch.cat([d1_fea_bic,d1_fea_avg,d1_fea_max,d1_fea_s2],dim = 1))

        # for stage down2
        d2_fea1 = self.d2_conv1(d1_fea)
        d2_fea_bic = F.interpolate(d2_fea1, scale_factor=0.5, mode='bicubic')
        d2_fea_avg = F.avg_pool2d(d2_fea1, kernel_size=2, stride=2)
        d2_fea_max = F.max_pool2d(d2_fea1, kernel_size=2, stride=2)
        d2_fea_s2 = self.d2_str2(d2_fea1)
        d2_fea = self.d2_Res(torch.cat([d2_fea_bic,d2_fea_avg,d2_fea_max,d2_fea_s2],dim = 1))

        # for stage up2
        up2_fea = self.u2_deconv(d2_fea)

        # for stage up1
        up2_fea = self.u1_conv(torch.cat([up2_fea,d1_fea],dim=1))
        up1 = self.u1_deconv(up2_fea)

        # for stage 0
        out = s0_fea + self.s0_u1(torch.cat([up1,s0_fea],dim=1))

        # residual block for to generate enhanced features
        out = self.hid_conv(out)

        return out

class HDRO(nn.Module):
    """
    Hybrid Dilation Reconstruction Operator
    """
    def __init__(self, nf=64, out_nc=1, base_ks=3, bias=True):
        super(HDRO, self).__init__()

        self.dilation_1 = nn.Conv2d(nf, out_nc, 3, stride=1, padding=1, dilation=1, bias=True)
        self.dilation_2 = nn.Conv2d(nf, out_nc, 3, stride=1, padding=2, dilation=2, bias=True)
        self.dilation_3 = nn.Conv2d(nf, out_nc, 3, stride=1, padding=4, dilation=4, bias=True)
        self.conv = nn.Conv2d(out_nc*3, out_nc, base_ks, padding=(base_ks//2),stride = 1, bias=bias)
    def forward(self, fea):
        fea1 = self.dilation_1(fea)
        fea2 = self.dilation_2(fea)
        fea3 = self.dilation_3(fea)
        out_fea = self.conv(torch.cat([fea1,fea2,fea3],dim=1))

        return out_fea

class ASAM(nn.Module):
    """
    Auxiliary Supervised Attention Module
    """
    def __init__(self, nf=64, out_nc=1, base_ks=3, bias=True):
        super(ASAM, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, base_ks, padding=(base_ks//2),stride = 1, bias=bias)
        self.conv2 = nn.Conv2d(nf, out_nc,  base_ks, padding=(base_ks//2),stride = 1, bias=bias)
        self.conv3 = nn.Conv2d(out_nc, nf,  base_ks, padding=(base_ks//2),stride = 1, bias=bias)
        self.hdro = HDRO(nf,out_nc,base_ks)
        
    def forward(self, fea, lq_img):
        x1 = self.conv1(fea)
        res = self.hdro(x1)
        img = res + lq_img
        x_img_fea = self.conv3(img)
        x2 = torch.sigmoid(x_img_fea)
        x1 = x1*x2
        x1 = x1+x_img_fea
        return x1, img
        
class GSRM(nn.Module):
    def __init__(self, nf=64, nb = 10, out_nc=1, base_ks=3):
        """
        Global Supervised Reconstruction Module
        Args:
            nf: num of channels (filters) of each conv layer.
            nb: num of blocks
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(GSRM, self).__init__()

        init_conv_lst = []
        for i in range(nb):
            init_conv_lst.append(ResBlock(nf))
        self.init_conv = nn.Sequential(*init_conv_lst)

        self.hdro = HDRO(nf,out_nc,base_ks)
        
    def forward(self, fea, lq_img):

        tail_fea = self.init_conv(fea)

        res = self.hdro(tail_fea)
        out_img = res + lq_img
        return out_img


class TSAN(nn.Module):
    """
    in: (B T C H W)
    out: (B C H W)
    """
    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(TSAN, self).__init__()

        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.tdam = TDAM(in_nc=opts_dict['tsan']['in_nc'] * self.input_len,
            nf=opts_dict['tsan']['nf'], 
            base_ks=opts_dict['tsan']['base_ks'],
            deform_ks=opts_dict['tsan']['deform_ks']
            )
        self.psfm = PSFM(
            nf=opts_dict['tsan']['nf'], 
            base_ks=opts_dict['tsan']['base_ks'])
        
        self.asam = ASAM(
            out_nc=opts_dict['tsan']['out_nc'], 
            nf=opts_dict['tsan']['nf'], 
            base_ks=opts_dict['tsan']['base_ks'])

        self.gsrm = GSRM(
            out_nc=opts_dict['tsan']['out_nc'], 
            nf=opts_dict['tsan']['nf'], 
            base_ks=opts_dict['tsan']['base_ks'])


    def forward(self, inputs):
        fea_tdam = self.tdam(inputs)
        fea_psfm = self.psfm(fea_tdam)

        lq_img = inputs[:, self.radius: self.radius+1, ...]  # res: add middle frame
        fea_asam, mid_img  = self.asam(fea_psfm,lq_img.squeeze(2))

        out = self.gsrm(fea_asam,lq_img.squeeze(2))

        return out, mid_img


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmedit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    # root logger name: mmedit
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        #print("pred_img_i", pred_img_i.size())
        # N = 1
        pred_img_i = pred_img_i.squeeze(2)
        #print("pred_img_i", pred_img_i.size())
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        # print('white_level', white_level.size())
        pred_img_i = pred_img_i / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i
