import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time as tm
from timm.models.layers.drop import drop_path, drop_block_fast_2d

def round(f):
    return math.ceil(f / 2.) * 2

def num_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def trace_net(net, inp, save_pth="traced_model.pt"):
    traced_script_module = torch.jit.trace(
        net, inp, strict=True)
    traced_script_module.save(save_pth)

class MultiBatchNorm2d(nn.Module):
    def __init__(self, n1, n2, num_branch=None):
        super().__init__()
        self.num_branch = num_branch
        if num_branch is None:
            self.b1 = nn.BatchNorm2d(n1)
            self.b2 = nn.BatchNorm2d(n2)
        else:
            assert n2 is None
            self.b = nn.ModuleList(
                [nn.BatchNorm2d(n1) for _ in range(num_branch)])

    def forward(self, x):
        if self.num_branch is None:
            x1, x2 = x
            x1 = self.b1(x1)
            x2 = self.b2(x2)
            out = (x1, x2)
        else:
            out = []
            for _x, _b in zip(x, self.b):
                out.append(_b(_x))

        return out


class Concat2d(nn.Module):
    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle

    def forward(self, x):
        if self.shuffle:
            b, _, h, w = x[0].shape
            x = [_x.unsqueeze(1) for _x in x]
            out = torch.cat(x, 1)
            out = out.transpose(1, 2)
            out = torch.reshape(out, (b, -1, h, w))
        else:
            out = torch.cat(x, 1)
        return out


class ReLU2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x1, x2 = x
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        return (x1, x2)


"""
source:
https://github.com/moskomule/senet.pytorch/blob/23839e07525f9f5d39982140fccc8b925fe4dee9/senet/se_module.py#L4-L19

"""
class SELayer(nn.Module):
    def __init__(self, channel, out_channel=None, reduction=16, version=1):
        super(SELayer, self).__init__()
        self.version = version
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if out_channel is None:
            out_channel = channel
        self.channel = channel
        self.out_channel = out_channel
        if version == 1:
            self.fc = nn.Sequential(
                nn.Linear(channel, out_channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_channel // reduction, out_channel, bias=False),
                nn.Sigmoid()
            )
        elif version == 2:
            reduction = reduction // 2
            self.fc = nn.Sequential(
                nn.AvgPool1d(reduction),
                nn.Linear(channel // reduction, out_channel, bias=False),
                nn.Sigmoid()
            )
        else:
            assert False, version

    def forward(self, x, x2=None):
        if x2 is None:
            assert self.out_channel == self.channel
            x2 = x
        b, c, _, _ = x.size()
        b, c2, _, _ = x2.size()
        assert c == self.channel
        assert c2 == self.out_channel

        y = self.avg_pool(x).view(b, c)
        if self.version == 2:
            y = y.view(b, 1, c)
        y = self.fc(y).view(b, c2, 1, 1)
        return x2 * y.expand_as(x2)


class SE1(nn.Module):
    # Squeeze-and-excitation block in https://arxiv.org/abs/1709.01507
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c_in, c_out, n=1, shortcut=True,  g=1, e=0.5, ver=1):
        super(SE1, self).__init__()
        self.ver = ver
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cvsig = ConvSig(c_in, c_out, 1, 1, g=g)

    def forward(self, x):
        x = self.cvsig(self.avg_pool(x))
        if self.ver == 2:
            x = 2 * x
        return x

def update_bn(loader, model, total_imgs=1000, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    # print(model)

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    if device != None:
        model.to(device)

    num_images_total = 0

    for i, data in tqdm.tqdm(enumerate(loader), total = total_imgs):
        if i*loader.batch_size >= total_imgs:
            break
        img = data['img']
        img = img.to(device)
        model(img)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

    print("update_bn is completed successfully, total_imgs = ", total_imgs)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ConvSig(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvSqu(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSqu, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# source: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(
            y.squeeze(-1).transpose(-1, -2)
            ).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock_train(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, drop_path=0.0, activation=nn.ReLU(),
                 padding_mode='zeros', deploy=False, avg_pool=False):
        super(RepVGGBlock_train, self).__init__()
        self.avg_pool = avg_pool
        self.deploy = deploy
        self.drop_path = drop_path
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 0x0, 1x1, 3x3
        self.largest_kernel_size = kernel_size
        self.kernels = (kernel_size // 2) + 2

        self.conv_out_channels = out_channels * self.kernels

        #self.kernels = 3
        #self.largest_kernel_size = (self.kernels-2)*2 + 1

        #assert kernel_size == 3

        padding=dilation*(kernel_size-1)//2

        self.nonlinearity = activation

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.conv_out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)

        self.bn = nn.BatchNorm2d(self.conv_out_channels)

        #print(f" self.conv.weight.shape = {self.conv.weight.shape}")

        self.prepare_mult_add()
        self.change_weights()


    def forward(self, x):
        if self.training:
            self.change_weights()

        conv_out = self.conv(x)

        if not self.bn is None:
            conv_out = self.bn(conv_out)

        out_list = []
        for i in range(self.kernels):
            start = i*self.out_channels
            end = (i+1)*self.out_channels
            drop_path_in = conv_out[:,start:end,:,:]
            if i > 0:
                drop_path_in = drop_path(drop_path_in, self.drop_path, self.training)
            out_list.append(drop_path_in)

        out = torch.sum(torch.stack(out_list), dim=0)

        """
        out = conv_out[:,0:self.out_channels,:,:]
        for i in range(self.kernels):
            start = i*self.out_channels
            end = (i+1)*self.out_channels
            if i > 0:
                drop_path_in = conv_out[:,start:end,:,:]
                drop_path_in = drop_path(drop_path_in, self.drop_path, self.training)
                out.add_(drop_path_in)
        """

        out = self.nonlinearity(out)

        return out


    def prepare_mult_add(self):
        with torch.no_grad():
            n = self.out_channels
            c = self.in_channels // self.groups
            h = self.largest_kernel_size
            w = self.largest_kernel_size
            # each_conv_shape = [n, c, h, w]

            dtype =  self.conv.weight.dtype
            device =  self.conv.weight.device
            shape =  self.conv.weight.shape

            mult_list = []
            add_list = []
            for i in range(self.kernels):
                ksize = (i-1)*2 + 1
                pad = (self.largest_kernel_size - ksize) // 2
                if i==0:
                    mult = torch.zeros([n,c,h,w], dtype=dtype, requires_grad=False)
                    if self.avg_pool:
                        pad = self.largest_kernel_size - self.stride
                        add = torch.ones([n,c,self.stride,self.stride], dtype=dtype, requires_grad=False) / (self.stride*self.stride)
                        add = torch.nn.functional.pad(add, [pad//2, pad - pad//2, pad//2, pad - pad//2])
                    else:
                        add = torch.ones([n,c,1,1], dtype=dtype, requires_grad=False)
                        add = torch.nn.functional.pad(add, [w//2, w//2, h//2, h//2])
                elif i>=1:
                    mult = torch.ones([n,c,ksize,ksize], dtype=dtype, requires_grad=False)
                    mult = torch.nn.functional.pad(mult, [pad, pad, pad, pad])
                    add = torch.zeros([n,c,h,w], dtype=dtype, requires_grad=False)

                #print(f"\n i = {i}, mult = {mult} \n end of mult, i = {i}. \n")
                #tm.sleep(5)
                #print(f"\n i = {i}, add = {add} \n end of add, i = {i}. \n")
                #tm.sleep(5)

                mult_list.append(mult)
                add_list.append(add)

            self.mult_weights =  torch.cat(mult_list, dim=0)
            self.add_weights = torch.cat(add_list, dim=0)
            self.mult_weights.requires_grad=False
            self.add_weights.requires_grad=False


    def change_weights(self):

        #print(f" self.conv.weight.shape = {self.conv.weight.shape}, mult_weights.shape={mult_weights.shape}, add_weights.shape={add_weights.shape}")

        if self.mult_weights.device != self.conv.weight.device or self.add_weights.device != self.conv.weight.device:
            #print(f"\n move self.mult_weights to {self.conv.weight.device}")
            self.mult_weights = self.mult_weights.to(self.conv.weight.device)
            self.add_weights = self.add_weights.to(self.conv.weight.device)
            #print(f" moved self.conv.weight.device = {self.conv.weight.device}, self.mult_weights.device={self.mult_weights.device}, self.add_weights.device={self.add_weights.device}")


        #print(f" self.conv.weight.device = {self.conv.weight.device}, self.mult_weights.device={self.mult_weights.device}, self.add_weights.device={self.add_weights.device}")
        #print("change_weights()")

        with torch.no_grad():
            self.conv.weight.data.mul(self.mult_weights).add(self.add_weights)
            #self.conv.weight.data = self.conv.weight.data * self.mult_weights + self.add_weights


        #self.conv.weight.requires_grad=True

        #print(f"\n self.conv.weight[0,:,:,:] = \n {self.conv.weight[0,:,:,:]} \n end of self.conv.weight. \n")
        #tm.sleep(5)

    def fuse_conv_bn(self):
        """
        # n,c,h,w - conv
        # n - bn (scale, bias, mean, var)

        if type(self.bn) is nn.Identity or type(self.bn) is None:
            return

        self.conv.weight
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        """
        std = (self.bn.running_var + self.bn.eps).sqrt()
        bias = self.bn.bias - self.bn.running_mean * self.bn.weight / std

        t = (self.bn.weight / std).reshape(-1, 1, 1, 1)
        weights = self.conv.weight * t

        self.bn = nn.Identity()
        self.conv = nn.Conv2d(in_channels = self.conv.in_channels,
                              out_channels = self.conv.out_channels,
                              kernel_size = self.conv.kernel_size,
                              stride=self.conv.stride,
                              padding = self.conv.padding,
                              dilation = self.conv.dilation,
                              groups = self.conv.groups,
                              bias = True,
                              padding_mode = self.conv.padding_mode)

        self.conv.weight = torch.nn.Parameter(weights)
        self.conv.bias = torch.nn.Parameter(bias)




class RepVGGBlock_train_shared_bn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, drop_path=0.0, activation=nn.ReLU(),
                 padding_mode='zeros', deploy=False, avg_pool=False):
        super(RepVGGBlock_train_shared_bn, self).__init__()
        self.avg_pool = avg_pool
        self.deploy = deploy
        self.drop_path = drop_path
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 0x0, 1x1, 3x3
        self.largest_kernel_size = kernel_size
        self.kernels = (kernel_size // 2) + 2

        self.conv_in_channels = in_channels * self.kernels

        #self.kernels = 3
        #self.largest_kernel_size = (self.kernels-2)*2 + 1

        #assert kernel_size == 3

        padding=dilation*(kernel_size-1)//2

        self.nonlinearity = activation

        self.conv = nn.Conv2d(in_channels=self.conv_in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)

        self.bn = nn.BatchNorm2d(out_channels)

        #print(f" self.conv.weight.shape = {self.conv.weight.shape}")

        self.prepare_mult_add()
        self.change_weights()


    def forward(self, x):
        if self.training:
            self.change_weights()

        input_list = []
        for i in range(self.kernels):
            drop_path_in = x
            if i > 0:
                drop_path_in = drop_path(drop_path_in, self.drop_path, self.training)
            input_list.append(drop_path_in)
        input = torch.cat(input_list, dim=1)

        conv_out = self.conv(input)

        if not self.bn is None:
            conv_out = self.bn(conv_out)

        out = self.nonlinearity(conv_out)

        return out


    def prepare_mult_add(self):
        with torch.no_grad():
            n = self.out_channels
            c = self.in_channels // self.groups
            h = self.largest_kernel_size
            w = self.largest_kernel_size
            # each_conv_shape = [n, c, h, w]

            dtype =  self.conv.weight.dtype
            device =  self.conv.weight.device
            shape =  self.conv.weight.shape

            mult_list = []
            add_list = []
            for i in range(self.kernels):
                ksize = (i-1)*2 + 1
                pad = (self.largest_kernel_size - ksize) // 2
                if i==0:
                    mult = torch.zeros([n,c,h,w], dtype=dtype, requires_grad=False)
                    if self.avg_pool:
                        pad = self.largest_kernel_size - self.stride
                        add = torch.ones([n,c,self.stride,self.stride], dtype=dtype, requires_grad=False) / (self.stride*self.stride)
                        add = torch.nn.functional.pad(add, [pad//2, pad - pad//2, pad//2, pad - pad//2])
                    else:
                        add = torch.ones([n,c,1,1], dtype=dtype, requires_grad=False)
                        add = torch.nn.functional.pad(add, [w//2, w//2, h//2, h//2])
                elif i>=1:
                    mult = torch.ones([n,c,ksize,ksize], dtype=dtype, requires_grad=False)
                    mult = torch.nn.functional.pad(mult, [pad, pad, pad, pad])
                    add = torch.zeros([n,c,h,w], dtype=dtype, requires_grad=False)

                #print(f"\n i = {i}, mult = {mult} \n end of mult, i = {i}. \n")
                #tm.sleep(5)
                #print(f"\n i = {i}, add = {add} \n end of add, i = {i}. \n")
                #tm.sleep(5)

                mult_list.append(mult)
                add_list.append(add)

            self.mult_weights =  torch.cat(mult_list, dim=1)
            self.add_weights = torch.cat(add_list, dim=1)
            self.mult_weights.requires_grad=False
            self.add_weights.requires_grad=False


    def change_weights(self):

        #print(f" self.conv.weight.shape = {self.conv.weight.shape}, mult_weights.shape={mult_weights.shape}, add_weights.shape={add_weights.shape}")

        if self.mult_weights.device != self.conv.weight.device or self.add_weights.device != self.conv.weight.device:
            #print(f"\n move self.mult_weights to {self.conv.weight.device}")
            self.mult_weights = self.mult_weights.to(self.conv.weight.device)
            self.add_weights = self.add_weights.to(self.conv.weight.device)
            #print(f" moved self.conv.weight.device = {self.conv.weight.device}, self.mult_weights.device={self.mult_weights.device}, self.add_weights.device={self.add_weights.device}")


        #print(f" self.conv.weight.device = {self.conv.weight.device}, self.mult_weights.device={self.mult_weights.device}, self.add_weights.device={self.add_weights.device}")
        #print("change_weights()")

        with torch.no_grad():
            self.conv.weight.data.mul(self.mult_weights).add(self.add_weights)
            #self.conv.weight.data = self.conv.weight.data * self.mult_weights + self.add_weights


        #self.conv.weight.requires_grad=True

        #print(f"\n self.conv.weight[0,:,:,:] = \n {self.conv.weight[0,:,:,:]} \n end of self.conv.weight. \n")
        #tm.sleep(5)

    def fuse_conv_bn(self):
        """
        # n,c,h,w - conv
        # n - bn (scale, bias, mean, var)

        if type(self.bn) is nn.Identity or type(self.bn) is None:
            return

        self.conv.weight
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        """
        std = (self.bn.running_var + self.bn.eps).sqrt()
        bias = self.bn.bias - self.bn.running_mean * self.bn.weight / std

        t = (self.bn.weight / std).reshape(-1, 1, 1, 1)
        weights = self.conv.weight * t

        self.bn = nn.Identity()
        self.conv = nn.Conv2d(in_channels = self.conv.in_channels,
                              out_channels = self.conv.out_channels,
                              kernel_size = self.conv.kernel_size,
                              stride=self.conv.stride,
                              padding = self.conv.padding,
                              dilation = self.conv.dilation,
                              groups = self.conv.groups,
                              bias = True,
                              padding_mode = self.conv.padding_mode)

        self.conv.weight = torch.nn.Parameter(weights)
        self.conv.bias = torch.nn.Parameter(bias)




def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
            print("FUSED ", previous_name, name)
            #torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, input=torch.zeros(1, 3, 224, 224), verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        print("try thop")
        flops = profile((model), inputs=(input,), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops)  # 224x224 FLOPS
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
