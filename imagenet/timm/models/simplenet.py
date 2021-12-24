"""PyTorch SimpleNet-RepVGG

A PyTorch implementation of:
* RepVGG-SimpleNet


"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, ConvBnAct, DropPath, create_attn, get_norm_act_layer
from .registry import register_model

from .simplenet_impl.simpnet import SimpNet

__all__ = ['SimpNet']  # model_registry will add each entrypoint fn to this


def _cfg_224(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        **kwargs
    }



simplenet_default_dict=dict(
    dataset='imagenet',
    planes = [64, 128, 256, 512],
    num_blocks = [5, 6, 6, 1],
    dropout_lin = 0.0,
    se_block = False,
    additional_branches = [],
    drop_path = 0.0,
)

default_cfgs = {
    'simplenet_repvgg': _cfg_224(url=''),
}

model_cfgs = dict()

def _create_simplenet(variant, default_cfgs_var=None, pretrained=False, **kwargs):
    default_cfgs_var = variant if default_cfgs_var is None else default_cfgs_var
    return build_model_with_cfg(
        SimpNet, variant, pretrained, default_cfg=default_cfgs[default_cfgs_var],
        feature_cfg=dict(flatten_sequential=True), **model_cfgs[variant])

@register_model
def simplenet_repvgg(pretrained=False, **kwargs):
    return SimpNet(**kwargs['model_cfg'])






import os
import sys
from copy import deepcopy
import torch.multiprocessing as mp
import torch.distributed as dist
from collections import OrderedDict


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def setup(rank, world_size, port, backend='nccl'):
    print(f" setup rank = {rank}")
    dist.init_process_group(
        backend, init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size, rank=rank)


def get_groups():
    groups = {
                (0, 1): dist.new_group([0, 1]),
                (0, 2): dist.new_group([0, 2]),
                (1, 2): dist.new_group([1, 2]),
            }

    return groups

def get_broadcast_group():
    group = dist.new_group([0, 1, 2])
    return group

def broadcast_send(x, group):
    dist.broadcast(x, src=0, group=group, async_op=True)

def broadcast_receive(x, group):
    dist.broadcast(x, src=0, group=group, async_op=False)
    return x

def send(x, dst, rank, groups):
    a = [rank, dst]
    a.sort()
    a = tuple(a)
    group = groups[a]
    dist.broadcast(x, src=rank, group=group, async_op=True)

def receive(x, src, rank, groups, async_op=False):
    a = [rank, src]
    a.sort()
    a = tuple(a)
    group = groups[a]
    dist.broadcast(x, src=src, group=group, async_op=async_op)
    return x



def load_state_dict_backbone(checkpoint_path, use_ema=False):
    print(f" checkpoint_path = {checkpoint_path}")
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        # _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        # _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

from .factory import create_model



def run_func_proc(rank, inits, streams, port, in_queue_list, out_queue_list, world_size):
    rank = rank + 1
    #print(f" run_func_proc setup rank = {rank}...")
    setup(rank, world_size=world_size, port=port, backend='nccl')
    #print(f" run_func_proc setup rank = {rank} successfull")
    torch.cuda.set_device(rank)
    #print(f"torch.cuda.set_device rank = {rank} ")

    groups = get_groups()
    broadcas_group = get_broadcast_group()

    streams = streams.to(rank).float().eval()
    inits = inits.to(rank).float().eval()

    if rank==1:
        execute_block = nn.Sequential(inits[0], inits[1], streams)
    elif rank==2:
        execute_block = nn.Sequential(inits[0], inits[1], inits[2], streams)

    in_recv = torch.zeros([1, 1, 1, 1], dtype=torch.float, device=rank)
    sys.stdout.flush()
    queue_use = True

    with torch.no_grad():
        while True:
            if queue_use:
                tensor_size = in_queue_list[rank].get()
                #print(f"got in_queue_list = {rank} ")
                #sys.stdout.flush()
                if tensor_size is None:
                    return
                if tensor_size != in_recv.shape:
                    in_recv = torch.zeros(tensor_size, dtype=torch.float, device=rank)

            x = broadcast_receive(in_recv, broadcas_group)
            #x = receive(in_recv, 0, rank, groups)
            #print(f"got in_recv = {rank} ")

            #for i in range(rank+1):
            #    x = inits[i](x)
            #y = streams(x)

            y = execute_block(x)

            if queue_use:
                out_queue_list[rank].put(y.shape)
            #print(f"out_queue_list[rank] = {rank} ")
            send(y, 0, rank, groups)
            #print(f"send final = {rank} ")
            queue_use = False

            if not queue_use:
                if not in_queue_list[rank].empty():
                    tensor_size = in_queue_list[rank].get()
                    print(f"tensor_size = {tensor_size}")
                    sys.stdout.flush()
                    if tensor_size is None:
                        print(f"Exit run_func_proc(), rank = {rank}")
                        return



class SimpNetMultiGPU(nn.Module):
    def __init__(self, model_name, weights, use_ema, **kwargs):
        super().__init__()

        # Create model
        #print(f" **kwargs = {kwargs}")
        self.model = create_model(model_name, pretrained=False, **kwargs)

        print(f" use_ema = {use_ema}")
        print(f" weights = {weights}")

        state_dict = load_state_dict_backbone(weights, use_ema=use_ema)
        self.model.load_state_dict(state_dict, strict=True)


        # Multi-GPU
        self.world_size = 3
        self.rank = 0
        torch.cuda.set_device(self.rank)

        self.model.to(0).float().eval()

        self.s1_receive = torch.zeros([1, 1, 1, 1], dtype=torch.float, device=0)
        self.s2_receive = torch.zeros([1, 1, 1, 1], dtype=torch.float, device=0)

        self.in_queue_list = [mp.SimpleQueue(), mp.SimpleQueue(), mp.SimpleQueue()]
        self.out_queue_list = [mp.SimpleQueue(), mp.SimpleQueue(), mp.SimpleQueue()]

        free_port = _find_free_port()
        print(f" free_port = {free_port}, setup...")
        sys.stdout.flush()


        print("run process - 0")
        self.p1 = mp.Process(target=run_func_proc, args=(0, deepcopy(self.model.inits), deepcopy(self.model.streams[1]), free_port, self.out_queue_list, self.in_queue_list, self.world_size, ))

        print("run process - 1")
        self.p2 = mp.Process(target=run_func_proc, args=(1, deepcopy(self.model.inits), deepcopy(self.model.streams[2]), free_port, self.out_queue_list, self.in_queue_list, self.world_size, ))

        self.p1.start()
        self.p2.start()

        print(" setup rank 0...")
        setup(rank=0, world_size=self.world_size, port=free_port, backend='nccl')
        torch.cuda.set_device(0)
        self.groups = get_groups()
        self.broadcas_group = get_broadcast_group()
        print(" setup rank 0!")

        self.execute_block = nn.Sequential(self.model.inits[0], self.model.streams[0], self.model.downsamples_2[0])
        self.combine_head = nn.Sequential(self.model.combine, self.model.head)

        self.queue_use = True


    def forward(self, x, out_feat=False):

        #x = x.float()
        #print("Run modules on different GPUs")
        #print(f" x.shape = {x.shape}")

        if self.queue_use:
            self.out_queue_list[1].put(x.shape)
            #send(x, 1, rank=0, groups=self.groups)

            self.out_queue_list[2].put(x.shape)
            #send(x, 2, rank=0, groups=self.groups)

        broadcast_send(x, self.broadcas_group)
        self.last_x = x

        #x0 = self.model.inits[0](x)
        #s0 = self.model.streams[0](x0)
        #d0 = self.model.downsamples_2[0](s0)

        d0 = self.execute_block(x)

        # From Stream-1
        if self.queue_use:
            while self.in_queue_list[1].empty():
                pass
            s1_receive_size = self.in_queue_list[1].get()
            if s1_receive_size != self.s1_receive.shape:
                self.s1_receive = torch.zeros(s1_receive_size, dtype=torch.float, device=0)

        s1 = receive(self.s1_receive, 1, rank=0, groups=self.groups, async_op=False)
        d1 = self.model.downsamples_2[1]((s1, d0))

        # From Stream-2
        if self.queue_use:
            while self.in_queue_list[2].empty():
                pass
            s2_receive_size = self.in_queue_list[2].get()
            if s2_receive_size != self.s2_receive.shape:
                self.s2_receive = torch.zeros(s2_receive_size, dtype=torch.float, device=0)

        s2 = receive(self.s2_receive, 2, rank=0, groups=self.groups, async_op=False)

        #d2 = self.model.combine((s2, d1))
        #out = self.model.head(d2)

        out = self.combine_head((s2, d1))

        #out = out.float()
        self.queue_use = False

        return out

    def __del__(self):
        print("SimpNetMultiGPU.__del__ ...")
        sys.stdout.flush()
        #self.out_queue_list[1].put(None)
        #self.out_queue_list[2].put(None)
        #broadcast_send(self.last_x, self.broadcas_group)
        self.p1.terminate()
        self.p2.terminate()
        dist.destroy_process_group()
        print("SimpNetMultiGPU.__del__!")
        sys.stdout.flush()


@register_model
def simplenet_repvgg_mgpu(pretrained=False, weights="", use_ema=True, **kwargs):
    print(kwargs)
    model_cfg = kwargs.get('model_cfg', None)
    if model_cfg is not None:
        weights = model_cfg.pop('weights', weights)
        use_ema = model_cfg.pop('use_ema', use_ema)
    return SimpNetMultiGPU(model_name="simplenet_repvgg", weights=weights, use_ema=use_ema, **kwargs)
