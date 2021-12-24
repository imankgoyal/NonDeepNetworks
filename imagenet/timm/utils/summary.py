""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
import subprocess
import sys
from collections import OrderedDict
import tensorboardX
import torch

def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


class TensorboardManager:
    def __init__(self, path):
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            self.writer.add_scalar('%s_%s' % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()

# source:
# https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in GB.
    """
    if sys.platform == "win32":
        result = subprocess.check_output(
            [
                'C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
    else:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [(int(x) / 1024.) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    gpu_memory_map = {(f'GPU {x}'): f'{y:.2} GB'
                      for x, y in gpu_memory_map.items()}
    return gpu_memory_map


def show_cuda_devices():
    c = 1024 ** 2  # bytes to MB
    ng = torch.cuda.device_count()
    x = [torch.cuda.get_device_properties(i) for i in range(ng)]
    s = 'Using CUDA '
    for i in range(0, ng):
        if i == 1:
            s = ' ' * len(s)
        print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (s, i, x[i].name, x[i].total_memory / c))
