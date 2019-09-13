from collections import namedtuple
from string import Template

import cupy, torch
import cupy as cp
import torch as t
from torch.autograd import Function

from model.roi.roi_cupy import kernel_backward, kernel_forward

Stream = namedtuple('Stream', ['ptr'])


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


forward_fn = load_kernel('roi_forward', kernel_forward)
backward_fn = load_kernel('roi_backward', kernel_backward)

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


class RoI(Function):

    @staticmethod
    def forward(ctx, x, rois, outh, outw, spatial_scale):
        # NOTE: MAKE SURE input is contiguous too
        x = x.contiguous()
        rois = rois.contiguous()
        ctx.in_size = B, C, H, W = x.size()
        ctx.N = N = rois.size(0)
        ctx.outh = outh
        ctx.outw = outw
        ctx.spatial_scale = spatial_scale
        output = t.zeros(N, C, outh, outw).cuda()
        ctx.argmax_data = t.zeros(N, C, outh, outw).int().cuda()
        ctx.rois = rois
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                ctx.argmax_data.data_ptr(),
                ctx.spatial_scale, C, H, W,
                ctx.outh, ctx.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        forward_fn(args=args,
                   block=(CUDA_NUM_THREADS, 1, 1),
                   grid=(GET_BLOCKS(output.numel()), 1, 1),
                   stream=stream)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        B, C, H, W = ctx.in_size
        grad_input = t.zeros(ctx.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                ctx.argmax_data.data_ptr(),
                ctx.rois.data_ptr(),
                grad_input.data_ptr(),
                ctx.N, ctx.spatial_scale, C, H, W, ctx.outh, ctx.outw,
                grad_input.numel()]
        backward_fn(args=args,
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                    stream=stream
                    )
        return grad_input, None, None, None, None


class RoIPooling2D(t.nn.Module):

    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

    def forward(self, x, rois):
        return RoI.apply(x, rois, self.outh, self.outw, self.spatial_scale)
