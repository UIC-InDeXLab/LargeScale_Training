
import torch
from torch.autograd import Function
import math
import torch.nn.functional
import torch.nn.functional as F
import time
from functional import approx_linear_forward_xA_b
class approx_linear_func(Function):

    @staticmethod
    def forward(ctx, inputs, weights, bias, lsh_tables, actives, inactives):
        
        start = time.time()
        # store non-tensor objects in ctx
        t0 = time.time()
        ctx.forward_feed = True
        in_features = inputs.size()[-1]
        if lsh_tables == None:
            ctx.active_set = list(range(weights.size()[0]))
            x = torch.nn.functional.linear(inputs, weights, bias)
            ctx.lsh = None
            ctx.save_for_backward(inputs, weights, bias)
            return x

        ctx.lsh = lsh_tables
        arr_input = inputs.detach().numpy().reshape(in_features, 1)
        active_idx = lsh_tables.query(arr_input)
        ctx.active_set = list(active_idx)
        active_weights = torch.zeros_like(weights)
        active_weights[ctx.active_set] = weights[ctx.active_set]
        x = torch.mm(inputs, active_weights.t())
        active_x = x
        ctx.save_for_backward(inputs, active_weights, bias)
        return active_x
        # return torch.nn.functional.linear(inputs, weights, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, bias = ctx.saved_tensors
        #ctx.forward_feed = False
        dx = None
        dw = None
        db =None
        start = time.time()
        active_grads = torch.zeros_like(grad_output)
        active_grads[:,ctx.active_set] = grad_output[:,ctx.active_set]
        #active_weights = torch.zeros_like(weights)
        #active_weights[ctx.active_set,:] = weights[ctx.active_set,:]
        if ctx.needs_input_grad[0]:
            # dE/dx = dE/dy dy/dz dz/dx
            dx = torch.mm(active_grads, weights)

        if ctx.needs_input_grad[1]:
            # dE/dw = dE/dy dy/dz dz/dw
            dw = torch.mm(active_grads.t(), inputs)

        if ctx.needs_input_grad[2]:
            db = torch.sum(active_grads, 0)

        return dx, dw, db, None, None, None


