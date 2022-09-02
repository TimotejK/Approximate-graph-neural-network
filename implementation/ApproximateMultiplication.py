import torch
from torch.autograd import Function, gradcheck
import torch.nn.functional as F

torch.ops.load_library("build/libcustom_opperations.so")

class Linear(Function):
    @staticmethod
    def forward(ctx, a, b, bias, w):
        ctx.save_for_backward(a, b, bias)
        a = a * 256
        b = b * 256
        if bias is not None:
            result = torch.ops.my_ops.mat_mult(a, b.t(), w) + bias * 256*256
        else:
            result = torch.ops.my_ops.mat_mult(a, b.t(), w)
        result = result / (256*256)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        a, b, bias = ctx.saved_tensors
        return grad_output @ b, (a.t() @ grad_output).t(), grad_output if bias is not None else None, None


if __name__ == '__main__':
    a = torch.tensor([[1,2], [3,4]], dtype=torch.float, requires_grad=True)
    b = torch.tensor([[5,6], [7,8]], dtype=torch.float, requires_grad=True)
    print(Linear.apply(a, b, None, 10))
    print(F.linear(a, b, None))
    print(a.grad, b.grad)