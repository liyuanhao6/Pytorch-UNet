import torch
from torch.autograd import Function


# 图像分割评价指标 - dice coefficient
# 评估两个样本的相似性, 衡量两个样本的重叠部分
class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        # 保存在后向传播需要的变量值input和target
        self.save_for_backward(input, target)
        # eps防止分母为0
        eps = 0.0001
        # dice coefficient公式
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()

        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        # 将保存的变量赋值给input和target
        input, target = self.saved_variables
        grad_input = grad_target = None

        # 如果forward里的第一个input需要梯度
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        # 如果forward里的第二个input需要梯度
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    # 初始化为0
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # 逐个进行计算
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
