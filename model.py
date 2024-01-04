import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class act(nn.Module):
    '''
    题目要求，
    隐藏层的非线性函数为：f(x) = 2/(1+e^(-ax)) -1
    输出层的非线性函数为：f(x) = 1/(1+e^(-ax))
    且用同一个参数控制权系数
    '''
    def __init__(self,factor=1.):
        super(act, self).__init__()
        self.a = Parameter(torch.Tensor([factor,]))
        self.stages = ['hidden', 'output']

    def hidden_act(self,x):
        return 2/(1+torch.pow(torch.e,-self.a*x)) -1
    def output_act(self,x):
        return 1/(1+torch.pow(torch.e,-self.a*x))

    def forward(self,x,stage='hidden'):
        assert stage in self.stages
        if stage != self.stages[-1]:
            x = self.hidden_act(x)
        else :
            x = self.output_act(x)
        return x


class TL_Net(nn.Module):
    def __init__(self, in_channel=9, hidden_channel=3, out_channel=1):
        super(TL_Net, self).__init__()
        self.fc1  = nn.Linear(in_channel, hidden_channel)
        self.act  = act()
        self.fc2  = nn.Linear(hidden_channel, out_channel)
        self.class_thresh = Parameter(torch.Tensor([0.5,]))

    def forward(self, x):
        x = x.float().flatten(1)
        x = self.act(self.fc1(x), stage="hidden")
        x = self.act(self.fc2(x), stage="output")
        if self.training:
            return x
        # import pdb;pdb.set_trace()
        else:
            return (x > self.class_thresh).float()
