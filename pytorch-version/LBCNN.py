import torch
import numpy as np
from torch.nn import Module, Conv2d, Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
#from torch.nn.parameter import Parameter

class LBCNN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,gbsparsity = 0.5):
        super(LBCNN, self).__init__()
        self.nInputPlane = in_channels
        self.nOutputPlane = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = groups
        self.kW = kernel_size
        self.spa = gbsparsity
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        ## init weight
        numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kW
        #print('self.nInputPlane\n', self.nInputPlane)
        #print('self.nOutputPlane\n', self.nOutputPlane)
        #print('self.kW\n', self.kW)
        
        # generate a mask of (0,1) with sparsity
        mask = np.ones(numElements)
        shred = int(numElements * (1 - self.spa))
        mask[:shred] = 0
        np.random.shuffle(mask)
        self.mask = torch.reshape(torch.from_numpy(mask), (self.nOutputPlane,self.nInputPlane,self.kW,self.kW)).type(torch.FloatTensor)
        #print('mask:\n', mask)
        
        # generate weights array of (-1,1)
        weights = torch.empty(self.nOutputPlane,self.nInputPlane,self.kW,self.kW).uniform_(0,1)
        self.weights = torch.bernoulli(weights)
        self.weights = torch.add(torch.mul(self.weights , 2), -1)
        #print('weights:\n', self.weights)
        
        # mask .* weights to get sparsity weights
        self.weight = Parameter(torch.mul(self.weights,self.mask))
        self.weight.requires_grad=False
        #print('final weight:\n', self.weight)
        
    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    #def forward(self, input):
        #return self.forward(input)