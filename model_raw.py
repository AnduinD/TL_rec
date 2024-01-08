# model.py
import numpy as np

def basic_act(x):
    return 1/(1+np.e**(-x))

    
class neuron:
    def __init__(self, in_channel=9, out_channel=3, act=basic_act):
        self.weight = np.random.rand(in_channel, out_channel)
        self.bias   = np.random.rand(out_channel)
        self.act = act

    def __call__(self, x, **kwargs):
        x = x.reshape(-1)
        return self.act(np.dot(x, self.weight) + self.bias, **kwargs)
    def forward(self, x, **kwargs):
        return self.__call__(x, **kwargs)
    
    def backward(self, x, y, lr=0.1, **kwargs):
        y_ = self.forward(x, **kwargs)
        delta = (y-y_)*y_*(1-y_)
        self.weight += lr*np.dot(delta, x)
        self.bias   += lr*delta
        return delta

class MLP:
    def __init__(self, in_channel=9, hidden_channel=3, out_channel=1, alpha=1., act=basic_act):
        super(MLP, self).__init__()
        def hid_act(x,alpha=alpha):
            return 2*act(alpha*x)-1
        def out_act(x,alpha=alpha):
            return act(alpha*x)
        self.alpha = alpha

        self.unit1 = neuron(in_channel, hidden_channel, hid_act)
        self.unit2 = neuron(hidden_channel, out_channel,out_act)

    def __call__(self, x):
        x = self.unit1(x, alpha=self.alpha)
        x = self.unit2(x, alpha=self.alpha)
        return x
    def forward(self, x):
        return self.__call__(x)
    
    def backward(self, x, y, lr=0.1):
        x = x.reshape(-1)
        y_ = self.forward(x)
        self.unit2.error = y - y_ # 1,
        self.unit2.delta = self.unit2.error*y_*(1-y_)*self.alpha # 1,

        y__ = self.unit1(x)
        self.unit1.error = np.dot(self.unit2.weight, self.unit2.delta) # 3,
        self.unit1.delta = self.unit1.error*2*y__*(1-y__)*self.alpha # 3,
        
        self.unit2.weight += lr*self.unit2.delta*np.atleast_2d(y__).T # 3,1
        self.unit2.bias   += lr*self.unit2.delta
        self.unit1.weight += lr*self.unit1.delta*np.atleast_2d(x).T # 9,3
        self.unit1.bias   += lr*self.unit1.delta
