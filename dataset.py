import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.functional import one_hot

T0=[[1, 1, 1], 
    [0, 1, 0], 
    [0, 1, 0]]
T1=[[1, 0, 0],
    [1, 1, 1],
    [1, 0, 0]]
T2=[[0, 1, 0], 
    [0, 1, 0], 
    [1, 1, 1]]
T3=[[0, 0, 1], 
    [1, 1, 1], 
    [0, 0, 1]]

L0=[[1, 0, 0], 
    [1, 0, 0], 
    [1, 1, 1]]
L1=[[0, 0, 1],
    [0, 0, 1],
    [1, 1, 1]]
L2=[[1, 1, 1], 
    [0, 0, 1], 
    [0, 0, 1]]
L3=[[1, 1, 1], 
    [1, 0, 0], 
    [1, 0, 0]]

NAME2SAMPLE={'T0':T0,  'T1':T1,  'T2':T2,  'T3':T3,
             'L0':L0,  'L1':L1,  'L2':L2,  'L3':L3}
NAME2LABEL ={'T0':[1,],'T1':[1,],'T2':[1,],'T3':[1,],
             'L0':[0,],'L1':[0,],'L2':[0,],'L3':[0,]}
IDX2NAME={0:'T0',1:'T1',2:'T2',3:'T3',4:'L0',5:'L1',6:'L2',7:'L3'}

# T_SAMPLES=[T0,T1,T2,T3]
# L_SAMPLES=[L0,L1,L2,L3]

def to_tensor(inp):
    if isinstance(inp,list):
        return torch.tensor(inp)
    elif isinstance(inp,np.ndarray):
        return torch.tensor(inp)
    elif isinstance(inp,torch.Tensor):
        return inp.clone().detach()

def to_np(inp):
    if isinstance(inp,list):
        return np.ndarray(inp)
    elif isinstance(inp,np.ndarray):
        return inp
    elif isinstance(inp,torch.Tensor):
        return inp.numpy()
    
class TL_dataset(Dataset):
    # Ts,Ls=[],[]
    # for T,L in zip(T_SAMPLES,L_SAMPLES):
    #     Ts.append(to_tensor(T))
    #     Ls.append(to_tensor(L))

    def __init__(self,idx_list):
        super(TL_dataset,self).__init__()
        self.label_list=[]
        self.data_list=[]
        for idx in idx_list:
            name = IDX2NAME[idx]
            data = NAME2SAMPLE[name]
            label= NAME2LABEL[name]
            self.data_list.append(data)
            self.label_list.append(label)

        self.datas=to_tensor(self.data_list).float()
        self.labels=to_tensor(self.label_list).float()
        # import pdb;pdb.set_trace()
        # self.labels=one_hot(to_tensor(self.label_list))

    def __getitem__(self, index):
        idx = index%self.__len__() 
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        # self.len = self.datas.shape[0] #.item()
        return self.datas.shape[0]