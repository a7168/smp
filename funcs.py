'''
Author: Egoist
Date: 2022-03-31 08:30:25
LastEditors: Egoist
LastEditTime: 2022-04-11 08:57:30
FilePath: /smp/funcs.py
Description: 

'''
# %%
import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def forward(self,data):
        return (data-self.mean)/self.std


class MAPE(nn.Module):
    def __init__(self,reduction='mean',eps=1e-8,percentage=False):
        super().__init__()
        self.reduction=reduction
        self.eps=eps
        self.percentage=percentage
        self.mae=nn.L1Loss(reduction='none')

    def forward(self,output,target):
        err_mae=self.mae(output,target)
        err_mape=err_mae/(target.abs()+self.eps)
        if self.percentage:
            err_mape=err_mape*100
        if self.reduction=='none':
            return err_mape
        elif self.reduction=='mean':
            return err_mape.mean()
        elif self.reduction=='sum':
            return err_mape.sum()
        raise TypeError(f'unknown reduction type {self.reduction}')

class pMAPE(nn.Module):
    def __init__(self,reduction='mean',tau=0.6,eps=1e-8,percentage=False):
        super().__init__()
        self.reduction=reduction
        self.tau=tau
        self.eps=eps
        self.percentage=percentage
        self.mae=nn.L1Loss(reduction='none')

    def forward(self,output,target):
        err_mae=self.mae(output,target)
        err_mape=err_mae/(target.abs()+self.eps)
        scale=torch.where(target>=output,2*self.tau,2*(1-self.tau))# if over then scale tau, under scale 1-tau
        err_mape=scale*err_mape
        if self.percentage:
            err_mape=err_mape*100
        if self.reduction=='none':
            return err_mape
        elif self.reduction=='mean':
            return err_mape.mean()
        elif self.reduction=='sum':
            return err_mape.sum()
        raise TypeError(f'unknown reduction type {self.reduction}')

def _test_MAPE():
    x=torch.rand(10)
    y=torch.rand(10)
    loss_func=MAPE(reduction='none',percentage=True)
    print(f'x={x}')
    print(f'y={y}')
    print(f'err={loss_func(x,y)}')
    ...

def _test_pMAPE():
    x=torch.rand(10)
    y=torch.rand(10)
    loss_pmae=pMAPE(reduction='none')
    loss_func=MAPE(reduction='none')
    print(f'x={x}')
    print(f'y={y}')
    print(f'pmape_err={loss_pmae(x,y)}')
    print(f'mape_err={loss_func(x,y)}')
    ...

if __name__=='__main__':
    _test_pMAPE()

