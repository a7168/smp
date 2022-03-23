'''
Author: philipperemy
Date: 2021-12-29 13:26:27
LastEditors: Egoist
LastEditTime: 2022-03-23 09:02:10
FilePath: /smp/nbeatmodel.py
Description: 
    Modify from pytorch implementation of nbeat by philipperemy
    source code at https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
'''
import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer


class NBeatsNet(nn.Module):#TODO loss computation belong to model
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    GENERIC_CNN = 'GenericCNN'

    def __init__(self,
                 name,
                 device=torch.device('cpu'),
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 backbone_layers=4,
                 nb_harmonics=None,
                 **argd):
        super(NBeatsNet, self).__init__()
        self.name=name
        self.argd={i:j for i,j in argd.items() if j is not None}
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.backbone_layers=backbone_layers
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        print('| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(block_id=block_id,
                                   units=self.hidden_layer_units,
                                   thetas_dim=self.thetas_dim[stack_id],
                                   device=self.device,
                                   backcast_length=self.backcast_length,
                                   forecast_length=self.forecast_length,
                                   backbone_layers=self.backbone_layers,
                                   nb_harmonics=self.nb_harmonics,
                                   **self.argd)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    # def save(self, filename: str):
    #     torch.save(self, filename)

    # @staticmethod
    # def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    #     return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        elif block_type == NBeatsNet.GENERIC_CNN:
            return GenericCNN
        else:
            return GenericBlock

    def compile(self, loss: str, optimizer: Union[str, Optimizer]):
        if loss == 'mae':
            loss_ = l1_loss
        elif loss == 'mse':
            loss_ = mse_loss
        elif loss == 'cross_entropy':
            loss_ = cross_entropy
        elif loss == 'binary_crossentropy':
            loss_ = binary_cross_entropy
        else:
            raise ValueError(f'Unknown loss name: {loss}.')
        # noinspection PyArgumentList
        if isinstance(optimizer, str):
            if optimizer == 'adam':
                opt_ = optim.Adam
            elif optimizer == 'sgd':
                opt_ = optim.SGD
            elif optimizer == 'rmsprop':
                opt_ = optim.RMSprop
            else:
                raise ValueError(f'Unknown opt name: {optimizer}.')
            opt_ = opt_(lr=1e-4, params=self.parameters())
        else:
            opt_ = optimizer
        self._opt = opt_
        self._loss = loss_

    def fit(self, x_train, y_train, validation_data=None, epochs=10, batch_size=32):

        def split(arr, size):
            arrays = []
            while len(arr) > size:
                slice_ = arr[:size]
                arrays.append(slice_)
                arr = arr[size:]
            arrays.append(arr)
            return arrays

        for epoch in range(epochs):
            x_train_list = split(x_train, batch_size)
            y_train_list = split(y_train, batch_size)
            assert len(x_train_list) == len(y_train_list)
            shuffled_indices = list(range(len(x_train_list)))
            random.shuffle(shuffled_indices)
            self.train()
            train_loss = []
            timer = time()
            for batch_id in shuffled_indices:
                batch_x, batch_y = x_train_list[batch_id], y_train_list[batch_id]
                self._opt.zero_grad()
                _, forecast = self(torch.tensor(batch_x, dtype=torch.float).to(self.device))
                loss = self._loss(forecast, squeeze_last_dim(torch.tensor(batch_y, dtype=torch.float).to(self.device)))
                train_loss.append(loss.item())
                loss.backward()
                self._opt.step()
            elapsed_time = time() - timer
            train_loss = np.mean(train_loss)

            test_loss = '[undefined]'
            if validation_data is not None:
                x_test, y_test = validation_data
                self.eval()
                _, forecast = self(torch.tensor(x_test, dtype=torch.float).to(self.device))
                test_loss = self._loss(forecast, squeeze_last_dim(torch.tensor(y_test, dtype=torch.float))).item()

            num_samples = len(x_train_list)
            time_per_step = int(elapsed_time / num_samples * 1000)
            print(f'Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs}')
            print(f'{num_samples}/{num_samples} [==============================] - '
                  f'{int(elapsed_time)}s {time_per_step}ms/step - '
                  f'loss: {train_loss:.4f} - val_loss: {test_loss:.4f}')

    def predict(self, x, return_backcast=False):
        self.eval()
        b, f = self(torch.tensor(x, dtype=torch.float).to(self.device))
        b, f = b.detach().numpy(), f.detach().numpy()
        if len(x.shape) == 3:
            b = np.expand_dims(b, axis=-1)
            f = np.expand_dims(f, axis=-1)
        if return_backcast:
            return b
        return f

    def forward(self, rawbackcast,reStacks=False):
        backcast = squeeze_last_dim(rawbackcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        stacks=[]
        for stack_id in range(len(self.stacks)):
            sf=torch.zeros_like(forecast)
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
                sf=sf.to(self.device) + f
            stacks.append(sf)
        if reStacks:
            return squeeze_last_dim(rawbackcast)-backcast, forecast, stacks
        return squeeze_last_dim(rawbackcast)-backcast, forecast

    def inference(self,data,trmode,gd):
        data=data.to(self.device)
        if trmode is True:
            self.train()
        else:
            self.eval()

        if gd is True:
            return self(data)
        with torch.no_grad():
            return self(data)

    def count_params(self,cond='all'):
        cond_f={'all':lambda x:True,
                'trainable':lambda x:x.requires_grad}.get(cond)
        return sum(p.numel() for p in self.parameters() if cond_f(p))

    def save(self,path,other_info={}):
        torch.save({'type':'nbeats',
                    'infodict':self.get_infodict(),
                    'weight':self.state_dict()}|other_info,path)

    @classmethod
    def build(cls,path,new_name=None,new_device=None):
        mdict=torch.load(path,map_location=new_device)
        ndict={'name':new_name,'device':new_device,}
        if mdict['type']=='nbeats':
            model=cls(**mdict['infodict']|{i:j for i,j in ndict.items() if j is not None})
            model.load_state_dict(mdict['weight'])
            return model
        else:
            print('this is not nbeats.')
        ...


    def loadbyfile(self,path,map_location=None):
        self.load_state_dict(torch.load(path,map_location=map_location))

    def get_infodict(self):
        return {'name':self.name,
                'device':self.device,
                'stack_types':self.stack_types,
                'nb_blocks_per_stack':self.nb_blocks_per_stack,
                'forecast_length':self.forecast_length,
                'backcast_length':self.backcast_length,
                'thetas_dim':self.thetas_dim,
                'share_weights_in_stack':self.share_weights_in_stack,
                'hidden_layer_units':self.hidden_layer_units,
                'backbone_layers':self.backbone_layers,
                'nb_harmonics':self.nb_harmonics,
                } | self.argd

def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length):
    ls = np.arange(-backcast_length, forecast_length, 1) / forecast_length
    b_ls = np.abs(np.flip(ls[:backcast_length]))
    f_ls = ls[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, backbone_layers=4,share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas

        fcstack=[]
        for i in range(backbone_layers):
            fcstack=fcstack+[nn.Linear(units if i!=0 else backcast_length, units), nn.ReLU()]
        self.fcstack=nn.Sequential(*fcstack)

        self.device = device
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = x.to(self.device)
        x = self.fcstack(x)
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, backbone_layers=4, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, backbone_layers, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, backbone_layers, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, backbone_layers=4,nb_harmonics=None):
       super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, backbone_layers, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, backbone_layers=4, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length, backbone_layers)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast

class Predictbypadding(nn.Module):
    def __init__(self,insize,outsize):
        super().__init__()
        self.insize=insize
        self.outsize=outsize

    def forward(self,x):
        return torch.zeros(len(x),*self.outsize,device=x.device)
    
    def __str__(self):
        return f'         | -- {type(self).__name__}(layers=None) at @{id(self)}'

class Predictbyfc(nn.Module):
    def __init__(self,insize,outsize,predict_module_layer=[]):
        super().__init__()
        self.insize=insize
        self.outsize=outsize
        self.flat=nn.Flatten()
        self.nodes=[insize.numel()]+predict_module_layer+[outsize.numel()]
        self.fc=nn.ModuleList([nn.Sequential(
            nn.Linear(i,j),
            nn.ReLU(),
            ) for i,j in zip(self.nodes,self.nodes[1:])]
        )

    def forward(self,x):
        x=self.flat(x)
        for layer in self.fc:
            x=layer(x)
        return x.reshape(len(x),*self.outsize)

    def __str__(self):
        return f'         | -- {type(self).__name__}(layers={self.nodes}) at @{id(self)}'

class PredictbyLSTM(nn.Module):
    def __init__(self,insize,outsize,predict_module_hidden_size=None,predict_module_num_layers=1):
        super().__init__()
        self.insize=insize
        self.outsize=outsize
        self.lstm=(nn.LSTM(insize[0],outsize[0],num_layers=predict_module_num_layers) 
                   if predict_module_hidden_size is None
                   else nn.LSTM(insize[0],predict_module_hidden_size,num_layers=predict_module_num_layers,proj_size=outsize[0]))
        ...

    def forward(self,x):
        x=x.permute(2,0,1) #origin(batch,channel,seq)
        self.lstm.flatten_parameters()
        o,(h,c)=self.lstm(x)
        return o[-1:].permute(1,2,0)

    def __str__(self):
        return f'         | -- {type(self).__name__}(layers={self.lstm}) at @{id(self)}'

#TODO predictbyseq2seq
class GenericCNN(nn.Module):
    PREDICT_METHOD={'pad':Predictbypadding,
                    'fc':Predictbyfc,
                    'lstm':PredictbyLSTM}

    def __init__(self,block_id,units,thetas_dim,device,backcast_length=10,forecast_length=5,backbone_layers=4,
                 share_thetas=False,nb_harmonics=None,predictModule=None,share_predict_module=False,backbone_kernel_size=7,**predict_module_setting):
        super(GenericCNN, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.device = device

        if predictModule is not None:
            if block_id==0 or share_predict_module is False:
                self.predictModule=self.PREDICT_METHOD.get(predictModule)(torch.Size([thetas_dim,7]),
                                                torch.Size([thetas_dim,1]),**predict_module_setting)
                self.setsharedpredictmodule(self.predictModule)
            else:
                self.predictModule=self.sharedpredictModule
        else:
            self.predictModule=None

        cnnstack=[]
        for i in range(backbone_layers):
            cnnstack=cnnstack+[nn.Conv1d(units if i!=0 else 1, units,backbone_kernel_size,padding='same'), nn.ReLU()]
        self.cnnstack=nn.Sequential(*cnnstack)
		
        self.theta = nn.Sequential(
			nn.Conv1d(units,thetas_dim,24,stride=24,bias=False),
            nn.ReLU(),
			)
        self.basis = nn.ConvTranspose1d(thetas_dim,1,24,stride=24) if predictModule is not None else nn.Sequential(
            nn.ConvTranspose1d(thetas_dim,thetas_dim,24,stride=24),
            nn.ReLU(),
            nn.ConvTranspose1d(thetas_dim,1,25),
        )
        ...

    def forward(self, x):
# 		x = squeeze_last_dim(x)
        x=x.unsqueeze(1)
        x=x.to(self.device)
        x= self.cnnstack(x)
        x=self.theta(x)

        if self.predictModule is not None:
            x=torch.cat([x,self.predictModule(x)],-1)
        x=self.basis(x)
        return x[...,:-self.forecast_length].squeeze(1), x[...,-self.forecast_length:].squeeze(1)

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}\n' \
               f'{self.predictModule if self.predictModule is not None else "         | -- No predict module"}'
    
    @classmethod
    def setsharedpredictmodule(cls,module):
        cls.sharedpredictModule=module


if __name__=='__main__':
    gtest=GenericBlock(0,8,4,torch.device("cpu"),168,24)
    ctest=GenericCNN(0,8,4,torch.device("cpu"),168,24)

    ptest=Predictbyfc(torch.Size([4,7]),torch.Size([4,1]),predict_module_layer=[])
    k=gtest(torch.rand(5,168))
    k=ctest(torch.rand(5,168))
    ...