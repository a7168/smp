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
                 device=torch.device('cpu'),
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 **argd):
        super(NBeatsNet, self).__init__()
        self.argd={i:j for i,j in argd.items() if j is not None}
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
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
                block = block_init(block_id, self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.device, self.backcast_length, self.forecast_length, self.nb_harmonics, **self.argd)
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

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

    def count_params(self,cond='all'):
        cond_f={'all':lambda x:True,
                'trainable':lambda x:x.requires_grad}.get(cond)
        return sum(p.numel() for p in self.parameters() if cond_f(p))

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

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
       super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

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

class PredictbyLSTM(nn.Module): #TODO LSTM layer proj?
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
        _,(h,c)=self.lstm(x)
        return h[-1:].permute(1,2,0)

    def __str__(self):
        return f'         | -- {type(self).__name__}(layers={self.lstm}) at @{id(self)}'

#TODO predictbyseq2seq
class GenericCNN(nn.Module):
    PREDICT_METHOD={'pad':Predictbypadding,
                    'fc':Predictbyfc,
                    'lstm':PredictbyLSTM}

    def __init__(self, block_id, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None,predictModule='pad',share_predict_module=False,**predict_module_setting):
        super(GenericCNN, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas

        if block_id==0 or share_predict_module is False:
            self.predictModule=self.PREDICT_METHOD.get(predictModule)(torch.Size([thetas_dim,7]),
                                            torch.Size([thetas_dim,1]),**predict_module_setting)
            self.setsharedpredictmodule(self.predictModule)
        else:
            self.predictModule=self.sharedpredictModule
        self.device = device

        self.cnn1 = nn.Sequential(
			nn.Conv1d(1,units,6)
			)
        self.cnn2 = nn.Sequential(
			nn.Conv1d(units,units,7)
			)
        self.cnn3 = nn.Sequential(
			nn.Conv1d(units,units,7)
			)
        self.cnn4 = nn.Sequential(
			nn.Conv1d(units,units,7)
			)
		
        self.theta = nn.Sequential(
			nn.Conv1d(units,thetas_dim,25,stride=20,bias=False),
            nn.ReLU(),
			)
        self.basis = nn.Sequential(
            # nn.ConstantPad1d((0,1), 0),
            nn.ConvTranspose1d(thetas_dim,thetas_dim,25,stride=20,output_padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(thetas_dim,1,24),
        )

    def forward(self, x):
# 		x = squeeze_last_dim(x)
        x=x.unsqueeze(1)
        x = F.relu(self.cnn1(x.to(self.device)))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))

        x=self.theta(x)
        x=torch.cat([x,self.predictModule(x)],-1)
        x=self.basis(x)
        return x[...,:-self.forecast_length].squeeze(1), x[...,-self.forecast_length:].squeeze(1)

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}\n' \
               f'{self.predictModule}'
    
    @classmethod
    def setsharedpredictmodule(cls,module):
        cls.sharedpredictModule=module


if __name__=='__main__':
    gtest=GenericBlock(8,4,torch.device("cpu"),168,24)
    ctest=GenericCNN(8,4,torch.device("cpu"),168,24)

    ptest=Predictbyfc(torch.Size([4,7]),torch.Size([4,1]),predict_module_layer=[])
    k=ctest(torch.rand(5,168))
    ...