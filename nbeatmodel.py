'''
Author: philipperemy
Date: 2021-12-29 13:26:27
LastEditors: Egoist
LastEditTime: 2022-05-30 13:30:24
FilePath: /smp/nbeatmodel.py
Description: 
    Modify from pytorch implementation of nbeat by philipperemy
    source code at https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
'''
import pickle
import random
from time import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer
import torch.nn.utils.parametrize as parametrize


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
                 downsampling_factor=24,
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
        self.downsampling_factor=downsampling_factor
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
            block_init = self.select_block(stack_type)
            blocksettings=self.select_init_argd(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(block_id=block_id,
                                   device=self.device,
                                   units=self.hidden_layer_units,
                                   thetas_dim=self.thetas_dim[stack_id],
                                   backbone_layers=self.backbone_layers,
                                #    nb_harmonics=self.nb_harmonics,
                                   **blocksettings,
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

    def select_init_argd(self,block_type):
        if block_type == NBeatsNet.GENERIC_CNN:
            return {'downsampling_factor':self.downsampling_factor}
        else:
            return {'backcast_length':self.backcast_length,
                    'forecast_length':self.forecast_length,
                    'nb_harmonics':self.nb_harmonics,}

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

    def forward(self, history,future=None,step=1):#TODO CPC infoLoss
        '''future is use in contrastive learning for obtain theta pair'''
        backcast = squeeze_last_dim(history)
        future=squeeze_last_dim(future) if future is not None else None
        forecast = torch.zeros(size=(backcast.size()[0], step*self.downsampling_factor,))
        # forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        # forecast=0
        theta_pred=[]
        context=[]
        if future is not None:
            theta_cnn=[]
        else:
            theta_cnn=None
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                result=self.stacks[stack_id][block_id](backcast,y=future,step=step)
                    
                b, f = result['backcast'],result['forecast']
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
                theta_pred.append(result['theta_pred'])
                context.append(result['context'])
                if future is not None:
                    future=future.to(self.device)-result['future']
                    theta_cnn.append(result['theta_future'])

        # return squeeze_last_dim(history)-backcast, forecast
        return {'backcast':squeeze_last_dim(history)-backcast,
                'forecast':forecast,
                'context':torch.cat(context,1).flatten(start_dim=1),
                'theta_pred':torch.cat(theta_pred,1).flatten(start_dim=1),
                'theta_cnn':torch.cat(theta_cnn,1).flatten(start_dim=1) if theta_cnn is not None else None,}

    def inference(self,data,future,step,trmode,gd):
        data=data.to(self.device)
        if trmode is True:
            self.train()
        else:
            self.eval()

        if gd is True:
            return self(data,future,step)
        with torch.no_grad():
            return self(data,step=step)

    def plot_basis(self):
        thetas_max=max(self.thetas_dim)
        total_blocks=len(self.stacks)*self.nb_blocks_per_stack
        fig=plt.figure()
        fig.suptitle(f'basis of {self.name}')

        for idx_stack in range(len(self.stacks)):
            for idx_block in range(self.nb_blocks_per_stack):
                block_id=idx_stack*self.nb_blocks_per_stack+idx_block
                for idx_basis,basis in enumerate(self.stacks[idx_stack][idx_block].basis.weight,1):
                    ax=fig.add_subplot(total_blocks,thetas_max,block_id*thetas_max+idx_basis)
                    ax.set_title(f'block_{block_id+1} basis_{idx_basis}')
                    ax.plot(basis[0].detach().numpy())
                    # ax.legend()
        fig.tight_layout()
        # plt.show(block=False)
        return fig

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
                'downsampling_factor':self.downsampling_factor,
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

class PredictbyLSTM(nn.Module):
    def __init__(self,context_size,thetas_dim,predict_module_num_layers=1):
        super().__init__()
        self.hidden_size =context_size
        self.thetas_dim=thetas_dim
        proj_size=0 if context_size==thetas_dim else thetas_dim
        self.lstm=nn.LSTM(thetas_dim,context_size,num_layers=predict_module_num_layers,proj_size=proj_size)
        ...

    def forward(self,c,step=1): #origin c.shape=(batch,channel,seq)
        self.lstm.flatten_parameters()
        batch_size=c.shape[0]
        c=c.permute(2,0,1)
        
        x=torch.zeros(1,batch_size,self.thetas_dim,device=c.device) #shape=(seq,batch,channel)
        h=torch.zeros_like(x)
        o=[]
        for i in range(step):
            x,(h,c)=self.lstm(x,(h,c))
            o.append(x)
        o=torch.cat(o,0)
        return o.permute(1,2,0)

    def __str__(self):
        return f'         | -- {type(self).__name__}(layers={self.lstm}) at @{id(self)}'

class LogvarParametrize(nn.Module):
    def forward(self, X):
        return X.exp()

class GenericCNN(nn.Module):
    PREDICT_METHOD={#'pad':Predictbypadding,
                    # 'fc':Predictbyfc,
                    'lstm':PredictbyLSTM}

    def __init__(self,block_id,
                 device,
                 units,
                 thetas_dim,
                 backbone_layers=4,
                 share_thetas=False,
                 downsampling_factor=24,
                 context_size=None,predictModule=None,share_predict_module=False,backbone_kernel_size=3,**predict_module_setting):
        super(GenericCNN, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.share_thetas = share_thetas
        self.downsampling_factor=downsampling_factor
        self.device = device

        # cnnstack=[]
        # for i in range(backbone_layers):
        #     cnnstack=cnnstack+[nn.Conv1d(units if i!=0 else 1, units,backbone_kernel_size,padding='same'), nn.ReLU()]
        # self.cnnstack=nn.Sequential(*cnnstack)
        theta_kernel_size=downsampling_factor-backbone_layers*(backbone_kernel_size-1)
        self.cnnstack=nn.Sequential(*([j for i in range(backbone_layers)
                                            for j in (nn.Conv1d(units if i else 1, units,backbone_kernel_size),
                                                      nn.ReLU())]+
                                    [nn.Conv1d(units,thetas_dim,theta_kernel_size,stride=downsampling_factor,bias=False),
                                     nn.ReLU()]))
        
        # self.theta = nn.Sequential(
        #                 nn.Conv1d(units,thetas_dim,theta_kernel_size,stride=downsampling_factor,bias=False),
        #                 nn.ReLU(),)
        self.context_layer=nn.GRU(thetas_dim,context_size) if context_size is not None else None

        self.predictModule=self.build_predictModule(block_id=block_id,
                                                    predictModule=predictModule,
                                                    context_size=context_size,
                                                    share_predict_module=share_predict_module,
                                                    predict_module_setting=predict_module_setting)

        self.basis = nn.ConvTranspose1d(thetas_dim,1,self.downsampling_factor,stride=self.downsampling_factor,bias=False)
        parametrize.register_parametrization(self.basis, "weight", LogvarParametrize())
        ...

    def forward(self, x,y=None,step=1):
        '''always need context_layer and predictModule'''
# 		x = squeeze_last_dim(x)
        if y is not None:
            future_step=y.shape[1]//self.downsampling_factor
            y=y.to(self.device)
            x=torch.cat((x,y),-1)
        x=x.unsqueeze(1)
        x=x.to(self.device)
        z= self.cnnstack(x)
        # z=self.theta(x)
        if y is not None:
            z_back=z[...,:-future_step]
            z_future=z[...,-future_step:]
        else:
            z_back=z
            z_future=None
        # z_back=z[...,:-future_step]if y is not None else z
        self.context_layer.flatten_parameters()
        c=self.context_layer(z_back.permute(2,0,1))[0][-1:].permute(1,2,0)# keep batch,channel,seq(layers)
        z_pred=self.predictModule(c,step=step)
        # z=torch.cat([z,z_pred],-1)
        # x=self.basis(z)
        # return x[...,:-self.forecast_length].squeeze(1), x[...,-self.forecast_length:].squeeze(1)
        return {'backcast':self.basis(z_back).squeeze(1),
                'forecast':self.basis(z_pred).squeeze(1),
                'future':self.basis(z_future).squeeze(1) if z_future is not None else None,
                'context':c,
                'theta_pred':z_pred,
                'theta_future':z_future}
        return {'backcast':x[...,:-self.forecast_length].squeeze(1),
                'forecast':x[...,-self.forecast_length:].squeeze(1),
                'theta_pred':z_pred,
                'future_pred':self.basis(z_pred).squeeze(1)}

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'downsampling_factor={self.downsampling_factor},' \
               f'share_thetas={self.share_thetas}) at @{id(self)}\n' \
               f'{self.predictModule if self.predictModule is not None else "         | -- No predict module"}'
    
    def build_predictModule(self,block_id,predictModule,context_size,share_predict_module,predict_module_setting):
        if predictModule is not None:
            if block_id==0 or share_predict_module is False:
                predict_size={'context_size':context_size,
                              'thetas_dim':self.thetas_dim,}
                pm=self.PREDICT_METHOD.get(predictModule)(**predict_size,**predict_module_setting)
                self.setsharedpredictmodule(pm)
                return pm
            else:
                return self.sharedpredictModule
        else:
            return None

    @classmethod
    def setsharedpredictmodule(cls,module):
        cls.sharedpredictModule=module


if __name__=='__main__':
    gtest=GenericBlock(0,8,4,torch.device("cpu"),168,24)
    ctest=GenericCNN(0,8,4,torch.device("cpu"),168,24)
    # ctest2=GenericCNN(0,8,4,torch.device("cpu"),168,24,context_size=8,predictModule='fc',predict_module_layer=[6])
    ctest3=GenericCNN(0,8,4,torch.device("cpu"),168,24,context_size=8,predictModule='lstm')

    # ptest=Predictbyfc(torch.Size([4,7]),torch.Size([4,1]),predict_module_layer=[])
    k=gtest(torch.rand(5,168))
    k=ctest3(torch.rand(5,168))
    k=ctest3(torch.rand(5,168),y=torch.rand(5,24))
    ...