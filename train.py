'''
Author: Egoist
Date: 2021-11-12 16:12:25
LastEditors: Egoist
LastEditTime: 2022-03-21 14:37:09
FilePath: /smp/train.py
Description: 

'''

import argparse
from readIHEPC import IHEPC
from readTMbase import TMbase
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
from torch.utils.tensorboard import SummaryWriter
# =============================================================================
# import matplotlib as mpl
# mpl.use('TkAgg')
# =============================================================================
import matplotlib.pyplot as plt
# from nbeats_pytorch.model import NBeatsNet
from nbeatmodel import NBeatsNet

import warnings
warnings.filterwarnings(action='ignore', message='Setting attributes')

class Trainer():
    def __init__(self,model,trloader,valloader,lossf,evmetric,opt,device,tb_log_dir,lossratio):
        self.model=model.to(device)
        self.trloader=trloader
        self.valloader=valloader
        self.lossf=lossf
        self.evmetric=evmetric
        self.opt=opt
        self.device=device
        self.writer=SummaryWriter(tb_log_dir)
        self.lossratio=lossratio
        self.logdf=pd.DataFrame(columns=['epoch','batch','iteration',
                                         'train_back','train_fore','train_all',
                                         'valiadate_back','valiadate_fore','valiadate_all'])

        print(f'params: {self.count_params()}')
        
    def train(self,epochs,record,save_log_path,save_model_path):
        iteration=-1
        for ep in range(epochs):
            for batch,(x,y) in enumerate(self.trloader):
                iteration+=1
                x,y=[i.squeeze(-1) for i in (x,y)]
                back,fore=self.inference(x,trmode=True,gd=True)

                loss=self.evaluate(x,y,back,fore,metric=self.lossf)

                self.update(sum([i*j for i,j in zip(loss,self.lossratio)]))

                
                if record[0]=='i' and iteration%(int(record[1:]))==0:
                    self.validate(ep,batch,iteration,loss,save_model_path)
                    
            if record[0]=='e':
                self.validate(ep,batch,iteration,loss,save_model_path)
        if save_log_path is not None:
            self.logdf.to_csv(save_log_path)
        ...

    def update(self,loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def validate(self,ep,batch,itrn,trainloss,save_model_path):
        verr=np.mean([self.evaluate(*[i.squeeze(-1) for i in (x,y)],
                                    *self.inference(x,trmode=False,gd=False),metric=self.evmetric,tocpu=True) for x,y in self.valloader],axis=0)

        stepstr=f'epoch/batch/iteration : {ep}/{batch}/{itrn}'
        trainstr=f'train back={trainloss[0].item():f} | train fore={trainloss[1].item():f} | train all={trainloss[2].item():f}'
        valstr=f'valiadate back={verr[0]:f} | valiadate fore={verr[1]:f} | valiadate all={verr[2]:f}'
        print(f'{stepstr} ][ {trainstr} ][ {valstr}')
        infodict={'epoch':[ep],'batch':[batch],'iteration':[itrn],
                'train_back':[trainloss[0].item()],'train_fore':[trainloss[1].item()],'train_all':[trainloss[2].item()],
                'valiadate_back':[verr[0]],'valiadate_fore':[verr[1]],'valiadate_all':[verr[2]]}
        isbestresult=self.add_log(infodict,selectby='valiadate_fore')
        if isbestresult and save_model_path is not None:
            self.model.save(save_model_path,other_info={'iteration':itrn})
        
        self.writer.add_scalar('train/all',trainloss[2].item(),itrn)
        self.writer.add_scalar('train/back',trainloss[0].item(),itrn)
        self.writer.add_scalar('train/fore',trainloss[1].item(),itrn)
        self.writer.add_scalar('validate/all',verr[2],itrn)
        self.writer.add_scalar('validate/back',verr[0],itrn)
        self.writer.add_scalar('validate/fore',verr[1],itrn)
        
        trainsample_x,trainsample_y=[torch.from_numpy(i) for i in self.trloader.dataset.getvisualbatch()]
        trainsample_b,trainsample_f=[i.cpu() for i in self.inference(trainsample_x,trmode=False,gd=False)]
        for idx,x,y,b,f in zip(self.trloader.dataset.visualindices,trainsample_x,trainsample_y,trainsample_b,trainsample_f):
            self.writer.add_figure(f'train_{idx}/all',self.plotall(x,y,b,f),itrn)
            self.writer.add_figure(f'train_{idx}/fore',self.plotfore(y,f),itrn)
            self.writer.add_figure(f'train_{idx}/back',self.plotback(x,b),itrn)
            
        valsample_x,valsample_y=[torch.from_numpy(i) for i in self.valloader.dataset.getvisualbatch()]
        valsample_b,valsample_f=[i.cpu() for i in self.inference(valsample_x,trmode=False,gd=False)]
        for idx,x,y,b,f in zip(self.valloader.dataset.visualindices,valsample_x,valsample_y,valsample_b,valsample_f):
            self.writer.add_figure(f'validate_{idx}/all',self.plotall(x,y,b,f),itrn)
            self.writer.add_figure(f'validate_{idx}/fore',self.plotfore(y,f),itrn)
            self.writer.add_figure(f'validate_{idx}/back',self.plotback(x,b),itrn)
        
        self.writer.flush()

    def inference(self,data,trmode,gd):
        data=data.to(self.device)
        if trmode is True:
            self.model.train()
        else:
            self.model.eval()

        if gd is True:
            return self.model(data)
        with torch.no_grad():
            return self.model(data)
    

    def evaluate(self,x,y,b,f,metric,tocpu=False):
        loss_back=metric(b,x.to(self.device))
        loss_fore=metric(f,y.to(self.device))
        loss_all =metric(torch.cat((b,f),-1),torch.cat((x,y),-1).to(self.device))
        loss=[i.cpu() if tocpu else i for i in (loss_back,loss_fore,loss_all)]
        return loss
    
    def plotall(self,x,y,b,f):
        xl,yl=len(x),len(y)
        
        fig, ax1 = plt.subplots()
        ax1.plot(range(xl+yl),torch.cat((x,y)),label='ground truth',color='g')
        ax1.plot(range(0,xl),b,label='back',color='b')
        ax1.plot(range(xl,xl+yl),f,label='forecast',color='r')
        
        ax1.legend()
        fig.tight_layout()
        return fig
    
    def plotfore(self,gt,fore):
        fl=len(fore)
        
        fig, ax1 = plt.subplots()
        ax1.plot(range(fl),gt,label='ground truth',color='g')
        ax1.plot(range(fl),fore,label='forecast',color='r')
        
        ax1.legend()
        fig.tight_layout()
        return fig

    def plotback(self,x,b):
        xl=len(x)

        fig, ax1 = plt.subplots()
        ax1.plot(range(xl),x,label='ground truth',color='g')
        ax1.plot(range(xl),b,label='BACKcast',color='b')
        
        ax1.legend()
        fig.tight_layout()
        return fig

    def add_log(self,infodict,selectby=None):
        # self.logdf=self.logdf.append(infodict,ignore_index=True) 
        self.logdf=pd.concat([self.logdf,pd.DataFrame(infodict)],ignore_index=True)
        if selectby is not None:
            return self.logdf[selectby].iloc[-1] <= self.logdf[selectby].min()

    def count_params(self,cond='all'):
        cond_f={'all':lambda x:True,
                'trainable':lambda x:x.requires_grad}.get(cond)
        return sum(p.numel() for p in self.model.parameters() if cond_f(p))

    # def save(self,path):
    # 	torch.save(self.model.state_dict(),path)

    # def load(self,path):
    # 	self.model.load_state_dict(torch.load(path))

    
class ARGS():
    def __init__(self): #TODO parse arg by tb_log_dir?
        parser=argparse.ArgumentParser()
        #dataset setting
        parser.add_argument('-ds','--dataset',type=str,default='IHEPC')
        # self.args=parser.parse_known_args()[0]
        parser.add_argument('-dp','--datapath',type=self.adddatasetprefix(parser),nargs='+',default=['dataset/IHEPC/household_power_consumption.txt'])
        parser.add_argument('-dr','--date_range',type=lambda s:None if s=='' else pd.Timestamp(s),nargs=2,default=None)
        parser.add_argument('-dc','--data_clean_threshold',type=lambda s:None if s=='' else self.data_clean_thresholdtodict(s),default=',100')
        # parser.add_argument('-nt','--nanThreshold',type=int,default=100)
        parser.add_argument('-nm','--normalized_method',type=str,default='z',choices=['z','max',''])
        parser.add_argument('-uc','--use_cols',type=str,default='g')
        parser.add_argument('-tu','--timeunit',type=int,default=60)
        parser.add_argument('-a','--align',type=int,default=60)
        
        #model setting
        parser.add_argument('-st','--stack_types',type=self.getstacktype,default='gg')
        parser.add_argument('-nbps','--nb_blocks_per_stack',type=int,default=2)
        parser.add_argument('-bbl','--backbone_layers',type=int,default=2)
        parser.add_argument('-bbk','--backbone_kernel_size',type=lambda s:None if s=='' else int(s),default=None) #for cnn block
        parser.add_argument('-fl','--forecast_length',type=int,default=24)
        parser.add_argument('-bl','--backcast_length',type=int,default=7*24)
        parser.add_argument('-tdim','--thetas_dim',type=self.tonumlist,default='4,4')
        parser.add_argument('-swis','--share_weights_in_stack',type=bool,default=False)
        parser.add_argument('-hlu','--hidden_layer_units',type=int,default=8)
        parser.add_argument('-pm','--predictModule',type=lambda s:None if s=='' else s,default=None) #for cnn block
        parser.add_argument('-pml','--predict_module_layer',type=lambda s:None if s=='' else self.tonumlist(s),default=None) #for cnn block
        parser.add_argument('-spm','--share_predict_module',type=lambda s:None if s=='' else bool(s),default=None) #for cnn block
        parser.add_argument('-pmhz','--predict_module_hidden_size',type=lambda s:None if s=='' else int(s),default=None) #for cnn block
        parser.add_argument('-pmnl','--predict_module_num_layers',type=lambda s:None if s=='' else int(s),default=None) #for cnn block

        #training setting
        parser.add_argument('-n','--name',type=str,default=None)
        parser.add_argument('-d','--cudadevice',type=int,default=0)
        parser.add_argument('-rs','--rngseed',type=int,default=None)
        parser.add_argument('-e','--epochs',type=int,default=35)
        parser.add_argument('-tbd','--tb_log_dir',type=self.addtbprefix(parser),default=None)
        parser.add_argument('-smp','--save_model_path',type=self.addmdlprefix(parser),default=None)
        parser.add_argument('-slp','--save_log_path',type=self.addlogprefix(parser),default=None)
        parser.add_argument('-r','--record',type=str,default='e') #i100
        parser.add_argument('-tb','--train_batch',type=int,default=512)
        # parser.add_argument('-vr','--valid_ratio',type=int,default=0.1)
        parser.add_argument('-vb','--valid_batch',type=int,default=512)
        parser.add_argument('-ss','--samplesize',type=int,default=8)
        parser.add_argument('-tl','--trainlosstype',type=self.lossfunc,default='huber')
        parser.add_argument('-lr','--lossratio',type=self.tofloatlist,default='0,0,1')
        parser.add_argument('-em','--evaluatemetric',type=self.lossfunc,default='mae')
        parser.add_argument('-opt','--optimizer',type=self.optalgo,default='adam')
        
        self.args=parser.parse_args()
        print(f'use args : =============')
        print(f'{self.args}')
        self.device=self.checkDevice(self.args.cudadevice)
        self.globalrng=np.random.default_rng(seed=self.args.rngseed)
        self.datasetprep={'IHEPC':IHEPC,
                          'TMbase':TMbase}.get(self.dataset)
        ...
        
    def __getattr__(self,key):
        return getattr(self.args,key)

    @staticmethod
    def data_clean_thresholdtodict(s):
        return {i:float(j) for i,j in zip(['value','length'],s.split(',')) if j!=''}

    @staticmethod
    def addtbprefix(parser):
        args=parser.parse_known_args()[0]
        def func(s):
            return f'runs/{s}/{args.name}' if s!='' else None
        return func

    @staticmethod
    def addmdlprefix(parser):
        args=parser.parse_known_args()[0]
        def func(s):
            prefix=f'model/{s}'
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            return f'{prefix}/{args.name}.mdl' if s!='' else None
        return func

    @staticmethod
    def addlogprefix(parser):
        args=parser.parse_known_args()[0]
        def func(s):
            prefix=f'log/{s}'
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            return f'{prefix}/{args.name}.csv' if s!='' else None
        return func


    @staticmethod
    def adddatasetprefix(parser):
        args=parser.parse_known_args()[0]
        def func(s):
            return f'dataset/{args.dataset}/{s}'
        return func

    @staticmethod
    def getstacktype(s):
        stacktype={'g':NBeatsNet.GENERIC_BLOCK,'s':NBeatsNet.SEASONALITY_BLOCK,
                't':NBeatsNet.TREND_BLOCK,'c':NBeatsNet.GENERIC_CNN}
        return [stacktype.get(i) for i in s]
    
    @staticmethod
    def tonumlist(s):
        return [int(i) for i in s.split(',')]
    
    @staticmethod
    def tofloatlist(s):
        return [float(i) for i in s.split(',')]

    @staticmethod
    def lossfunc(s):
        lossdict={'mae':nn.L1Loss,
                'mse':nn.MSELoss,
                'huber':nn.HuberLoss,}
        return lossdict.get(s)

    @staticmethod
    def optalgo(s):
        optdict={'adam':torch.optim.Adam,
                'sgd':torch.optim.SGD,}
        return optdict.get(s)

    @staticmethod
    def checkDevice(cudadevice):
        print('=============================')
        print(f'python ver. {sys.version}')
        print(f'pytorch ver. {torch.__version__}')
        print(f'cuda ver. {torch.version.cuda}')
        print(f'cuda avail : {(cuava:=torch.cuda.is_available())}')
        print(f'use device: {(dev:=torch.device(f"cuda:{cudadevice}" if cuava else "cpu"))}')
        print('=============================')
        return dev

    def get_argdict(self):
        return {'datasetprep':self.datasetprep,
                'datapath':self.datapath,
                'date_range':self.date_range,
                'data_clean_threshold':self.data_clean_threshold,
                'normalized_method':self.normalized_method,
                'use_cols':self.use_cols,
                'timeunit':self.timeunit,
                'align':self.align,
                'forecast_length':self.forecast_length,
                'backcast_length':self.backcast_length,
                'globalrng':self.globalrng,
                'samplesize':self.samplesize,
                'train_batch':self.train_batch,
                'valid_batch':self.valid_batch,
                'name':self.name,
                'device':self.device,
                'stack_types':self.stack_types,
                'nb_blocks_per_stack':self.nb_blocks_per_stack,
                'thetas_dim':self.thetas_dim,
                'share_weights_in_stack':self.share_weights_in_stack,
                'hidden_layer_units':self.hidden_layer_units,
                'backbone_layers':self.backbone_layers,
                'backbone_kernel_size':self.backbone_kernel_size,
                'predictModule':self.predictModule,
                'predict_module_layer':self.predict_module_layer,
                'share_predict_module':self.share_predict_module,
                'predict_module_hidden_size':self.predict_module_hidden_size,
                'predict_module_num_layers':self.predict_module_num_layers,
                'trainlosstype':self.trainlosstype,
                'evaluatemetric':self.evaluatemetric,
                'optimizer':self.optimizer,
                'tb_log_dir':self.tb_log_dir,
                'lossratio':self.lossratio,
                'epochs':self.epochs,
                'record':self.record,
                'save_log_path':self.save_log_path,
                'save_model_path':self.save_model_path}
    
def main(datasetprep,datapath,date_range,data_clean_threshold,normalized_method,
         use_cols,timeunit,align,forecast_length,backcast_length,
         globalrng,samplesize,train_batch,valid_batch,
         
         name,device,stack_types,nb_blocks_per_stack,thetas_dim,share_weights_in_stack,
         hidden_layer_units,backbone_layers,backbone_kernel_size,
         predictModule,predict_module_layer,share_predict_module,predict_module_hidden_size,predict_module_num_layers,
         
         trainlosstype,evaluatemetric,optimizer,tb_log_dir,lossratio,
         epochs,record,save_log_path,save_model_path):
    dataset=datasetprep(datapath=datapath,
                        date_range=date_range,
                        #  nanThreshold=args.nanThreshold,
                        data_clean_threshold=data_clean_threshold,
                        normalized_method=normalized_method,
                        use_cols=use_cols,
                        timeunit=timeunit,
                        align=align,
                        forecast_length=forecast_length,
                        backcast_length=backcast_length,
                        globalrng=globalrng,
                        samplesize=samplesize,
                        train_batch=train_batch,
                        valid_batch=valid_batch,)
    net = NBeatsNet(name=name,
                    device=device,
                    stack_types=stack_types,
                    nb_blocks_per_stack=nb_blocks_per_stack,
                    forecast_length=forecast_length,
                    backcast_length=backcast_length,
                    thetas_dim=thetas_dim,
                    share_weights_in_stack=share_weights_in_stack,
                    hidden_layer_units=hidden_layer_units,
                    backbone_layers=backbone_layers,
                    backbone_kernel_size=backbone_kernel_size,
                    predictModule=predictModule,
                    predict_module_layer=predict_module_layer,
                    share_predict_module=share_predict_module,
                    predict_module_hidden_size=predict_module_hidden_size,
                    predict_module_num_layers=predict_module_num_layers)
    exp=Trainer(model=net,
                trloader=dataset.trainloader,
                valloader=dataset.valloader,
                lossf=trainlosstype(),
                evmetric=evaluatemetric(),
                opt=optimizer(net.parameters()),
                device=device,
                tb_log_dir=tb_log_dir,
                lossratio=lossratio)
    exp.train(epochs=epochs,
              record=record,
              save_log_path=save_log_path,
              save_model_path=save_model_path)
    ...

if __name__=='__main__':
    args=ARGS()
    default_argdict=args.get_argdict()
    loop_exec=[{},]
    for d in loop_exec:
        main(**(default_argdict|d))