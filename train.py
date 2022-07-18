'''
Author: Egoist
Date: 2021-11-12 16:12:25
LastEditors: Egoist
LastEditTime: 2022-07-17 19:46:00
FilePath: /smp/train.py
Description: 

'''
# %%
import argparse
from readIHEPC import IHEPC
from readTMbase import TMbase,TMbaseset
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
import json
from torch.utils.tensorboard import SummaryWriter
# =============================================================================
# import matplotlib as mpl
# mpl.use('TkAgg')
# =============================================================================
import matplotlib.pyplot as plt
# from nbeats_pytorch.model import NBeatsNet
from nbeatmodel import NBeatsNet
import funcs

import warnings
warnings.filterwarnings(action='ignore', message='Setting attributes')

class Trainer():
    def __init__(self,name,expname,model,trloader,negloader,valloader,lossf,evmetric,opt,device,lossratio,samplesize):
        self.name=name
        self.expname=expname
        self.init_record()
        self.model=model.to(device)
        self.trloader=trloader
        self.negloader=negloader
        self.valloader=valloader
        self.lossf=lossf
        self.evmetric=evmetric
        self.opt=opt
        self.device=device
        # self.writer=SummaryWriter(tb_log_dir)
        self.lossratio=lossratio
        self.samplesize=samplesize
        self.logdf=pd.DataFrame(columns=['epoch','batch','iteration',
                                         'train_back','train_fore','train_all','info_NCE',
                                         'valiadate_back','valiadate_fore','valiadate_all'])

        print(f'params: {self.count_params()}')
        
    def init_record(self):
        if self.expname is None:
            self.tb_log_dir=None
            self.save_log_path=None
            self.save_model_path=None
            self.writer=None
        else:
            # prefix=f'exp/{self.expname}'
            # if not os.path.isdir(prefix):
            #     os.makedirs(f'{prefix}/run')
            #     os.makedirs(f'{prefix}/log')
            #     os.makedirs(f'{prefix}/model')
            tb_log_dir=f'exp/{self.expname}/run/{self.name}'
            self.save_log_path=f'exp/{self.expname}/log/{self.name}.csv'
            self.save_model_path=f'exp/{self.expname}/model/{self.name}.mdl'
            self.writer=SummaryWriter(tb_log_dir)
        
    def train(self,epochs,record,):
        iteration=-1
        for ep in range(epochs):
            if self.negloader is not None:
                neg_iter=iter(self.negloader)
                cross_entropy=nn.CrossEntropyLoss()
            else:
                neg_iter=None

            # for batch,(x,y) in enumerate(self.trloader):
            for batch,x in enumerate(self.trloader):
                iteration+=1
                x,y=x.split([7*24,1*24],dim=1)
                x,y=[i.squeeze(-1) for i in (x,y)]
                result=self.model.inference(x,y,step=1,trmode=True,gd=True)
                # r2=self.model.inference(x,y,step=3,trmode=True,gd=True)
                # r3=self.model.inference(x,y,step=3,trmode=False,gd=False)
                back,fore=result['backcast'],result['forecast']
                loss=self.evaluate(x,y,back,fore,metric=self.lossf)

                if neg_iter is not None:
                    nx,ny=next(neg_iter).split([7*24,1*24],dim=1)
                    result2=self.inference(nx,ny,step=1,trmode=True,gd=True)
                    loss.append(self.evaluate_infoNCE(result,result2,cross_entropy))

                self.update(sum([i*j for i,j in zip(loss,self.lossratio)]))

                # context=torch.cat([result['context'],result2['context']])
                # embeding={'mat':context,'metadata':['+']*len(result['context'])+['-']*len(result2['context'])}
                if record[0]=='i' and iteration%(int(record[1:]))==0:
                    self.validate(ep,batch,iteration,loss,)
                    
            if record[0]=='e':
                self.validate(ep,batch,iteration,loss,)
        if self.save_log_path is not None:
            self.logdf.to_csv(self.save_log_path)
        ...

    def update(self,loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def validate(self,ep,batch,itrn,trainloss,):
        err_batch=[]
        for x in self.valloader:
            x,y=x.split([7*24,1*24],dim=1)
            result=self.inference(x,future=None,step=1,trmode=False,gd=False)
            err=self.evaluate(x.squeeze(-1),y.squeeze(-1),result['backcast'],result['forecast'],metric=self.evmetric,tocpu=True)
            err_batch.append(err)
        verr=np.mean(err_batch,axis=0)

        stepstr=f'epoch/batch/iteration : {ep}/{batch}/{itrn}'

        infoNCE_str=f'None' if len(trainloss)<4 else f'{trainloss[3].item():f}'
        trainstr=f'train back={trainloss[0].item():f} | train fore={trainloss[1].item():f} | train all={trainloss[2].item():f} | infoNCE={infoNCE_str}'
        
        valstr=f'valiadate back={verr[0]:f} | valiadate fore={verr[1]:f} | valiadate all={verr[2]:f}'
        # print(f'{stepstr} ][ {trainstr} ][ {valstr}')
        print(f'{stepstr}',f'    {trainstr}',f'    {valstr}',sep='\n')
        infodict={'epoch':[ep],'batch':[batch],'iteration':[itrn],
                'train_back':[trainloss[0].item()],'train_fore':[trainloss[1].item()],'train_all':[trainloss[2].item()],
                'info_NCE':[np.nan if len(trainloss)<4 else trainloss[3].item()],
                'valiadate_back':[verr[0]],'valiadate_fore':[verr[1]],'valiadate_all':[verr[2]]}
        isbestresult=self.add_log(infodict,selectby='valiadate_all')
        if isbestresult and self.save_model_path is not None:
            self.model.save(self.save_model_path,other_info={'iteration':itrn})
        if self.writer is not None:
            self.record_tensorboard()

    def inference(self,data,future,step,trmode,gd):
        data=data.to(self.device)
        if trmode is True:
            self.model.train()
        else:
            self.model.eval()

        if gd is True:
            return self.model(data,future,step)
        with torch.no_grad():
            return self.model(data)
    

    def evaluate(self,x,y,b,f,metric,tocpu=False):
        loss_back=metric(b,x.to(self.device))
        loss_fore=metric(f,y.to(self.device))
        loss_all =metric(torch.cat((b,f),-1),torch.cat((x,y),-1).to(self.device))
        loss=[i.cpu() if tocpu else i for i in (loss_back,loss_fore,loss_all)]
        return loss

    @staticmethod
    def evaluate_backcast(input,result,metric):
        backcast=result['backcast']
        history=input['history']
        device=backcast.device
        return metric(backcast,history.to(device))

    @staticmethod
    def evaluate_forecast(input,result,metric):
        forecast=result['forecast']
        future=input['future']
        device=forecast.device
        return metric(forecast,future.to(device))

    @staticmethod
    def evaluate_output(input,result,metric):
        output=torch.cat((result['backcast'],result['forecast']),-1)
        ground_truth=torch.cat((input['history'],input['future']),-1)
        device=output.device
        return metric(output,ground_truth.to(device))

    @staticmethod
    def evaluate_infoNCE(positive,negative,metric):
        theta_cnn=positive['theta_cnn']
        count_pos=theta_cnn.shape[0]
        theta_pred=torch.cat([positive['theta_pred'],negative['theta_pred']],0)
        simularity=theta_cnn.mm(theta_pred.T)
        index_pos=torch.arange(theta_cnn.shape[0]).unsqueeze(1)
        index_neg=torch.arange(theta_cnn.shape[0],simularity.shape[1]).broadcast_to(theta_cnn.shape[0],-1)
        simularity=simularity.gather(dim=1,index=torch.cat([index_pos,index_neg],1).to(simularity.device))
        return metric(simularity,torch.zeros(theta_cnn.shape[0],device=simularity.device,dtype=torch.long))
    
    def record_tensorboard(self,):
        get_current=lambda column:self.logdf[column].iloc[-1]
        itrn=get_current('iteration')
        self.writer.add_scalar('train/all',get_current('train_all'),itrn)
        self.writer.add_scalar('train/back',get_current('train_back'),itrn)
        self.writer.add_scalar('train/fore',get_current('train_fore'),itrn)
        if not np.isnan(infonce:=get_current('info_NCE')):
            self.writer.add_scalar('train/infoNCE',infonce,itrn)
        self.writer.add_scalar('validate/all',get_current('valiadate_all'),itrn)
        self.writer.add_scalar('validate/back',get_current('valiadate_back'),itrn)
        self.writer.add_scalar('validate/fore',get_current('valiadate_fore'),itrn)
        
        # trainsample_x,trainsample_y=[torch.from_numpy(i) for i in self.trloader.dataset.getvisualbatch()]
        trainsample_x,trainsample_y=torch.from_numpy(self.trloader.dataset.getvisualbatch(self.samplesize)).split([7*24,1*24],dim=1)
        trainsample_result=self.inference(trainsample_x,future=None,step=1,trmode=False,gd=False)
        trainsample_b,trainsample_f=trainsample_result['backcast'].cpu(),trainsample_result['forecast'].cpu()
        for idx,x,y,b,f in zip(self.trloader.dataset.visualindices,trainsample_x,trainsample_y,trainsample_b,trainsample_f):
            self.writer.add_figure(f'train_{idx}/all',self.plotall(x,y,b,f),itrn)
            self.writer.add_figure(f'train_{idx}/fore',self.plotfore(y,f),itrn)
            self.writer.add_figure(f'train_{idx}/back',self.plotback(x,b),itrn)
            
        # valsample_x,valsample_y=[torch.from_numpy(i) for i in self.valloader.dataset.getvisualbatch()]
        valsample_x,valsample_y=torch.from_numpy(self.valloader.dataset.getvisualbatch(self.samplesize)).split([7*24,1*24],dim=1)
        valsample_result=self.inference(valsample_x,future=None,step=1,trmode=False,gd=False)
        valsample_b,valsample_f=valsample_result['backcast'].cpu(),valsample_result['forecast'].cpu()
        for idx,x,y,b,f in zip(self.valloader.dataset.visualindices,valsample_x,valsample_y,valsample_b,valsample_f):
            self.writer.add_figure(f'validate_{idx}/all',self.plotall(x,y,b,f),itrn)
            self.writer.add_figure(f'validate_{idx}/fore',self.plotfore(y,f),itrn)
            self.writer.add_figure(f'validate_{idx}/back',self.plotback(x,b),itrn)
        
        #embedding part
        px,py=torch.from_numpy(self.trloader.dataset.getvisualbatch(128)).split([7*24,1*24],dim=1)
        p_result=self.inference(px,future=None,step=1,trmode=False,gd=False)
        nx,ny=torch.from_numpy(self.negloader.dataset.getvisualbatch(128)).split([7*24,1*24],dim=1)
        n_result=self.inference(nx,future=None,step=1,trmode=False,gd=False)
        mat=torch.cat([p_result['context'],n_result['context']])
        metadata=['+']*128+['-']*128
        self.writer.add_embedding(mat=mat,metadata=metadata,global_step=itrn,tag=self.name)
        self.writer.flush()

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

class ARGS():
    def __init__(self,argv=None):
        parser=argparse.ArgumentParser()
        #dataset setting
        parser.add_argument('-ds','--dataset',type=str,default='IHEPC')
        parser.add_argument('-dp','--datapath',type=self.adddatasetprefix(parser,argv),nargs='+',default=['dataset/IHEPC/household_power_consumption.txt'])
        parser.add_argument('-dr','--date_range',type=self.nullstr_to_None(pd.Timestamp),nargs=2,default=None)
        parser.add_argument('-dc','--data_clean_threshold',type=self.nullstr_to_None(self.data_clean_thresholdtodict),default=',100')
        parser.add_argument('-cul','--cleaned_user_list',type=self.nullstr_to_None(str),default=None)
        parser.add_argument('-nm','--normalized_method',type=str,default='z',choices=['z','max',''])
        parser.add_argument('-uc','--use_cols',type=str,default='g')
        parser.add_argument('-tu','--timeunit',type=int,default=60)
        parser.add_argument('-a','--align',type=int,default=24)
        
        #model setting
        parser.add_argument('-st','--stack_types',type=self.getstacktype,default='gg')
        parser.add_argument('-nbps','--nb_blocks_per_stack',type=int,default=2)
        parser.add_argument('-bbl','--backbone_layers',type=int,default=2)
        parser.add_argument('-bbk','--backbone_kernel_size',type=self.nullstr_to_None(int),default=None) #for cnn block
        parser.add_argument('-fl','--forecast_length',type=int,default=24)
        parser.add_argument('-bl','--backcast_length',type=int,default=7*24)
        parser.add_argument('-dsf','--downsampling_factor',type=int,default=24)
        parser.add_argument('-tdim','--thetas_dim',type=self.tonumlist,default='4,4')
        parser.add_argument('-bc','--basis_constrain',type=bool,default=None)
        parser.add_argument('-swis','--share_weights_in_stack',type=bool,default=False)
        parser.add_argument('-hlu','--hidden_layer_units',type=int,default=8)
        parser.add_argument('-cs','--context_size',type=self.nullstr_to_None(int),default=None)
        parser.add_argument('-pm','--predictModule',type=self.nullstr_to_None(str),default=None) #for cnn block
        parser.add_argument('-spm','--share_predict_module',type=self.nullstr_to_None(bool),default=None) #for cnn block
        parser.add_argument('-pmnl','--predict_module_num_layers',type=self.nullstr_to_None(int),default=None) #for cnn block

        #training setting
        parser.add_argument('-n','--name',type=str,default=None)
        parser.add_argument('-expn','--expname',type=self.nullstr_to_None(str),default=None)
        parser.add_argument('-d','--cudadevice',type=int,default=0)
        parser.add_argument('-rs','--rngseed',type=int,default=None)
        parser.add_argument('-e','--epochs',type=int,default=35)
        parser.add_argument('-r','--record',type=str,default='e') #i100
        parser.add_argument('-tb','--train_batch',type=int,default=128)
        parser.add_argument('-tnb','--train_negative_batch',type=self.nullstr_to_None(int),default=None)
        parser.add_argument('-vb','--valid_batch',type=int,default=512)
        parser.add_argument('-ss','--samplesize',type=int,default=8)
        parser.add_argument('-tl','--trainlosstype',type=self.lossfunc,default='mape')
        parser.add_argument('-lr','--lossratio',type=self.tofloatlist,default='0,0,1')
        parser.add_argument('-em','--evaluatemetric',type=self.lossfunc,default='mape')
        parser.add_argument('-opt','--optimizer',type=self.optalgo,default='adam')
        
        parser.parse_args(args=argv,namespace=self)
        # print(f'use args : =============')
        # print(f'{self.args}')
        self.device=self.checkDevice(self.cudadevice)
        self.globalrng=np.random.default_rng(seed=self.rngseed)
        self.datasetprep={'IHEPC':IHEPC,
                          'TMbase':TMbase}.get(self.dataset)

        print(f'args:{vars(self)}','='*15,sep='\n')
        self.checkdir(f'exp/{self.expname}')
        with open(f'exp/{self.expname}/info_{self.name}.json', 'w', encoding='utf-8') as f:
            json.dump({i:str(j) for i,j in vars(self).items()}, f, ensure_ascii=False, indent=4)
        ...
        
    # def __getattr__(self,key):
    #     return getattr(self.args,key)

    def nullstr_to_None(func):
        def check(s):
            return None if s==''else func(s)
        return check

    @staticmethod
    def data_clean_thresholdtodict(s):
        return {i:float(j) for i,j in zip(['value','length'],s.split(',')) if j!=''}

    @classmethod
    def addtbprefix(cls,parser,argv):
        args=parser.parse_known_args(argv)[0]
        @cls.nullstr_to_None
        def func(s):
            return f'runs/{s}/{args.name}'
        return func

    @classmethod
    def addmdlprefix(cls,parser,argv):
        args=parser.parse_known_args(argv)[0]
        @cls.nullstr_to_None
        def func(s):
            prefix=f'model/{s}'
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            return f'{prefix}/{args.name}.mdl'
        return func

    @classmethod
    def addlogprefix(cls,parser,argv):
        args=parser.parse_known_args(argv)[0]
        @cls.nullstr_to_None
        def func(s):
            prefix=f'log/{s}'
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            return f'{prefix}/{args.name}.csv'
        return func

    @staticmethod
    def adddatasetprefix(parser,argv):
        args=parser.parse_known_args(argv)[0]
        def func(s):
            return f'dataset/{args.dataset}/{s}'
        return func

    @staticmethod
    def getstacktype(s):
        stacktype={'g':NBeatsNet.GENERIC_BLOCK,'s':NBeatsNet.SEASONALITY_BLOCK,
                't':NBeatsNet.TREND_BLOCK,'c':NBeatsNet.GENERIC_CNN}
        return [stacktype.get(i) for i in s]
    
    @staticmethod
    @nullstr_to_None
    def tonumlist(s):
        return [int(i) for i in s.split(',')]
    
    @staticmethod
    def tofloatlist(s):
        return [float(i) for i in s.split(',')]

    @staticmethod
    def lossfunc(s):
        loss_type,*loss_arg=s.split(':')
        typedict={'mae':nn.L1Loss,
                  'mape':funcs.MAPE,
                  'pmape':funcs.pMAPE,
                  'mse':nn.MSELoss,
                  'huber':nn.HuberLoss,}
        argdict={'reduction':str,
                 'eps':float,
                 'percentage':bool,
                 'tau':float}
        if loss_arg:
            args=loss_arg[0].split(',')
            argd={j:argdict.get(j)(k) for i in args for j,k in [i.split('=')]}
            return partial(typedict.get(loss_type),**argd)
        else:
            return typedict.get(loss_type)

    @staticmethod
    def optalgo(s):
        optdict={'adam':torch.optim.Adam,
                 'sgd':torch.optim.SGD,}
        return optdict.get(s)

    @staticmethod
    def checkDevice(cudadevice):
        # print('=============================')
        print(f'python ver. {sys.version}')
        print(f'pytorch ver. {torch.__version__}')
        print(f'cuda ver. {torch.version.cuda}')
        print(f'cuda avail : {(cuava:=torch.cuda.is_available())}')
        print(f'use device: {(dev:=torch.device(f"cuda:{cudadevice}" if cuava else "cpu"))}')
        print('='*15)
        return dev

    @staticmethod
    def checkdir(path):
        if not os.path.isdir(path):
            os.makedirs(f'{path}/run')
            os.makedirs(f'{path}/log')
            os.makedirs(f'{path}/model')

    nullstr_to_None=staticmethod(nullstr_to_None)
    
def main(datasetprep,datapath,date_range,data_clean_threshold,cleaned_user_list,normalized_method,
         use_cols,timeunit,align,forecast_length,backcast_length,downsampling_factor,
         globalrng,samplesize,train_batch,train_negative_batch,valid_batch,
         
         name,expname,device,stack_types,nb_blocks_per_stack,thetas_dim,basis_constrain,share_weights_in_stack,
         hidden_layer_units,backbone_layers,backbone_kernel_size,context_size,
         predictModule,
         share_predict_module,
         predict_module_num_layers,
         
         trainlosstype,evaluatemetric,optimizer,lossratio,
         epochs,record,**unused_argd):
    dataset=datasetprep(datapath=datapath,
                        date_range=date_range,
                        data_clean_threshold=data_clean_threshold,
                        cleaned_user_list=cleaned_user_list,
                        normalized_method=normalized_method,
                        use_cols=use_cols,
                        timeunit=timeunit,
                        align=align,
                        forecast_length=forecast_length,
                        backcast_length=backcast_length,
                        globalrng=globalrng,
                        samplesize=samplesize,
                        train_batch=train_batch,
                        train_negative_batch=train_negative_batch,
                        valid_batch=valid_batch,)
    net = NBeatsNet(name=name,
                    device=device,
                    stack_types=stack_types,
                    nb_blocks_per_stack=nb_blocks_per_stack,
                    forecast_length=forecast_length,
                    backcast_length=backcast_length,
                    downsampling_factor=downsampling_factor,
                    thetas_dim=thetas_dim,
                    basis_constrain=basis_constrain,
                    share_weights_in_stack=share_weights_in_stack,
                    hidden_layer_units=hidden_layer_units,
                    backbone_layers=backbone_layers,
                    backbone_kernel_size=backbone_kernel_size,
                    context_size=context_size,
                    predictModule=predictModule,
                    share_predict_module=share_predict_module,
                    predict_module_num_layers=predict_module_num_layers)
    exp=Trainer(name=name,expname=expname,
        model=net,
                trloader=dataset.trainloader,
                negloader=dataset.negative_loader,
                valloader=dataset.valloader,
                lossf=trainlosstype(),
                evmetric=evaluatemetric(),
                opt=optimizer(net.parameters()),
                device=device,
                lossratio=lossratio,
                samplesize=samplesize,)
    exp.train(epochs=epochs,
              record=record,
              )
    ...

def main_cmdargv(argv):
    args=ARGS(argv)
    main(**vars(args))
    ...

def make_argv():
    device=0
    cond1=TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list4.json'),floors=[4,11,15,18])
    return [['--dataset', 'TMbase',
             "--datapath","data_200501_211031.csv","data_2111.csv","data_2112.csv",
                          "data_2201.csv","data_2202.csv","data_2203.csv","data_2204.csv",
             "--date_range", "2020-6-1","",
             "--data_clean_threshold", "0.01,600",
             "--cleaned_user_list","dataset/TMbase/use_list4.json",
             "--timeunit","1",
             "--align","24",
             "--use_cols",i,
             "--normalized_method","",

             "--stack_types", "c",
             "--nb_blocks_per_stack", "1",
             "--backbone_layers", "2",
             "--backbone_kernel_size", "3",
             "--hidden_layer_units", "8",
             "--thetas_dim", "6",
             "--predictModule","lstm",
             "--context_size","12",

             "--name",i,
             "--expname","B1_L2_K3_U8_T6_C12_align24_a00_ep200",
             "--epochs", "200",
             "--cudadevice", f"{0}",
             "--rngseed", "6666",
             "--trainlosstype","mape",
             "--lossratio","0,0,1,0",
             "--evaluatemetric","mape",
             "--train_negative_batch","256",
             ] for i in cond1[device::2]]
            #  ] for i in ['N16-F04-A01']]

if __name__=='__main__':
    settings=[None]

    for s in settings:
        main_cmdargv(s)
        ...
    ...