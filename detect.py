'''
Author: Egoist
Date: 2022-03-07 13:22:43
LastEditors: Egoist
LastEditTime: 2022-07-18 08:38:28
FilePath: /smp/detect.py
Description: 

'''
# %%
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from nbeatmodel import NBeatsNet
from readTMbase import TMbaseset
import funcs

class Detector():
    def __init__(self,modelpath_prefix,datasetpath,device=None,tbwriter=None):
        self.modelpath_prefix=modelpath_prefix
        self.net=None
        self.set=TMbaseset(datasetpath)
        self.device=device
        self.tbwriter=None if tbwriter is None else SummaryWriter(f'runs/detect/{tbwriter}')
        ...
    @staticmethod
    def detect2(model,data,metric):# use in detct_month_mean   use month_mean_data not origin
        result=model.inference(data,future=None,step=1,trmode=False,gd=False)
        metric={'mae':torch.nn.L1Loss(reduction='none'),
                'mape':funcs.MAPE(reduction='none')
                }.get(metric)
        error=metric(result['backcast'],data)
        return {'reconstruct':result['backcast'],
                'error':error,
                'mean':error.mean(),
                'std':error.std()}

    def detct_month_mean(self,model_NFA,data_NFA,month,metric):
        if self.net is None or self.net.name!=model_NFA: #load model if need
            self.net=NBeatsNet.build(f'{self.modelpath_prefix}/{model_NFA}.mdl',new_name=model_NFA,new_device=self.device)
            fig=self.net.plot_basis()
            if self.tbwriter is not None:
                self.tbwriter.add_figure(f'basis/{model_NFA}',fig)
                self.tbwriter.flush()
            else:
                plt.show(block=False)
        data=self.set.get_month_mean(month,data_NFA).unsqueeze(0)
        result=self.detect2(self.net,data,metric)

        info={'model':self.net.name,'data':data_NFA,'date':f'{month}'}
        title='\n'.join([f'{i}: {j}' for i,j in info.items()])

        labels=['target',f'reconstruct (mean:{result["mean"]:.4f} , std:{result["std"]:.4f})']
        lines=torch.cat([data,result['reconstruct']])
        fig=self.plot(lines,labels=labels,title=title,err=result['error'].squeeze(),metric=metric)
        if self.tbwriter is not None:
            self.tbwriter.add_figure(f'{model_NFA} {month}/data: {data_NFA}',fig)
            self.tbwriter.flush()
        else:
            plt.show(block=False)

    def detect(self,model_NFA,data_NFA,date,days=1,shift=0,metric='mape',):
        if self.net is None or self.net.name!=model_NFA: #load model if need
            self.net=NBeatsNet.build(f'{self.modelpath_prefix}/{model_NFA}.mdl',new_name=model_NFA,new_device=self.device)
        
        date=date if isinstance(date,pd.Timestamp) else pd.Timestamp(date)
        target=self.set.getitembydate(date,length=days)[data_NFA].to_numpy(dtype=np.float32)
        x=torch.from_numpy(target).unsqueeze(0)
        x=x if shift==0 else x.roll(shift,-1)
        inference=self.net.inference(x,future=None,step=1,trmode=False,gd=False)

        metric={'mae':torch.nn.L1Loss(reduction='none'),
                'mape':funcs.MAPE(reduction='none')
                }.get(metric)
        results_err=metric(inference['backcast'],x)
        return {'target':x,
                'reconstruct':inference['backcast'],
                'error':results_err,
                'mean':results_err.mean(),
                'std':results_err.std()}
        
    def detect_day(self,model_NFA,data_NFA,date,days=1,shift=0,metric='mape',threshold=None):
        date=date if isinstance(date,pd.Timestamp) else pd.Timestamp(date)
        end_date=date+pd.Timedelta(days-1,unit='d')
        end_date_str='' if days==1 else f'to {end_date.year}-{end_date.month}-{end_date.day}'

        result_stat=self.detect(model_NFA,data_NFA,date,days=days,shift=shift,metric=metric)
        if threshold is None:
            rate={}
        else:
            err=result_stat['error']
            rate={f'abnormal rate(T={threshold})':(err>threshold).count_nonzero()/err.numel()}
        
        info={'model':self.net.name,
              'data':data_NFA,
              'date':f'{date.year}-{date.month}-{date.day} {end_date_str}',}|rate
            #   f'abnormal rate(T={threshold})':(err>threshold).count_nonzero()/err.numel()}
        # title='\n'.join([f'model: {self.net.name}',f'data: {data_NFA}',f'date: {date.year}-{date.month}-{date.day}'])
        title='\n'.join([f'{i}: {j}' for i,j in info.items()])

        labels=['target',f'reconstruct (mean:{result_stat["mean"]:.4f} , std:{result_stat["std"]:.4f})']
        lines=torch.cat([result_stat['target'],result_stat['reconstruct']])
        xticklabel=None if days<7 else date
        fig=self.plot(lines,labels=labels,title=title,err=result_stat['error'].squeeze(),metric=metric,xticklabel_start=xticklabel)

        if self.tbwriter is not None:
            # self.tbwriter.add_figure(f'data: {data_NFA} {info["date"]}/model: {info["model"]}',fig,(date-self.set.start).days)
            self.tbwriter.add_figure(f'model: {info["model"]}/data: {data_NFA} {info["date"]}',fig,(date-self.set.start).days)
            self.tbwriter.flush()
        else:
            plt.show(block=False)
        # self.set.plot_user('N16-F04-A01',days=8)
        if threshold is not None:
            return rate[f'abnormal rate(T={threshold})']

    @staticmethod
    def get_anormaly_interval(error,window,threshold_value,threshold_window):
        err_3c=error.unsqueeze(1) #convert to 3 channel batch,channel,time
        result_bool=torch.ones_like(err_3c).where(err_3c>=threshold_value,torch.zeros_like(err_3c))
        count_window=F.conv1d(result_bool,torch.ones(1,1,window)).flatten()
        index_interval=(count_window>=threshold_window).nonzero()


    @staticmethod
    def plot(output,labels,title,xticklabel_start=None,err=None,metric='MAPE'):
        data_size=output.shape[-1]
        fig, ax1 = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)
        fig.suptitle(title)
        for o,l in zip(output,labels):
            ax1.plot(range(len(o)),o,label=l)
        ax1.set_ylabel('kwh')
        ax1.legend()
        if err is not None:
            ax2=ax1.twinx()
            ax2.plot(range(len(err)),err,alpha=1,label=metric,color='r')
            yt=ax2.get_yticks()
            # yt_range=yt[-1]-yt[0]
            # yt_new=[yt[-1]*(1-0.5*i) for i in range(7)][::-1]
            step=4
            yt_new=[-2*yt[-1]]+[yt[-1]*i/step for i in range(step+1)]
            yt_new_label=[f'{i:.1f}' if i>=0 else '' for i in yt_new]
            ax2.set_yticks(yt_new,labels=yt_new_label)
            ax2.set_ylabel('err', color='r')  
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend()

        if xticklabel_start is not None:
            ticks=ax1.get_xticks()
            tick_labels=[]
            for i in ticks:
                t=xticklabel_start+pd.Timedelta(i,unit='h')
                tick_labels.append(f'{t.year%100}-{t.month}-{t.day}')
            ax1.set_xticklabels(tick_labels)
        fig.tight_layout()
        # plt.pause(1)
        # plt.show(block=False)
        return fig

    @staticmethod
    def rangestr(date,length,err):
        end=date+pd.Timedelta(length-1,unit='d')
        return f'{date.year%100}.{date.month}.{date.day} ~ {end.year%100}.{end.month}.{end.day} @ {err:.4f}'


def face_north(NFA,total_list):
    floor=(4,9,14,19,21)
    north=(['N16-F{i:02}-A09' for i in floor]
          +['N17-F{i:02}-A13' for i in floor]
          +['N19-F{i:02}-A09' for i in floor])
    north=[i for i in north if i in total_list]
    return NFA in north

def task_detect_period():
    ...


def task_find_profile(model_NFA):
    det=Detector(modelpath_prefix='exp/B3_L2_K3_U8_T3_C3_align24/model',
                 datasetpath=['dataset/TMbase/data_200501_211031.csv',
                              'dataset/TMbase/data_2111.csv',
                              'dataset/TMbase/data_2112.csv',
                              'dataset/TMbase/data_2201.csv',
                              'dataset/TMbase/data_2202.csv',
                              'dataset/TMbase/data_2203.csv',
                              'dataset/TMbase/data_2204.csv',],
                 device=torch.device('cpu'),
                 tbwriter=f'profile2_B3_L2_K3_U8_T3_C3_align24/{model_NFA}')#f'profile_B2_L2_K3_U8_T3_C3_align24/{model_NFA}'
    userlist=TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list4.json'),floors=[4,11,15,18])
    for data_NFA in userlist:
        for month in ('2022-4','2022-3','2022-2','2022-1'):
            det.detct_month_mean(model_NFA=model_NFA,data_NFA=data_NFA,month=month,metric='mape')

if __name__=='__main__':#TODO axvspan 209-
    # userlist=TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list4.json'),floors=[4,11,15,18])
    # for i in userlist:
    #     print(f'start {i}')
    #     task_find_profile(i)
    ...
    det=Detector(#modelpath_prefix='exp/B3_L2_K3_U8_T3_C3_align24/model',
                 modelpath_prefix='exp/B2_L2_K3_U8_T3_C6_align24_a05_ep200/model',
                #  modelpath_prefix='exp/B1_L2_K3_U8_T6_C12_align24_a00_ep200/model',
                 datasetpath=['dataset/TMbase/data_200501_211031.csv',
                              'dataset/TMbase/data_2111.csv',
                              'dataset/TMbase/data_2112.csv',
                              'dataset/TMbase/data_2201.csv',
                              'dataset/TMbase/data_2202.csv',
                              'dataset/TMbase/data_2203.csv',
                              'dataset/TMbase/data_2204.csv',],
                 device=torch.device('cpu'),
                 tbwriter=None)

    userlist=TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list4.json'),floors=[4,11,15,18])
    userdict={}

    # temp_model=NBeatsNet.build(f'exp/B3_L2_K3_U8_T3_C3_align24/model/N16-F04-A01.mdl',new_name='N16-F04-A01',new_device=torch.device('cpu'))
    for data in userlist:
        floor=data.split('-')[1]
        compare_list=[i for i in userlist if floor in i] #compare same floor
        userdict[data]=compare_list

    result={}#detect self data with different threshold
    for user in userlist:
        for T in (0.5,0.6,0.7):
            result[f'{user} T={T}']=det.detect_day(model_NFA=user,data_NFA=user,date=f'2020-5-1',days=730,metric='mape',threshold=T)
            ...
    ...

    # for data,models in userdict.items():
    #     for m in models:
    #         det.detect_day(model_NFA=m,data_NFA=data,date=f'2020-5-1',days=730,metric='mape',threshold=0.5)
    #         # det.detect_day(model_NFA=m,data_NFA=data,date=f'2020-5-1',days=610,metric='mape',threshold=0.5)
    #         # det.detect_day(model_NFA=m,data_NFA=data,date=f'2022-1-1',days=90,metric='mape',threshold=0.5)
    #         ...
    # ...
