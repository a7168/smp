'''
Author: Egoist
Date: 2022-03-07 13:22:43
LastEditors: Egoist
LastEditTime: 2022-04-13 01:23:26
FilePath: /smp/detect.py
Description: 

'''

import numpy as np
import pandas as pd
import torch
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

    def detect(self,model_NFA,data_NFA,date,nstart=None,nend=None,simple=True,plot_idf=''):
        if self.net is None or self.net.name!=model_NFA:
            self.net=NBeatsNet.build(f'{self.modelpath_prefix}/{model_NFA}.mdl',new_name=model_NFA,new_device=self.device)
        date=date if isinstance(date,pd.Timestamp) else pd.Timestamp(date)

        target=self.set.getitembydate(date,length=1,nstart=nstart,nend=nend)[data_NFA].to_numpy(dtype=np.float32)
        data_rangestart=[date+pd.Timedelta(-7+i,unit='d') for i in range(8)]
        raw_inputs=[self.set.getitembydate(i,length=7,nstart=nstart,nend=nend)[data_NFA].to_numpy(dtype=np.float32) for i in data_rangestart]
        x=torch.from_numpy(np.stack(raw_inputs))
        inference=self.net.inference(x,future=None,step=1,trmode=False,gd=False)
        o=torch.cat([inference['backcast'],inference['forecast']],-1) # concat backcast and forecast
        results=[o[i,-24*(i+1):-24*i if i!=0 else None].cpu() for i in range(len(o))]
        all=np.stack([target]+results)
        result_mean=np.stack(results[1:]).mean(axis=0)

        mae=torch.nn.L1Loss(reduction='none')
        mape=funcs.MAPE(reduction='none')
        metric=mape
        results_err=metric(torch.from_numpy(all),torch.from_numpy(target).broadcast_to(all.shape)).mean(dim=-1)
        result_mean_err=metric(torch.from_numpy(result_mean),torch.from_numpy(target))
        result_mean_stat={'mean':result_mean_err.mean(),'std':result_mean_err.std()}
        

        info={'model':self.net.name,'data':data_NFA,'date':f'{date.year}-{date.month}-{date.day}'}
        # title='\n'.join([f'model: {self.net.name}',f'data: {data_NFA}',f'date: {date.year}-{date.month}-{date.day}'])
        title='\n'.join([f'{i}: {j}' for i,j in info.items()])
        if simple:
            labels=['target',f'reconstruct (mean:{result_mean_stat["mean"]:.4f} | std:{result_mean_stat["std"]:.4f})']
            fig=self.plot(np.stack([target,result_mean]),labels=labels,title=title)
        else:
            labels=['target']+[self.rangestr(i,7,e) for i,e in zip(data_rangestart,results_err[1:])]
            fig=self.plot(all,labels=labels,title=title)
        if self.tbwriter is not None:
            self.tbwriter.add_figure(f'data: {data_NFA}{plot_idf}/model: {info["model"]}',fig,(date-self.set.start).days)
            self.tbwriter.flush()
        else:
            plt.show(block=False)


        return {'fig':fig,'target':target,'reconstruct':result_mean}

    def detect_month(self,model_NFA,data_NFA,month,nstart=None,nend=None,plot_idf=''):
        month_first=pd.Timestamp(month)
        days=month_first.days_in_month
        everyday_target=[]
        everyday_reconstruct=[]
        for d in range(1,days+1):
            day=self.detect(model_NFA=model_NFA,data_NFA=data_NFA,date=f'{month}-{d}',
                            nstart=nstart,nend=nend,plot_idf=plot_idf,simple=True)
            everyday_target.append(day['target'])
            everyday_reconstruct.append(day['reconstruct'])
        target=np.concatenate(everyday_target)
        reconstruct=np.concatenate(everyday_reconstruct)

        mae=torch.nn.L1Loss(reduction='none')
        mape=funcs.MAPE(reduction='none')
        metric=mape
        err=metric(torch.from_numpy(reconstruct),torch.from_numpy(target))
        stat={'mean':err.mean(),'std':err.std()}

        
        info={'model':self.net.name,'data':data_NFA,'month':f'{month_first.year}-{month_first.month}'}
        title='\n'.join([f'{i}: {j}' for i,j in info.items()])
        labels=['target',f'reconstruct (mean:{stat["mean"]:.4f} | std:{stat["std"]:.4f})']
        fig=self.plot(np.stack([target,reconstruct]),labels=labels,title=title,xticklabel_start=month_first,err=err)
        if self.tbwriter is not None:
            self.tbwriter.add_figure(f'data: {data_NFA} on {info["month"]}{plot_idf}/model: {info["model"]}',fig,(month_first-self.set.start).days)
            self.tbwriter.flush()
        else:
            plt.show(block=False)
        ...



    @staticmethod
    def plot(output,labels,title,xticklabel_start=None,err=None):
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
            ax2.plot(range(len(err)),err,alpha=0.5,label='MAE',color='r')
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

if __name__=='__main__':
    det=Detector(modelpath_prefix='model/B3_L4_K3_U8_T4_C8_pmape_2',
                 datasetpath=['dataset/TMbase/data_200501_211031.csv',
                              'dataset/TMbase/data_2111.csv',
                              'dataset/TMbase/data_2112.csv',
                              'dataset/TMbase/data_2201.csv',
                              'dataset/TMbase/data_2202.csv',
                              'dataset/TMbase/data_2203.csv',],
                 device=torch.device('cpu'),
                 tbwriter='pmape_exp2')

    userlist=TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list3.json'),floors=[4,11,15,18])
    userdict={}
    for data in userlist:
        floor=data.split('-')[1]
        compare_list=[i for i in userlist if floor in i] #compare same floor
        userdict[data]=compare_list

    for data,models in userdict.items():
        for m in models:
            det.detect_month(model_NFA=m,data_NFA=data,month=f'{2021}-{3}',nend=None)
            det.detect_month(model_NFA=m,data_NFA=data,month=f'{2021}-{6}',nend=None)
            det.detect_month(model_NFA=m,data_NFA=data,month=f'{2021}-{9}',nend=None)
            det.detect_month(model_NFA=m,data_NFA=data,month=f'{2021}-{12}',nend=None)
            det.detect_month(model_NFA=m,data_NFA=data,month=f'{2022}-{1}',nend=None)
            det.detect_month(model_NFA=m,data_NFA=data,month=f'{2022}-{2}',nend=None)

    # det.detect_month(model_NFA='N16-F04-A07',data_NFA='N16-F04-A07',month=f'{2020}-{12}',nend=None)
    # det.detect_month(model_NFA='N16-F04-A07',data_NFA='N16-F04-A07',month=f'{2021}-{12}',nend=None)


    # for model in userlist:
    #     for data in userlist:
    #         for i in range(1,8):
    #             det.detect_month(model_NFA=model,data_NFA=data,date=f'{2022}-{1}-{7}')

        # f1=det.detect(model_NFA='N16-F09-A00',data_NFA='N16-F04-A00',date=(2022,1,i))
        # f2=det.detect(model_NFA='N16-F09-A00',data_NFA='N16-F09-A00',date=(2022,1,i))
        # f3=det.detect(model_NFA='N16-F04-A03',data_NFA='N16-F04-A00',date=(2022,1,i)) 21/1/4~21/3/5

    
    ...