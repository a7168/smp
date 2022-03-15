'''
Author: Egoist
Date: 2022-03-07 13:22:43
LastEditors: Egoist
LastEditTime: 2022-03-14 15:44:00
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

class Detector():
    def __init__(self,modelpath_prefix,datasetpath,device=None,tbwriter=None):
        self.modelpath_prefix=modelpath_prefix
        self.net=None
        self.set=TMbaseset(datasetpath,'g')
        self.device=device
        self.tbwriter=None if tbwriter is None else SummaryWriter(f'runs/detect/{tbwriter}')
        ...

    def detect(self,model_NFA,data_NFA,date):
        if self.net is None or self.net.name!=model_NFA:
            self.net=NBeatsNet.build(f'{self.modelpath_prefix}/{model_NFA}.mdl',new_name=model_NFA,new_device=self.device)
        date=date if isinstance(date,pd.Timestamp) else pd.Timestamp(*date)

        target=self.set.getitembydate(date,1)[data_NFA].to_numpy(dtype=np.float32)
        data_rangestart=[date+pd.Timedelta(-7+i,unit='d') for i in range(8)]
        raw_inputs=[self.set.getitembydate(i,7)[data_NFA].to_numpy(dtype=np.float32) for i in data_rangestart]
        x=torch.from_numpy(np.stack(raw_inputs))
        o=torch.cat(self.net.inference( x,trmode=False,gd=False),-1) # concat backcast and forecast
        results=[o[i,-24*(i+1):-24*i if i!=0 else None].cpu() for i in range(len(o))]
        all=np.stack([target]+results)

        mae=torch.nn.L1Loss(reduction='none')
        err=mae(torch.from_numpy(all),torch.from_numpy(target).broadcast_to(all.shape)).mean(dim=-1)

        info={'model':self.net.name,'data':data_NFA,'date':f'{date.year}-{date.month}-{date.day}'}
        # title='\n'.join([f'model: {self.net.name}',f'data: {data_NFA}',f'date: {date.year}-{date.month}-{date.day}'])
        title='\n'.join([f'{i}: {j}' for i,j in info.items()])
        labels=['target']+[self.rangestr(i,7,e) for i,e in zip(data_rangestart,err[1:])]
        fig=self.plot(all,labels=labels,title=title)
        if self.tbwriter is not None:
            self.tbwriter.add_figure(f'data: {data_NFA}/model: {info["model"]}',fig,(date-self.set.start).days)
            self.tbwriter.flush()
        else:
            plt.show(block=False)


        return fig

    @staticmethod
    def plot(output,labels,title):
        fig, ax1 = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)
        fig.suptitle(title)
        for o,l in zip(output,labels):
            ax1.plot(range(24),o,label=l)
        ax1.legend()
        fig.tight_layout()
        # plt.pause(1)
        # plt.show(block=False)
        return fig

    @staticmethod
    def rangestr(date,length,err):
        end=date+pd.Timedelta(length-1,unit='d')
        return f'{date.year%100}.{date.month}.{date.day} ~ {end.year%100}.{end.month}.{end.day} @ {err:.4f}'

if __name__=='__main__':
    det=Detector(modelpath_prefix='model/C2L2K13T4B2U8',
                 datasetpath=['dataset/TMbase/data_200501_211031.csv',
                        'dataset/TMbase/data_2111.csv',
                        'dataset/TMbase/data_2112.csv',
                        'dataset/TMbase/data_2201.csv',
                        'dataset/TMbase/data_2202.csv',],
                 device=torch.device('cpu'),
                 tbwriter='NS0314_2')


    userlist=['N16-F04-A01','N16-F04-A09',
              'N16-F09-A01','N16-F09-A09',
              'N16-F14-A01','N16-F14-A09',
              'N16-F19-A01','N16-F19-A09',
              'N16-F21-A01','N16-F21-A09',
                #---------------------------
              'N17-F04-A00','N17-F04-A06',
              'N17-F09-A00',
              'N17-F14-A00','N17-F14-A06',
                            'N17-F19-A06',
              'N17-F21-A00',
                #---------------------------
              'N19-F04-A01','N19-F04-A09',
                            'N19-F09-A09',
              'N19-F14-A01','N19-F14-A09',
                            
                            'N19-F21-A09',]

    for model in userlist:
        for data in userlist:
            for i in range(1,8):
                det.detect(model_NFA=model,data_NFA=data,date=(2022,1,i))

        # f1=det.detect(model_NFA='N16-F09-A00',data_NFA='N16-F04-A00',date=(2022,1,i))
        # f2=det.detect(model_NFA='N16-F09-A00',data_NFA='N16-F09-A00',date=(2022,1,i))
        # f3=det.detect(model_NFA='N16-F04-A03',data_NFA='N16-F04-A00',date=(2022,1,i))

    
    ...