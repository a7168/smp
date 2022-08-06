'''
Author: Egoist
Date: 2022-06-10 15:43:23
LastEditors: Egoist
LastEditTime: 2022-08-06 02:44:06
FilePath: /smp/main.py
Description: 
    run different task in this script
'''
#%%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from readTMbase import TMbaseset
from nbeatmodel import NBeatsNet
from train import main_cmdargv
from detect import Detector
import api

DATAPATH=['dataset/TMbase/data_200501_211031.csv',
          'dataset/TMbase/data_2111.csv',
          'dataset/TMbase/data_2112.csv',
          'dataset/TMbase/data_2201.csv',
          'dataset/TMbase/data_2202.csv',
          'dataset/TMbase/data_2203.csv',
          'dataset/TMbase/data_2204.csv',]


def filter_userlist(floors=[4,11,15,18]):
    return TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list4.json'),floors=floors)

def detect_user_period(expname,
                       user,
                       date=f'2020-5-1',
                       days=730,
                       threshold=0.5,
                       showerr=True):
    det=Detector(expname=expname,
                 datasetpath=DATAPATH,
                 device=torch.device('cpu'),
                 tbwriter=None)
    det.detect_day(model_NFA=user,data_NFA=user,date=date,days=days,metric='mape',threshold=threshold,showerr=showerr)
    ...

def detect_compute_ratio(expname,
                         date=f'2020-5-1',
                         days=730,
                         output_place=None,
                         thresholds=(0.5,0.6,0.7),
                         floors=[4,11,15,18]):
    det=Detector(expname=expname,
                 datasetpath=DATAPATH,
                 device=torch.device('cpu'),
                 tbwriter=output_place)
    userlist=filter_userlist(floors=floors)
    result={}#detect self data with different threshold
    for user in userlist:
        for T in thresholds:
            result[f'{user} T={T}']=det.detect_day(model_NFA=user,data_NFA=user,date=date,days=days,metric='mape',threshold=T)
            ...
    for i,(k,v) in enumerate(result.items()):
        print(k,v)
        if i%3==2:
            print('='*10)
    ...

def detect_apply_on_other_data(expname,
                               date=f'2020-5-1',
                               days=730,
                               output_place=None,
                               floors=[4,11,15,18]):
    det=Detector(expname=expname,
                 datasetpath=DATAPATH,
                 device=torch.device('cpu'),
                 tbwriter=output_place)
    userlist=filter_userlist(floors=floors)
    userdict={}
    for data in userlist:
        floor=data.split('-')[1]
        compare_list=[i for i in userlist if floor in i] #compare same floor
        userdict[data]=compare_list
    for m,datas in userdict.items():
        for d in datas:
            det.detect_day(model_NFA=m,data_NFA=d,date=date,days=days,metric='mape',threshold=0.5)
            # det.detect_day(model_NFA=m,data_NFA=data,date=f'2020-5-1',days=610,metric='mape',threshold=0.5)
            # det.detect_day(model_NFA=m,data_NFA=data,date=f'2022-1-1',days=90,metric='mape',threshold=0.5)
            ...
    ...
    ...

def train_model():
    def make_argv(): #return different NFA's traing arguments
        device=1 #GPU index on training
        torch.cuda.set_device(device)
        cond1=filter_userlist(floors=[4,11,15,18])
        return [["--dataset", "TMbase",
                 "--datapath","data_200501_211031.csv","data_2111.csv","data_2112.csv",
                              "data_2201.csv","data_2202.csv","data_2203.csv","data_2204.csv",
                 "--date_range", "2020-6-1","",
                 "--data_clean_threshold", "0.01,600",
                 "--cleaned_user_list","dataset/TMbase/use_list4.json",
                 "--timeunit","1",
                 "--align","24",
                 "--use_cols",NFA,
                 "--normalized_method","",

                 "--stack_types", "c",
                 "--nb_blocks_per_stack", "10",
                 "--backbone_layers", "2",
                 "--backbone_kernel_size", "3",
                 "--hidden_layer_units", "8",
                 "--thetas_dim", "3",
                 "--basis_constrain","",
                 "--predictModule","lstm",
                 "--context_size","6",

                 "--name",NFA,
                 "--expname","ablation_block10_a00_woc",
                 "--epochs", "200",
                 "--cudadevice", f"{device}",
                 "--rngseed", "6666",
                 "--trainlosstype","mape",
                 "--lossratio","0,0,1,0",
                 "--evaluatemetric","mape",
                 "--train_batch","32",
                 "--train_negative_batch","256",
                 "--context_visualization","",
                 "--samplesize","2",
                ] for NFA in cond1[device::2]]
                #  ] for NFA in ['N16-F04-A01']]
        
    for s in make_argv():
        main_cmdargv(s)
    ...

def plot_disentangle_schematic():
    a,b,c=np.random.rand(3,24)
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    x=np.arange(24)
    ax.plot(x,a,0,zdir='y')
    ax.plot(x,b,1,zdir='y')
    ax.plot(x,c,2,zdir='y')
    ax.plot(x,(a+b+c),3,zdir='y')
    ax.set_yticks([0,1,2,3],['x1','x2','x3','X'])
    plt.show()
    ...

def plot_user_data(user='N16-F04-A01',days=8,start_date=None,figsize=(30,10)):
    TMbaseset(datapath=DATAPATH).plot_user(user,days,start_date,figsize)

def plot_model_basis(expname='B2_L2_K3_U8_T3_C6_align24_a05_ep200',
                     NFA='N16-F04-A01',
                     device=torch.device('cpu')):
    model=NBeatsNet.build(f'exp/{expname}/model/{NFA}.mdl',new_name=f'{expname}/{NFA}',new_device=device)
    fig=model.plot_basis()
    plt.show(block=False)
    ...

def plot_model_result(expname,
                      NFA,
                      date=f'2020-5-1',
                      days=1,
                      device=torch.device('cpu')):
    model=NBeatsNet.build(f'exp/{expname}/model/{NFA}.mdl',new_name=NFA,new_device=device)
    set=TMbaseset(DATAPATH)
    date=date if isinstance(date,pd.Timestamp) else pd.Timestamp(date)
    target=set.getitembydate(date,length=days)[NFA].to_numpy(dtype=np.float32)
    x=torch.from_numpy(target).unsqueeze(0)
    inference=model.inference(x,future=None,step=1,trmode=False,gd=False)
    fig, ax1 = plt.subplots()
    ax1.plot(inference['backcast'].squeeze(),color='C1')
    plt.show(block=False)

def access_api_data(startdate=(2022,5,1),
                    enddate=(2022,5,1),
                    outdir='tempdata'):
    '''need to open proxy server before run the code
       1.ssh connect to server
       2.run openproxyserver.bat
       3.run this code'''
    api.main(startdate=startdate,
             enddate=enddate,
             outdir=outdir)


if __name__=='__main__':
    '''run the task
       adjust argument in above function if needed'''

    exp=['ablation_block2',
         'ablation_block2_a00'
         'ablation_block2_a00_woc',
         'ablation_block3',
         'ablation_block3_a00'
         'ablation_block3_a00_woc',
         'ablation_block4',
         'ablation_block4_a00'
         'ablation_block4_a00_woc',]

    access_api_data(startdate=(2022,7,1),
                    enddate=(2022,7,31))

    # plot_disentangle_schematic()
    
    # [plot_user_data(user='N17-F04-A13',days=1,start_date=f'2021-5-{i}',figsize=None) for i in range(1,10)]

    # plot_model_basis(expname='ablation_block4_a00_woc',
    #                  NFA='N16-F04-A09')#NFA='N19-F11-A01'

    
    # train_model()
    
    # detect_compute_ratio(expname='ablation_basis_wo_constrain',output_place=None)

    # detect_apply_on_other_data(expname='ablation_block4bat32',output_place='otherdata_ablation_basis4')

    # plot_model_result(expname='ablation_block4bat32_a00',
    #                   NFA='N16-F04-A09',
    #                   date=f'2021-5-1',
    #                   days=1,)

    # detect_user_period(expname='ablation_block2_a00_woc',
    #                    user='N16-F04-A09',
    #                    date=f'2021-5-1',
    #                    days=7,
    #                    threshold=0.5)
    ...