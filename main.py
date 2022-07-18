'''
Author: Egoist
Date: 2022-06-10 15:43:23
LastEditors: Egoist
LastEditTime: 2022-07-18 09:52:01
FilePath: /smp/main.py
Description: 
    run different task in this script
'''
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

from readTMbase import TMbaseset
from nbeatmodel import NBeatsNet
from train import main_cmdargv
from detect import Detector
import api

def load_TMdataset():
    return TMbaseset(datapath=['dataset/TMbase/data_200501_211031.csv',
                               'dataset/TMbase/data_2111.csv',
                               'dataset/TMbase/data_2112.csv',
                               'dataset/TMbase/data_2201.csv',
                               'dataset/TMbase/data_2202.csv',
                               'dataset/TMbase/data_2203.csv',
                               'dataset/TMbase/data_2204.csv',],)

def filter_userlist(floors=[4,11,15,18]):
    return TMbaseset.filter_use_list(TMbaseset.load_use_list('dataset/TMbase/use_list4.json'),floors=floors)

def detect_compute_ratio(expname,
                         date=f'2020-5-1',
                         days=730,
                         output_place=None,
                         thresholds=(0.5,0.6,0.7),
                         floors=[4,11,15,18]):
    det=Detector(modelpath_prefix=f'exp/{expname}/model',
                 datasetpath=['dataset/TMbase/data_200501_211031.csv',
                              'dataset/TMbase/data_2111.csv',
                              'dataset/TMbase/data_2112.csv',
                              'dataset/TMbase/data_2201.csv',
                              'dataset/TMbase/data_2202.csv',
                              'dataset/TMbase/data_2203.csv',
                              'dataset/TMbase/data_2204.csv',],
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
    det=Detector(modelpath_prefix=f'exp/{expname}/model',
                 datasetpath=['dataset/TMbase/data_200501_211031.csv',
                              'dataset/TMbase/data_2111.csv',
                              'dataset/TMbase/data_2112.csv',
                              'dataset/TMbase/data_2201.csv',
                              'dataset/TMbase/data_2202.csv',
                              'dataset/TMbase/data_2203.csv',
                              'dataset/TMbase/data_2204.csv',],
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
                 "--nb_blocks_per_stack", "2",
                 "--backbone_layers", "2",
                 "--backbone_kernel_size", "3",
                 "--hidden_layer_units", "8",
                 "--thetas_dim", "3",
                 "--basis_constrain","",
                 "--predictModule","lstm",
                 "--context_size","6",

                 "--name",NFA,
                 "--expname","ablation_basis_wo_constrain",
                 "--epochs", "200",
                 "--cudadevice", f"{device}",
                 "--rngseed", "6666",
                 "--trainlosstype","mape",
                 "--lossratio","0,0,1,0.5",
                 "--evaluatemetric","mape",
                 "--train_batch","128",
                 "--train_negative_batch","256",
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

def plot_user_data(user='N16-F04-A01',days=8,start_date=None):
    load_TMdataset().plot_user(user,days,start_date)

def plot_model_basis(expname='B2_L2_K3_U8_T3_C6_align24_a05_ep200',
                     NFA='N16-F04-A01',
                     device=torch.device('cpu')):
    model=NBeatsNet.build(f'exp/{expname}/model/{NFA}.mdl',new_name=NFA,new_device=device)
    fig=model.plot_basis()
    plt.show(block=False)
    ...

def access_api_data(startdate=(2022,5,1),
                    enddate=(2022,5,1),
                    outdir='tempdata'):
    api.main(startdate=startdate,
             enddate=enddate,
             outdir=outdir)


if __name__=='__main__':
    '''run the task
       adjust argument in above function if needed'''

    # access_api_data(startdate=(2022,5,1),
    #                 enddate=(2022,5,1))

    # plot_disentangle_schematic()

    # plot_user_data()

    # plot_model_basis(expname='ablation_basis_wo_constrain',
    #                  NFA='N16-F04-A09')

    # train_model()
    
    detect_compute_ratio(expname='ablation_basis_wo_constrain',output_place=None)
    # detect_apply_on_other_data(expname='ablation_basis_wo_constrain',output_place='otherdata_ablation_basis_wo_constrain')
    ...