'''
Author: Egoist
Date: 2022-05-02 14:32:20
LastEditors: Egoist
LastEditTime: 2022-05-03 08:01:01
FilePath: /smp/discriminate.py
Description: 

'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nbeatmodel import NBeatsNet
from readTMbase import TMbaseset
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self,context_dim,hidden_dim):
        super().__init__()
        layers=[nn.Flatten(),
                nn.Linear(context_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,1),
                nn.Sigmoid()]
        self.fc=nn.Sequential(*layers)

    def forward(self,x):
        return self.fc(x)



if __name__=='__main__':
    device=torch.device('cuda')
    modelpath_prefix='exp/B3_L2_K3_U8_T3_C3_align24/model/'
    model_NFA='N16-F04-A01'
    model=NBeatsNet.build(f'{modelpath_prefix}/{model_NFA}.mdl',new_device=device)

    rawdata=TMbaseset(datapath=['dataset/TMbase/data_200501_211031.csv',
                                'dataset/TMbase/data_2111.csv',
                                'dataset/TMbase/data_2112.csv',
                                'dataset/TMbase/data_2201.csv',
                                'dataset/TMbase/data_2202.csv',
                                'dataset/TMbase/data_2203.csv',
                                ])
    rawdata.parse(date_range=None,
                  align=24,
                  threshold={'value':0.01,'length':600},
                  cleaned_user_list='dataset/TMbase/use_list3.json',
                  normalize='',
                  use_cols=model_NFA,
                  seq_length=1*(168+24),)
    rawdata.setbackfore(7*24)
    trainset,validateset=rawdata.splitbyratio(0.1)
    trainloader=torch.utils.data.DataLoader(trainset,256,shuffle=True)
    validate_loader=torch.utils.data.DataLoader(validateset,256,shuffle=False)

    negative_set=rawdata.get_negative_dataset(postive_name=model_NFA,used_ratio=0.1)
    neg_count=len(negative_set)
    neg_train_count=int(np.floor(neg_count*0.9))
    negative_train_set=torch.utils.data.Subset(negative_set,range(neg_train_count))
    negative_train_loader=torch.utils.data.DataLoader(negative_train_set,69,shuffle=True)

    negative_validate_set=torch.utils.data.Subset(negative_set,range(neg_train_count,neg_count))
    negative_validate_loader=torch.utils.data.DataLoader(negative_validate_set,69,shuffle=False)

    dis=Discriminator(9,5).to(device)
    opt=torch.optim.Adam(dis.parameters())
    bce=nn.BCELoss()
    p_label=torch.ones(256,1).to(device)
    n_label=torch.zeros(256,1).to(device)
    writer=SummaryWriter(f'runs/discriminate/{model_NFA}_exp3')
    for epoch in range(200):
        for batch,(p,n) in enumerate(zip(trainloader,negative_train_loader)):
            px=p[0].to(device)
            pc=model.inference(px,future=None,step=1,trmode=False,gd=False)['context']
            po=dis(pc)
            ploss=bce(po,p_label[:px.shape[0]])

            nx=n[0].to(device)
            nc=model.inference(nx,future=None,step=1,trmode=False,gd=False)['context']
            no=dis(nc)
            nloss=bce(no,n_label[:nx.shape[0]])

            total_loss=ploss+nloss
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            train_acc=((po>=0.5).count_nonzero()+(no<0.5).count_nonzero())/(px.shape[0]+nx.shape[0])
            ...
        #validate
        for batch,(vp,vn) in enumerate(zip(validate_loader,negative_validate_loader)):
            vpx=vp[0]
            vpc=model.inference(vpx,future=None,step=1,trmode=False,gd=False)['context']
            vpo=dis(vpc)
            vploss=bce(vpo,p_label[:vpx.shape[0]])

            vnx=vn[0]
            vnc=model.inference(vnx,future=None,step=1,trmode=False,gd=False)['context']
            vno=dis(vnc)
            vnloss=bce(vno,n_label[:vnx.shape[0]])

            val_acc=((vpo>=0.5).count_nonzero()+(vno<0.5).count_nonzero())/(vpx.shape[0]+vnx.shape[0])
            print(f'epoch:{epoch}')
            print(f'    train ploss:{ploss}, nloss:{nloss}, acc={train_acc}')
            print(f'    validate ploss:{vploss}, nloss{vnloss}, acc={val_acc}')
            writer.add_scalar('train/ploss',ploss,epoch)
            writer.add_scalar('train/nloss',nloss,epoch)
            writer.add_scalar('train/acc',train_acc,epoch)
            writer.add_scalar('validate/ploss',vploss,epoch)
            writer.add_scalar('validate/nloss',vnloss,epoch)
            writer.add_scalar('validate/acc',val_acc,epoch)
            writer.flush()
            break

    ...