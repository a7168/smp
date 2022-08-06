'''
Author: Egoist
Date: 2022-02-18 16:21:42
LastEditors: Egoist
LastEditTime: 2022-07-19 17:20:26
FilePath: /smp/readTMbase.py
Description: 

'''
# %%
import json
import math
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data

# import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset


class TMbaseset(Dataset):
    def __init__(self,datapath):
        dfs=[pd.read_csv(d,index_col=0) for d in datapath]
        df=pd.concat(dfs,ignore_index=True).dropna(axis=1)
        self.df=df
        self.start=pd.to_datetime(self.df['time'])[0]
        ...
        
    @staticmethod
    def save_use_list(df,path='dataset/TMbase/use_list.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([i for i in df.columns if i !='time'], f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_use_list(path):
        with open(path, 'r') as f:
            uselist=json.load(f)
        return uselist

    @staticmethod
    def filter_use_list(uselist,floors):
        match=[]
        for nfa in uselist:
            n,f,a=nfa.split('-')
            if int(f[1:]) not in floors:
                continue
            if n=='N16' and (a=='A01' or a=='A09'):
                match.append(nfa)
            elif n=='N17' and (a=='A00' or a=='A13'):
                match.append(nfa)
            elif n=='N19' and (a=='A01' or a=='A09'):
                match.append(nfa)
        return match


    def parse(self,date_range,align,threshold,cleaned_user_list,normalize,use_cols,seq_length):
        #select used dataset range
        df=self.select_timerange(self.df,date_range[0],date_range[1]) if date_range is not None else self.df

        if cleaned_user_list is not None:
            self.use_list=self.load_use_list(cleaned_user_list)
        else:
            useless_list=self.data_cleansing(df,threshold['value'],threshold['length'],0)
            self.use_list=[c for c in df.columns if c !='time' and c not in useless_list]
        if use_cols=='g':
            if threshold is not None:
                df=df[self.use_list]
            data=df.to_numpy(dtype=np.float32).mean(axis=1,keepdims=True)
        else:
            data=np.expand_dims(df[use_cols].to_numpy(dtype=np.float32),axis=-1)

        self.data={'max':self.normalize01,
                   'z':self.normalizeZ,
                   '':self.normalize_none}.get(normalize)(data)

        self.seq_length=seq_length
        self.indices=list(range(0,len(self.data)-seq_length+1,align))
        self.align=align

    def setbackfore(self,backend,forestart=None):
        self.sep=(backend,forestart if forestart is not None else backend)

    def __getitem__(self, index):
        head=self.indices[index]
        seq=self.data[head:head+self.seq_length]
        return seq
        return seq[:self.sep[0]],seq[self.sep[1]:]

    def getitembydate(self,date,length=1,NFA=None):
        data=self.df
        date=date if isinstance(date,pd.Timestamp) else pd.Timestamp(*date)
        startidx=(date-self.start).days*24
        if NFA is None:
            return data.iloc[startidx:startidx+24*length]
        else:
            return data.iloc[startidx:startidx+24*length][NFA]

    def get_month_mean(self,month,NFA):
        month_first=pd.Timestamp(month)
        days=month_first.days_in_month
        total_data=self.getitembydate(date=month_first,length=days,NFA=NFA)
        return torch.from_numpy(total_data.to_numpy(dtype=np.float32)).view(-1,24).mean(0)
        ...


    @staticmethod
    def select_timerange(df,start_date,end_date):
        t0=pd.to_datetime(df['time'])[0]
        idx_start=(start_date-t0).days*24 if start_date is not None else None
        idx_end=(end_date-t0).days*24+24 if end_date is not None else None
        return df.iloc[idx_start:idx_end]

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def normalize_none(data):
        return data

    @staticmethod
    def normalizeZ(data):#z-normalization for dataframe
        mean=data.mean()
        std=data.std()
        return (data-mean)/std
    
    @staticmethod
    def normalize01(data):#map to 0-1 for dataframe
        dfmax=data.max()
        dfmin=data.min()
        return (data-dfmin)/(dfmax-dfmin)

    def splitbyratio(self,ratio): #use last block to validate
        total=len(self)
        bound=int(np.floor(total*ratio))
        train=range(0,total-bound)
        validate=range(total-bound,total)
        # train=self.indices[:-bound]
        # validate=self.indices[-bound:]
        return TMbasesubset(self,train),TMbasesubset(self,validate)

    def plot_user(self,user,days,start_date=None,figsize=(30,10)):
        start_date=self.start if start_date is None else pd.Timestamp(start_date)
        data=self.getitembydate(start_date,length=days)[user].to_numpy()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data)
        plt.title(user)
        plt.show()
        return fig
    
    def get_negative_dataset(self,postive_name,used_ratio):
        total_length=len(self.df)
        bound=int(np.floor(total_length*used_ratio))
        user_dataset_list=[]
        cond1=TMbaseset.filter_use_list(self.use_list,floors=[4,11,15,18])
        for name in cond1:
            if name!=postive_name:
                dataset=TMUserDataSet(name=name,
                                      data=np.expand_dims(self.df[name].iloc[:-bound].to_numpy(dtype=np.float32),axis=-1),
                                      seq_length=self.seq_length,
                                      sep=self.sep,
                                      align=self.align)
                user_dataset_list.append(dataset)
        
        return TMUserConcatSet(user_dataset_list)

    @staticmethod
    def data_cleansing(input,value_threshold,conti_day,gragh):
        """ input : dataframe
            value_threshold : the value considered to small to be recorded
            conti_day : continue days of abscent value
            gragh : 0 no gragh 1 only useless gragh 2 only useful gragh 3 all
            output : a list with all the abondan electricity meter"""
        useless_cnt = 0
        useless_list = []
        each_sensor = input.columns
        for sensor in each_sensor:
            useless = False
            if sensor == "time":
                continue
            data = list(input[sensor])
            valid = []
            for i in range(len(data)):
                if data[i] == 0:
                    valid.append(False)
                else:
                    valid.append(True)
            scatter_x = np.arange(len(data))
            scatter_y = np.zeros((len(data),))
            cont_zero = 0
            normal_cnt = 0
            in_con = False
            for i in range(len(data)):
                scatter_y[i] = data[i]
                if data[i] < value_threshold and not in_con:
                    in_con = True
                    cont_zero += 1
                    normal_cnt = 0
                elif in_con and data[i] < value_threshold:
                    cont_zero += 1
                    if normal_cnt > 1:
                        normal_cnt -= 1
                elif in_con and data[i] >= value_threshold:
                    normal_cnt += 1
                if normal_cnt > 5:
                    in_con = False
                    cont_zero = 0
                if cont_zero > conti_day:
                    useless = True
            group = np.zeros(len(data),)
            for i in range(len(data)):
                if valid[i] == True:
                    group[i] = 2
                else: 
                    group[i] = 2
            cdict = {1: 'red', 2: 'blue'}
            if useless:
                k = "useless"
            else:
                k = "useful"
            if useless:
                useless_cnt += 1
                useless_list.append(sensor)
            if gragh == 0:
                continue
            elif gragh == 1:
                if not useless:
                    continue
            elif gragh == 2:
                if useless:
                    continue  
            fig, ax = plt.subplots(figsize=(30,10))
            for g in np.unique(group):
                ix = np.where(group == g)
                ax.plot(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g)
            ax.legend()
            plt.title(sensor + " " + str(k),fontsize = 20)
            plt.show()
        print("Abandon ratio : " + str(useless_cnt / len(each_sensor) * 100) + "%")
        return useless_list

class TMbasesubset(Data.Subset):
    def __init__(self,dataset,indices):
        super().__init__(dataset,indices)
        self.visualindices=None

    def setvisualindices(self):
        indices=list(range(len(self)))
        self.visualindices=self.rng.choice(indices,size=len(self),replace=False)

    def getvisualbatch(self,size):
        if self.dataset.sep is None:
            return np.stack([self[i] for i in self.visualindices[:size]])
        else:
            return np.stack([self[i] for i in self.visualindices[:size]])
            # return [np.stack(j) for j in zip(*[self[i] for i in self.visualindices])]

    @classmethod
    def setrng(cls,seed):
        cls.rng=np.random.default_rng(seed=seed)

class TMUserDataSet(torch.utils.data.Dataset):
    def __init__(self,name,data,seq_length,sep,align):
        self.name=name
        self.data=data
        self.seq_length=seq_length
        self.indices=list(range(0,len(self.data)-seq_length+1,align))
        self.sep=sep
        ...

    def __getitem__(self,index):
        head=self.indices[index]
        seq=self.data[head:head+self.seq_length]
        return seq
        return seq[:self.sep[0]],seq[self.sep[1]:]

    def __len__(self):
        return len(self.indices)

class TMUserConcatSet(torch.utils.data.ConcatDataset):
    def __init__(self,datasets):
        super().__init__(datasets)

    def setvisualindices(self):
        indices=list(range(len(self)))
        self.visualindices=self.rng.choice(indices,size=len(self),replace=False)

    def getvisualbatch(self,size):
        return np.stack([self[i] for i in self.visualindices[:size]])

    @classmethod
    def setrng(cls,seed):
        cls.rng=np.random.default_rng(seed=seed)

class TMbase():
    def __init__(self,datapath,
                 date_range,data_clean_threshold,cleaned_user_list,normalized_method,use_cols,
                 timeunit,align,
                 forecast_length,backcast_length,
                 globalrng,samplesize,
                 train_batch,train_negative_batch,valid_batch):

        rawdata=TMbaseset(datapath)
        rawdata.parse(date_range=date_range,
                      align=align,
                      threshold=data_clean_threshold,
                      cleaned_user_list=cleaned_user_list,
                      normalize=normalized_method,
                      use_cols=use_cols,
                      seq_length=timeunit*(forecast_length+backcast_length),)
        rawdata.setbackfore(backcast_length)

        trainset,validateset=rawdata.splitbyratio(0.1)
        seed=self.generate_seed(globalrng,1000000)
        trainset.setrng(seed=seed) #generate a num as seed to control sample
        trainset.setvisualindices()
        validateset.setvisualindices()

        trainloader=torch.utils.data.DataLoader(trainset,train_batch,shuffle=True)
        valloader=torch.utils.data.DataLoader(validateset,valid_batch,shuffle=False)

        self.trainset=trainset
        self.validateset=validateset
        self.trainloader=trainloader
        self.valloader=valloader

        if train_negative_batch is not None:
            self.negative_set=rawdata.get_negative_dataset(postive_name=use_cols,
                                                           used_ratio=0.1)
            self.negative_set.setrng(seed=seed)
            self.negative_set.setvisualindices()
            self.negative_loader=torch.utils.data.DataLoader(self.negative_set,train_negative_batch,shuffle=True)
        else:
            self.negative_set=None
            self.negative_loader=None
        ...

    @staticmethod
    def generate_seed(rng,maxseed):
        return rng.integers(maxseed)


def _localtest():
    rawdata=TMbaseset(datapath=['dataset/TMbase/data_200501_211031.csv',
                                'dataset/TMbase/data_2111.csv',
                                'dataset/TMbase/data_2112.csv',
                                'dataset/TMbase/data_2201.csv',
                                'dataset/TMbase/data_2202.csv',
                                'dataset/TMbase/data_2203.csv',
                                'dataset/TMbase/data_2204.csv',
                                ])
    rawdata.parse(date_range=None,
                  align=1,
                  threshold={'value':0.01,'length':600},
                  cleaned_user_list=None,
                  normalize='',
                  use_cols='N16-F04-A01',
                  seq_length=1*(168+24),)
    rawdata.setbackfore(7*24)
    trainset,validateset=rawdata.splitbyratio(0.1)
    a=trainset[2]
    ...

if __name__=='__main__':
    _localtest()