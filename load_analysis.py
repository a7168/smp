'''
Author: Egoist
Date: 2022-01-03 20:11:40
LastEditors: Egoist
LastEditTime: 2022-08-07 15:16:31
FilePath: /smp/load_analysis.py
Description: 

'''
#%%
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import date as Date
# import re


def getargs(args=None):
    parser=argparse.ArgumentParser()
    #files
    parser.add_argument('--info',type=str,default='data/info.csv')
    parser.add_argument('--data',type=str,nargs='+')
    #
    parser.add_argument('--show',type=str,
                  help='''n-None
                          y-year
                          ''')
    parser.add_argument('--projection',type=str,default='b')
    args=parser.parse_args(args=args)
    print(f'use args : {args}')
    return args


class Rules():
    def __init__(self):
        self.rules={'year':{i:f'{i:02}-' for i in range(2020,2023)},
                    'month':{i:f'-{i:02}-' for i in range(1,13)},
                    'day':{i:f'-{i:02} ' for i in range(1,32)},
                    'hour':{i:f' {i:02}:' for i in range(24)},
                    'week':{i:(i+6)%7 for i in range(7)},
                    
                    'building':{16:16,17:17,19:19},
                    'floor':{i:i for i in range(2,22)},
                    'roomtype':{0:0,1:1,2:2,3:3},
                    'residents':{i:i for i in range(1,10)}
                    }
    def __getitem__(self,key):
        return self.rules[key]
    
    def save(self,path):
        ...
    def load(self,load):
        ...
    def add(self,dict):
        ...

class AssociateDataFrame():
    def __init__(self,name,idx,info,data):
        self.name=name
        self.idx=idx
        self.info=info
        self.data=data
        
# =============================================================================
# 	def select  year,month,season,day,hour
# 				base,building,floor,roomType,residents
# =============================================================================

    def select(self,rules,stype,addnone=False):
        if stype is None:
            return [self]
        elif stype in ('year','month','day','hour'):
            return self.select_time(stype,rules.rules[stype],addnone)
        elif stype=='week':
            return self.select_weekday(stype,rules.rules[stype],addnone)
        elif stype=='residents':
            return self.select_residents(stype,rules.rules[stype],addnone)
        elif stype=='roomtype':
            return self.select_roomtype(stype,rules.rules[stype],addnone)
        elif stype=='floor':
            return self.select_floor(stype,rules.rules[stype],addnone)
        elif stype=='building':
            return self.select_building(stype,rules.rules[stype],addnone)
        else:
            print('condition undefined')
        
    def select_weekday(self,name,conds,addnone=False):
        new=type(self)
        returndfs=[]
        for idx,pattern in conds.items():
            mask=pd.to_datetime(self.data['time']).apply(lambda x:True 
                                   if x.day_of_week==pattern else False)
            if mask.any():
                returndfs.append(new(name,idx,self.info,self.data[mask]))
            elif addnone:
                returndfs.append(None)
        return returndfs

    def select_building(self,name,conds,addnone=False):
        new=type(self)
        returndfs=[]
        for idx,pattern in conds.items():
            mask=self.info['NFAcode'].apply(lambda x:True 
                                   if int(x.split('-')[0][1:])==pattern else False)
            if mask.any():
                selectedinfo=self.info[mask]
                dc=self.data.columns
                mask_dc=dc.isin(selectedinfo['NFAcode'])
                mask_dc=self.addtime(mask_dc)
                mask_data=dc[mask_dc]
                returndfs.append(new(name,idx,selectedinfo,self.data[mask_data]))
            elif addnone:
                returndfs.append(None)
        return returndfs

    def select_floor(self,name,conds,addnone=False):
        new=type(self)
        returndfs=[]
        for idx,pattern in conds.items():
            mask=self.info['NFAcode'].apply(lambda x:True 
                                   if int(x.split('-')[1][1:])==pattern else False)
            if mask.any():
                selectedinfo=self.info[mask]
                dc=self.data.columns
                mask_dc=dc.isin(selectedinfo['NFAcode'])
                mask_dc=self.addtime(mask_dc)
                mask_data=dc[mask_dc]
                returndfs.append(new(name,idx,selectedinfo,self.data[mask_data]))
            elif addnone:
                returndfs.append(None)
        return returndfs

    def select_roomtype(self,name,conds,addnone=False):
        new=type(self)
        returndfs=[]
        for idx,pattern in conds.items():
            mask=self.info['NFAcode'].apply(lambda x:True 
                                   if self.getroomtype(x)==pattern else False)
            if mask.any():
                selectedinfo=self.info[mask]
                dc=self.data.columns
                mask_dc=dc.isin(selectedinfo['NFAcode'])
                mask_dc=self.addtime(mask_dc)
                mask_data=dc[mask_dc]
                returndfs.append(new(name,idx,selectedinfo,self.data[mask_data]))
            elif addnone:
                returndfs.append(None)
        return returndfs
    
    def select_residents(self,name,conds,addnone=False): #cond is [int,]
        new=type(self)
        returndfs=[]
        for idx,pattern in conds.items():
            mask=(self.info['residents']==pattern)
            if mask.any():
                selectedinfo=self.info[mask]
                dc=self.data.columns
                mask_dc=dc.isin(selectedinfo['NFAcode'])
                mask_dc=self.addtime(mask_dc)
                mask_data=dc[mask_dc]
                returndfs.append(new(name,idx,selectedinfo,self.data[mask_data]))
            elif addnone:
                returndfs.append(None)
        return returndfs
        
    def select_time(self,name,conds,addnone=False): #cond is [str,]
        new=type(self)
        returndfs=[]
        for idx,pattern in conds.items():
            mask=self.data['time'].str.contains(pattern)
            if mask.any():
                returndfs.append(new(name,idx,self.info,self.data[mask]))
            elif addnone:
                returndfs.append(None)
        return returndfs
    
    def select_timerange(self,rstart,rend):
        dfstart=self.todate(self.data['time'][0])
        startidx=(rstart-dfstart).days*24
        endidx=((rend-dfstart).days+1)*24
        self.data=self.data.iloc[startidx:endidx]
    
    @staticmethod
    def addtime(mask):
        mask[0]=True
        return mask
        
    @staticmethod
    def clipinfo(info,data):
        preserved=info['NFAcode'].isin(data.columns)
        return info[preserved]
        
    @staticmethod
    def todate(datestr):
        ymd=datestr.split(' ')[0].split('-')
        return Date(*[int(i) for i in ymd])
    
    
    @staticmethod
    def getroomtype(NFA):
        """0:suite
            1:type 1
            2:type 2
            3:type 3"""
        roomtype={'N16':[1,2,0,2,None,2,1,1,2,2],
                    'N17':[2,1,1,1,None,1,2,0,0,1,1,1,1,2],
                    'N19':[3,3,1,1,None,1,1,1,1,2,0,0,1,1,None,1,1]}
        no,floor,area=NFA.split('-')
        area=int(area[1:])
        return roomtype.get(no)[area]


class Visualizer():
    def __init__(self,info,*datas,rules,projection):
# 		self.infodf=pd.read_csv(info)
        dfs=[pd.read_csv(d,index_col=0) for d in datas]
        data=pd.concat(dfs,ignore_index=True).dropna(axis=1)
        info=AssociateDataFrame.clipinfo(pd.read_csv(info),data)
        self.adf=AssociateDataFrame('total',0,info,data)
        self.rules=rules
        self.projection='3d' if projection=='b' else None
        self.table={'y':'year',
                    'm':'month',
                    'w':'week',
                    'd':'day',
                    'h':'hour',
                    'b':'building',
                    'f':'floor',
                    't':'roomtype',
                    'r':'residents'}
    
    def fillplot(self,padf,ax,config):
        ax.set_title(f'{padf.name}-{padf.idx}')
        rows=padf.select(self.rules,self.table.get(config[2]),addnone=True)
        for idx, row in enumerate(rows):
            if row is None:
                continue
            bars=row.select(self.rules,self.table.get(config[3]))
            barvalues=[i.data[[c for c in i.data.columns if c !='time']].to_numpy().mean() 
              for i in bars]
            barlabels=[i.idx for i in bars]
            
            if self.projection is not None:
                ax.bar(barlabels,barvalues,row.idx,zdir='y',color=f'C{idx}')
            else:
                ax.plot(barlabels,barvalues,label=row.idx,color=f'C{idx}')
                # ax.scatter(barlabels,barvalues,label=row.idx,color=f'C{idx}')
                
        ax.set_xlabel(bars[0].name)
        if len(dic:=self.rules[self.table.get(config[3])])<6:
            ax.set_xticks([i for i in dic.keys()])
        
        if self.projection is not None:
            ax.set_ylabel(row.name)
            if len(rows)<6:
                ax.set_yticks([t.idx for t in rows])
        else:
            ax.legend(fontsize='x-small')
            ax.set_ylabel('load')
        
        ...
    
    def fillfig(self,adf,config):
        fig=plt.figure()
        fig.suptitle(f'{adf.name}-{adf.idx}')
        
        padfs=adf.select(self.rules,self.table.get(config[1]))
        w=int(np.ceil(np.sqrt(len(padfs))))
        h=int(np.ceil(len(padfs)/w))
        print(f'(w/h)={w}/{h}')
        if h>2:
            fig.set_figheight(2.3*h)
        if w>3:
            fig.set_figwidth(2.3*w)
        for i,padf in enumerate(padfs,1):
            ax=fig.add_subplot(w,h,i,projection=self.projection)
            self.fillplot(padf,ax,config)
        fig.tight_layout()
    
    def show(self,config):
        for adf in self.adf.select(self.rules,self.table.get(config[0])):
            self.fillfig(adf,config)
        if self.projection is not None:
            plt.subplots_adjust(left=0.125,
                            bottom=0.05,
                            right=0.9,
                            top=0.93,
                            wspace=0.4,
                            hspace=0.55)
        plt.show()
    

def main(argv=None):
    args=getargs(argv)
    ru=Rules()
# 	vis=Visualizer('info.csv','data_200501_211031.csv','data_2111.csv','data_2112.csv')
    vis=Visualizer(args.info,*args.data,rules=ru,projection=args.projection)
    vis.show(args.show)


if __name__=='__main__':
    main(["--info","dataset/TMbase/info.csv",
          "--data","dataset/TMbase/data_200501_211031.csv",
                   "dataset/TMbase/data_2111.csv",
                   "dataset/TMbase/data_2112.csv",
                   "dataset/TMbase/data_2201.csv",
          "--show","wmbh",
          "--projection","l",])
    ...