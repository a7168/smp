# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:34:01 2021

@author: egoist
"""

import pandas as pd
import numpy as np
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

class IHEPCread():
	def __init__(self,datasetpath,usesubs='g',preview=False):
		cols={'g':'Global_active_power',
				'1':'Sub_metering_1',
				'2':'Sub_metering_2',
				'3':'Sub_metering_3',}
		
		print('reading IHEPC dataset')
		usecols=[cols.get(i) for i in usesubs]
		df=pd.read_csv(datasetpath,sep=';',
				 parse_dates={'dt': ['Date', 'Time']},
				 infer_datetime_format=True,
				 low_memory=False,
				 na_values=['nan', '?'],
				 index_col='dt',
				 usecols=['Date', 'Time']+usecols,
				 dtype={i:np.float32 for i in usecols})
		if preview:
			print('----dataset preview----\n',df.head())
		idxMv=np.where(df['Global_active_power'].isnull())[0] # index of every miss value
		print(f'IHEPC dataset miss value count: {len(idxMv)}')
		idxrCmv=[] #index range of continuous miss value
		for i,j in zip(np.insert(idxMv,0,-2),idxMv):
			if i+1==j:
				idxrCmv[-1][-1]=j
			else:
				idxrCmv.append([j,j])
		for i in idxrCmv:
			i.append(i[1]-i[0]+1)
# 		print(idxrCmv)
		self.idxrCmv=idxrCmv
		self.df=df.interpolate()

	def normalizeZ(self):#z-normalization
		mean=self.df.mean()
		std=self.df.std()
		return (self.df-mean)/std
	
	def normalize01(self):#map to 0-1
		dfmax=self.df.max()
		dfmin=self.df.min()
		return (self.df-dfmin)/(dfmax-dfmin)
		
	def parse(self,seqLength,timeunit,align=60,normalize='z',nanRT=100,msg=True):
		df=self.normalize01() if normalize=='max' else self.normalizeZ()
		idxrCmv=[i for i in self.idxrCmv if i[2]>nanRT] #index range of continuous miss value after threshold
		idxrCmv_=[(0,-1)]+idxrCmv+[(len(df),0)]
		idxrData=[[idxrCmv_[i-1][1]+1,idxrCmv_[i][0]-1] for i in range(1,len(idxrCmv_))]
		idxrData=[i for i in idxrData if i[1]-i[0]+1>=seqLength]
		if msg:
			print('------------------')
			print(f'dataset contain {len(idxrData)} segments:')
			print(f'{"range":^15}\t\tcount')
			for start,end in idxrData:
				print(f'{start:7}-{end:7}: \t{end-start+1}')
		
		dfs=[df.iloc[i:j+1] for i,j in idxrData]
		blocks=[IHEPCblock(df,seqLength,align) for df in dfs]
		return IHEPCset(blocks,timeunit)
		
class IHEPCblock(Dataset):
	def __init__(self,df,seqLength,align):
		self.df=df
		self.align=align
		self.starttimestamp=df.index[0]
		self.indices=list(range(self.getbias(),
						  len(df)-seqLength+1,
						  self.align))
		self.seqLength=seqLength
		
	def setalign(self,align):
		self.align=align

	def __getitem__(self,idx):# decide which sequence return
		head=self.indices[idx]
		return self.df.iloc[head:head+self.seqLength].to_numpy()
		
	def __len__(self):
		return len(self.indices)
		
	def getbias(self):
		if self.align==1:
			return 0
		minbias=self.starttimestamp.minute
		if self.align==60:
			return -minbias%60
		hourbias=self.starttimestamp.hour
		if self.align==1440:
			return -(hourbias*60+minbias)%1440
		
# =============================================================================
# 	def getstep(self):
# 		return {'minute':1,'hour':60,'day':1440}.get(self.align)
# =============================================================================
	
class IHEPCset(Data.ConcatDataset):
	def __init__(self,blocks,timeunit,sep=None):
		self.timeunit=timeunit
		self.sep=sep
		super().__init__(blocks)
		
	def setbackfore(self,backend,forestart=None):
		self.sep=(backend,forestart if forestart is not None else backend)
		
	def __getitem__(self, index):
		seq_min=super().__getitem__(index)
		if self.timeunit==1:
			seq=seq_min
		elif self.timeunit==60:
			seq=seq_min.reshape(-1,60,1).mean(axis=1)
		else:
			raise TypeError(f'format {self.timeunit} not defined')
			
		if self.sep is None :
			return seq
		else:
# 			unitsize=seq.shape[0]//(self.back+self.fore)
			return seq[:self.sep[0]],seq[self.sep[1]:]
			
	def splitbyblock(self): #use last block to validate
		train=list(range(self.cumulative_sizes[-2]))
		validate=list(range(self.cumulative_sizes[-2],self.cumulative_sizes[-1]))
		return IHEPCsubset(self,train),IHEPCsubset(self,validate)
	
	def splitbyeachblock(self):
		...
	
	
class IHEPCsubset(Data.Subset):
	def __init__(self,dataset,indices):
		super().__init__(dataset,indices)
		self.visualindices=None
		
	def setvisualindices(self,rng,size):
		self.visualindices=self.getsample(rng,list(range(len(self))),size)
		
	def getvisualbatch(self):
		if self.dataset.sep is None:
			return np.stack([self[i] for i in self.visualindices])
		else:
			return [np.stack(j) for j in zip(*[self[i] for i in self.visualindices])]
			
	@staticmethod
	def getsample(rng,indices,size):
		return rng.choice(indices,size=size,replace=False)
	
	

if __name__=='__main__':
	ihepcdf=IHEPCread('dataset/IHEPC/household_power_consumption.txt',usesubs='g')
	ds1=ihepcdf.parse(11520,timeunit=1,align=60)
	ds2=ihepcdf.parse(11520,timeunit=60,align=60)
	ds3=ihepcdf.parse(15,'hour',normalize='max')
	dataloader=Data.DataLoader(ds1,batch_size=128,shuffle=True)
	for x in dataloader:
		print(x.shape)
		break
# 	print(ds1[2])

