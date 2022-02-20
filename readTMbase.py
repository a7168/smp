"""
Created on Fri Feb 18 16:21:42 2022

@author: egoist
"""
import math
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset


class TMbaseset(Dataset):
	def __init__(self,datapath):
		dfs=[pd.read_csv(d,index_col=0) for d in datapath]
		df=pd.concat(dfs,ignore_index=True).dropna(axis=1)
		self.data=df[[c for c in df.columns if c !='time']].to_numpy(dtype=np.float32).mean(axis=1,keepdims=True)
		...

	def parse(self,seqLength,normalize):
		self.seqLength=seqLength
		self.indices=list(range(0,len(self.data)-seqLength+1))
		self.data={'max':self.normalize01,
					'z':self.normalizeZ}.get(normalize)()

	def setbackfore(self,backend,forestart=None):
		self.sep=(backend,forestart if forestart is not None else backend)

	def __getitem__(self, index):
		head=self.indices[index]
		seq=self.data[head:head+self.seqLength]
		return seq[:self.sep[0]],seq[self.sep[1]:]

	def __len__(self):
		return len(self.indices)

	def normalizeZ(self):#z-normalization
		mean=self.data.mean()
		std=self.data.std()
		return (self.data-mean)/std
	
	def normalize01(self):#map to 0-1
		dfmax=self.data.max()
		dfmin=self.data.min()
		return (self.data-dfmin)/(dfmax-dfmin)

	def splitbyratio(self,ratio): #use last block to validate
		total=len(self)
		bound=int(np.floor(total*ratio))
		train=self.indices[:-bound]
		validate=self.indices[-bound:]
		return TMbasesubset(self,train),TMbasesubset(self,validate)

class TMbasesubset(Data.Subset):
	def __init__(self,dataset,indices):
		super().__init__(dataset,indices)
		self.visualindices=None

	def setvisualindices(self,size):
		self.visualindices=self.getsample(list(range(len(self))),size)

	def getvisualbatch(self):
		if self.dataset.sep is None:
			return np.stack([self[i] for i in self.visualindices])
		else:
			return [np.stack(j) for j in zip(*[self[i] for i in self.visualindices])]
			
	def getsample(self,indices,size):
		return self.rng.choice(indices,size=size,replace=False)

	@classmethod
	def setrng(cls,seed):
		cls.rng=np.random.default_rng(seed=seed)

class TMbase():
	def __init__(self,datapath,use_cols,
				timeunit,align,normalized_method,nanThreshold,forecast_length,backcast_length,
				globalrng,samplesize,
				train_batch,valid_batch):

		rawdata=TMbaseset(datapath)
		rawdata.parse(seqLength=timeunit*(forecast_length+backcast_length),
						normalize=normalized_method,)
		rawdata.setbackfore(backcast_length)

		trainset,validateset=rawdata.splitbyratio(0.1)
		trainset.setrng(seed=self.generate_seed(globalrng,1000000)) #generate a num as seed to control sample
		trainset.setvisualindices(samplesize)
		validateset.setvisualindices(samplesize)

		trainloader=torch.utils.data.DataLoader(trainset,train_batch,shuffle=True)
		valloader=torch.utils.data.DataLoader(validateset,valid_batch,shuffle=False)

		self.trainset=trainset
		self.validateset=validateset
		self.trainloader=trainloader
		self.valloader=valloader

	@staticmethod
	def generate_seed(rng,maxseed):
		return rng.integers(maxseed)


def _localtest():
	rawdata=TMbaseset(['dataset/TMbase/data_200501_211031.csv',
						'dataset/TMbase/data_2111.csv'])
	rawdata.parse(seqLength=7*24+24,
						normalize='z',)
	rawdata.setbackfore(7*24)
	...

if __name__=='__main__':
	_localtest()