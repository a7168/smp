"""
Created on Fri Feb 18 16:21:42 2022

@author: egoist
"""
import json
import math
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset


class TMbaseset(Dataset):
	def __init__(self,datapath,use_cols):
		dfs=[pd.read_csv(d,index_col=0) for d in datapath]
		df=pd.concat(dfs,ignore_index=True).dropna(axis=1)
		useless_list=self.data_cleansing(df,0.01,120*5,0)
		df=df.drop(useless_list, axis=1)

		if use_cols=='g':
			self.data=df[[c for c in df.columns if c !='time']].to_numpy(dtype=np.float32).mean(axis=1,keepdims=True)
		else:
			self.data=np.expand_dims(df[use_cols].to_numpy(dtype=np.float32),axis=-1)
		...
		
	@staticmethod
	def save_use_list(df,path='dataset/TMbase/use_list.json'):
		with open('use_list.json', 'w', encoding='utf-8') as f:
			json.dump([i for i in df.columns if i !='time'], f, ensure_ascii=False, indent=4)

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

		rawdata=TMbaseset(datapath,use_cols)
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
						'dataset/TMbase/data_2111.csv',
						'dataset/TMbase/data_2112.csv',
						'dataset/TMbase/data_2201.csv'],'g')
	rawdata.parse(seqLength=7*24+24,
						normalize='z',)
	rawdata.setbackfore(7*24)
	...

if __name__=='__main__':
	_localtest()