# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 22:36:12 2021

@author: egoist
"""
import sys
from functools import partial
import torch
import torch.nn as nn

import readIHEPC

def checkDevice():
	print(f'python ver. {sys.version}')
	print(f'pytorch ver. {torch.__version__}')
	print(f'cuda ver. {torch.version.cuda}')
	print(f'cuda avail : {(cuava:=torch.cuda.is_available())}')
	print(f'use device: {(dev:=torch.device("cuda" if cuava else "cpu"))}')
	return dev

def MAPE(x,y,mae=nn.L1Loss(reduction='none')):
	return (mae(x,y)/y).abs().mean()
	

class LSTM(nn.Module):
	def __init__(self,inSize,hidSize,layernum,concerned=None):
		self.concerned=concerned
		super().__init__()
		self.lstm=nn.LSTM(inSize,hidSize,layernum,batch_first=True)
		self.relu=nn.ReLU()
		self.fc=nn.Linear(hidSize,1)
		
	def forward(self,x):
		x,hc=self.lstm(x)
		output=self.fc(self.relu(x))
		return output if self.concerned is None else output[:,-self.concerned:,:]
def getfixSample(dataset,totalnum,batch=None):
	dl=torch.utils.data.DataLoader(dataset,totalnum,True)
	for i in dl:
		fixSampleSet=torch.utils.data.TensorDataset(*i)
		return torch.utils.data.DataLoader(fixSampleSet,
									 batch if batch else totalnum,shuffle=False)
	
class Trainer():
	def __init__(self,model,device):
		self.model=model.to(device)
		self.device=device
		
	def train(self,epoch,dataloader,lossf,opt,validate):
		for ep in range(epoch):
			for batch,(x,y) in enumerate(dataloader):
				output=self.model(x.to(self.device))
				loss=lossf(output,y.to(self.device))
				opt.zero_grad()
				loss.backward()
				opt.step()
				print(f'epoch:{ep}/{batch}\t val={validate()}') if batch %20==0 else None
				
	def inference(self,data):
		with torch.no_grad():
			return self.model(data.to(self.device))
		
	def evaluate(self,dataloader,metric):
		error=0
		for batch,(x,gt) in enumerate(dataloader,1):
# 		used,gt=pairdata
			result=self.inference(x)
			error+=metric(result,gt.to(self.device))
			...
		return error/batch

if __name__=='__main__':
	device=checkDevice()
	
	dataset=readIHEPC.IHEPCpreproc('household_power_consumption.txt')
	train,*vt=dataset(10079,60) 
	vtsampleArgs=((i,2048,256) for i in vt)
# 	validsample,testsample=map(getfixSample,*zip(*vtsampleArgs))
	validsample,testsample=(getfixSample(*i) for i in vtsampleArgs)
	trainloader=torch.utils.data.DataLoader(train,512,True)
	
	lstm=LSTM(1,20,2)
	opt=torch.optim.Adam(lstm.parameters())
	exp=Trainer(lstm,device)
	validate=partial(exp.evaluate,validsample,MAPE)
	exp.train(200,trainloader,nn.MSELoss(),opt,validate)
