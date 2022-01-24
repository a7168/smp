# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:12:25 2021

@author: egoist
"""

import argparse
import readIHEPC
import numpy as np
import torch
import torch.nn as nn
import sys
from torch.utils.tensorboard import SummaryWriter
# =============================================================================
# import matplotlib as mpl
# mpl.use('TkAgg')
# =============================================================================
import matplotlib.pyplot as plt
# from nbeats_pytorch.model import NBeatsNet
from nbeatmodel import NBeatsNet

import warnings
warnings.filterwarnings(action='ignore', message='Setting attributes')

class Trainer():
	def __init__(self,model,trloader,valloader,lossf,opt,device,tb_log_dir,useback2train,useback2eval):
		self.model=model.to(device)
		self.trloader=trloader
		self.valloader=valloader
		self.lossf=lossf
		self.opt=opt
		self.device=device
		self.writer=SummaryWriter(tb_log_dir)
		self.useback2train=useback2train
		self.useback2eval=useback2eval
		
		
	def train(self,epochs,backcoef,record):
		iteration=-1
# 		rng=np.random.default_rng()
		for ep in range(epochs):
			for batch,(x,y) in enumerate(self.trloader):#TODO useback2train
				self.model.train()
				iteration+=1
				back,fore=self.model(x.to(self.device))
				if self.useback2train is True:
					fore=torch.cat((back,fore),-1)
					y=torch.cat((x,y),1)

				# loss=self.lossf(fore,y[...,0].to(self.device))+backcoef*self.lossf(back,torch.zeros_like(x[...,0]).to(self.device))
				loss=self.lossf(fore,y[...,0].to(self.device))
				self.opt.zero_grad()
				loss.backward()
				self.opt.step()
				
				if record[0]=='i' and iteration%(int(record[1:]))==0:
					self.validate(ep,batch,iteration,loss.item())
					
			if record[0]=='e':
				self.validate(ep,batch,iteration,loss.item())

	def update(self,loss):
		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

	def validate(self,ep,batch,itrn,trainloss):
		test_err=self.evaluate(self.valloader,self.lossf).item()
		stepstr=f'epoch/batch/iteration : {ep}/{batch}/{itrn}'
		print(f'{stepstr} | train loss={trainloss:f} | test err={test_err:f}')
		
		self.writer.add_scalar('loss/train',trainloss,itrn)
		self.writer.add_scalar('loss/validate',test_err,itrn)
		
		trainsample_x,trainsample_y=[torch.from_numpy(i) for i in self.trloader.dataset.getvisualbatch()]
		trainsample_b,trainsample_f=[i.cpu() for i in self.inference(trainsample_x)]
		for idx,x,y,b,f in zip(self.trloader.dataset.visualindices,trainsample_x,trainsample_y,trainsample_b,trainsample_f):
			self.writer.add_figure(f'train/all_{idx}',self.plotall(x,y,b,f),itrn)
			self.writer.add_figure(f'train/fore_{idx}',self.plotfore(y,f),itrn)
			self.writer.add_figure(f'train/back_{idx}',self.plotback(x,b),itrn)
			
		valsample_x,valsample_y=[torch.from_numpy(i) for i in self.valloader.dataset.getvisualbatch()]
		trainsample_b,valsample_f=[i.cpu() for i in self.inference(valsample_x)]
		for idx,x,y,f in zip(self.valloader.dataset.visualindices,valsample_x,valsample_y,valsample_f):
			self.writer.add_figure(f'validate/all_{idx}',self.plotall(x,y,b,f),itrn)
			self.writer.add_figure(f'validate/fore_{idx}',self.plotfore(y,f),itrn)
			self.writer.add_figure(f'train/back_{idx}',self.plotback(x,b),itrn)
		
		self.writer.flush()
		
	def inference(self,data):
		self.model.eval()
		with torch.no_grad():
			back,fore=self.model(data.to(self.device))
			return back,fore
		
	def evaluate(self,dataloader,metric): #TODO eval不含推斷
		error=0
		for batch,(x,gt) in enumerate(dataloader,1):
# 		used,gt=pairdata
			result=self.inference(x)
			if self.useback2eval is True:
				result=torch.cat(result,-1)
				gt=torch.cat((x,gt),1)

			error+=metric(result[1],gt[...,0].to(self.device))
			
		return error/batch
	
	def plotall(self,x,y,b,f):
		xl,yl=len(x),len(y)
		
		fig, ax1 = plt.subplots()
		ax1.plot(range(xl+yl),torch.cat((x,y)),label='ground truth',color='g')
		ax1.plot(range(0,xl),b,label='back',color='b')
		ax1.plot(range(xl,xl+yl),f,label='forecast',color='r')
		
		ax1.legend()
		fig.tight_layout()
		return fig
	
	def plotfore(self,gt,fore):
		fl=len(fore)
		
		fig, ax1 = plt.subplots()
		ax1.plot(range(fl),gt,label='ground truth',color='g')
		ax1.plot(range(fl),fore,label='forecast',color='r')
		
		ax1.legend()
		fig.tight_layout()
		return fig

	def plotback(self,x,b):
		xl=len(x)

		fig, ax1 = plt.subplots()
		ax1.plot(range(xl),x,label='ground truth',color='g')
		ax1.plot(range(xl),b,label='BACKcast',color='b')
		
		ax1.legend()
		fig.tight_layout()
		return fig

	
class ARGS():
	rng=np.random.default_rng()
	def __init__(self):
		parser=argparse.ArgumentParser()
		#dataset setting
		parser.add_argument('-dp','--datapath',type=str,default='dataset/IHEPC/household_power_consumption.txt')
		parser.add_argument('-uc','--use_cols',type=str,default='g')
		parser.add_argument('-tu','--timeunit',type=int,default=60)
		parser.add_argument('-a','--align',type=int,default=60)
		parser.add_argument('-nm','--normalized_method',type=str,default='z',choices=['z','max'])
		parser.add_argument('-nt','--nanThreshold',type=int,default=100)
		
		#model setting
		parser.add_argument('-st','--stack_types',type=str,default='gg')
		parser.add_argument('-nbps','--nb_blocks_per_stack',type=int,default=3)
		parser.add_argument('-fl','--forecast_length',type=int,default=24)
		parser.add_argument('-bl','--backcast_length',type=int,default=7*24)
		parser.add_argument('-tdim','--thetas_dim',type=str,default='4,8')
		parser.add_argument('-swis','--share_weights_in_stack',type=bool,default=False)
		parser.add_argument('-hlu','--hidden_layer_units',type=int,default=128)
		#training setting
		#rngseed
		parser.add_argument('-e','--epochs',type=int,default=30)
		parser.add_argument('-tbd','--tb_log_dir',type=str,default=None)
		parser.add_argument('-r','--record',type=str,default='i100')
		parser.add_argument('-tb','--train_batch',type=int,default=512)
		parser.add_argument('-vr','--valid_ratio',type=int,default=0.1)
		parser.add_argument('-vb','--valid_batch',type=int,default=512)
		parser.add_argument('-ss','--samplesize',type=int,default=8)
		parser.add_argument('-bc','--backcoef',type=float,default=0)
		parser.add_argument('-ub2t','--useback2train',type=bool,default=False)
		parser.add_argument('-ub2e','--useback2eval',type=bool,default=False)
		
		self.args=parser.parse_args()
		print(f'use args : {self.args}')
		self.dev=self.checkDevice()
		
	def __getattr__(self,key):
		attr=getattr(self.args,key)
		
		if key=='stack_types':
			return self.getstacktype(attr)
		elif key=='thetas_dim':
			return self.getthetas_dim(attr)
		return attr
	
	@staticmethod
	def getstacktype(s):
		stacktype={'g':NBeatsNet.GENERIC_BLOCK,'s':NBeatsNet.SEASONALITY_BLOCK,
				't':NBeatsNet.TREND_BLOCK} #,'c':NBeatsNet.GENERIC_CNN
		return [stacktype.get(i) for i in s]
	
	@staticmethod
	def getthetas_dim(s):
		return [int(i) for i in s.split(',')]
	
	@staticmethod
	def checkDevice():
		print('=============================')
		print(f'python ver. {sys.version}')
		print(f'pytorch ver. {torch.__version__}')
		print(f'cuda ver. {torch.version.cuda}')
		print(f'cuda avail : {(cuava:=torch.cuda.is_available())}')
		print(f'use device: {(dev:=torch.device("cuda" if cuava else "cpu"))}')
		print('=============================')
		return dev
	
if __name__=='__main__':
	args=ARGS()
	
	rawdata=readIHEPC.IHEPCread(datasetpath=args.datapath,
								usesubs=args.use_cols)
	wholedataset=rawdata.parse(seqLength=args.timeunit*(args.forecast_length+args.backcast_length),
								timeunit=args.timeunit,
								align=args.align,
								normalize=args.normalized_method,
								nanRT=args.nanThreshold)
	wholedataset.setbackfore(args.backcast_length)

	trainset,validateset=wholedataset.splitbyblock()
	trainset.setvisualindices(args.rng,args.samplesize)
	validateset.setvisualindices(args.rng,args.samplesize)

	trainloader=torch.utils.data.DataLoader(trainset,args.train_batch,shuffle=True)
	valloader=torch.utils.data.DataLoader(validateset,args.valid_batch,shuffle=False)
	
	net = NBeatsNet(
		device=args.dev,
		stack_types=args.stack_types,
		nb_blocks_per_stack=args.nb_blocks_per_stack,
		forecast_length=args.forecast_length,
		backcast_length=args.backcast_length,
		thetas_dim=args.thetas_dim,
		share_weights_in_stack=args.share_weights_in_stack,
		hidden_layer_units=args.hidden_layer_units,)
	
	opt=torch.optim.Adam(net.parameters())
	exp=Trainer(net,trainloader,valloader,nn.MSELoss(),opt,
				device=args.dev,
				tb_log_dir=args.tb_log_dir,
				useback2train=args.useback2train,
				useback2eval=args.useback2eval)
	exp.train(epochs=args.epochs,
		   backcoef=args.backcoef,
		   record=args.record)
	...