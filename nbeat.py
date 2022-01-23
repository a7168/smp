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
from nbeats_pytorch.model import NBeatsNet
# from nbeatmodel import NBeatsNet

import warnings
warnings.filterwarnings(action='ignore', message='Setting attributes')

#todo: lossf, parser(or env)

# =============================================================================
# class Dataset():
# 	def __init__(self,fpath,forecast_length,backcast_length,usesubs,align,normalize):
# 		self.ihepcdf=readIHEPC.IHEPCpreproc(fpath,usesubs=usesubs)
# 		self.dataset=self.ihepcdf(backcast_length-1,
# 									forecast_length,
# 									forecast_length,
# 									align=align,
# 									normalize=normalize)
# 		
# 	def divide(self,validateRatio,samplesize):
# 		splitset=torch.utils.data.Subset
# 		indices=list(range(len(self.dataset)))
# 		
# 		lastsize=len(self.dataset.datasets[7])
# # 		testsize=len(self.dataset)*testSize if testSize<1 else testSize
# # 		trInd,valInd=indices[testsize:],indices[:testsize]
# # 		trInd,valInd=indices[:1893678],indices[1893678:]
# 		trInd,valInd=indices[:-lastsize],indices[-lastsize:]
# # 		trInd,valInd=indices[125:],indices[:125]
# # 		trInd,valInd=self.divide_eachblock()
# # 		trsam=self.getsample(trInd,samplesize)
# # 		valsam=self.getsample(valInd,samplesize)
# 
# # 		rng=np.random.default_rng()
# 		ARGS.rng.shuffle(valInd)
# # 		valInd=ARGS.rng.choice(valInd,size=512,replace=False)
# 		
# # =============================================================================
# # 		todo
# # 		trInd=valInd=[]
# # 		for ...:
# # 		showsample...
# # =============================================================================
# 		return splitset(self.dataset,trInd),splitset(self.dataset,valInd)
# 		
# 	def divide_eachblock(self,validateRatio=0.1):
# 		trInd=valInd=[]
# 		startidx=0
# 		for ds in self.dataset.datasets:
# 			lds=len(ds)
# 			allidx=list(range(startidx,startidx+lds))
# 			trInd=trInd+allidx[:-np.floor(lds*validateRatio).astype(int)]
# 			valInd=valInd+allidx[-np.floor(lds*validateRatio).astype(int):]
# 			startidx=startidx+lds
# 		
# 		return trInd,valInd
# 	
# 	@staticmethod
# 	def getsample(indices,size):
# 		return ARGS.rng.choice(indices,size=size,replace=False)
# 	
# # =============================================================================
# # 	@staticmethod
# # 	def getfixSample(dataset,totalnum,batch=None):
# # 		dl=torch.utils.data.DataLoader(dataset,totalnum,True)
# # 		for i in dl:
# # 			fixSampleSet=torch.utils.data.TensorDataset(*i)
# # 			return torch.utils.data.DataLoader(fixSampleSet,
# # 											  batch if batch else totalnum,shuffle=False)
# # =============================================================================
# =============================================================================
		
class Trainer():
	def __init__(self,model,trloader,valloader,lossf,opt,device,tb_log_dir):
		self.model=model.to(device)
		self.trloader=trloader
		self.valloader=valloader
		self.lossf=lossf
		self.opt=opt
		self.device=device
		self.writer=SummaryWriter(tb_log_dir)
		
# =============================================================================
# 	@staticmethod
# 	def checkDevice():
# 		print(f'python ver. {sys.version}')
# 		print(f'pytorch ver. {torch.__version__}')
# 		print(f'cuda ver. {torch.version.cuda}')
# 		print(f'cuda avail : {(cuava:=torch.cuda.is_available())}')
# 		print(f'use device: {(dev:=torch.device("cuda" if cuava else "cpu"))}')
# 		return dev
# =============================================================================
		
	def train(self,epochs,backcoef,record):
		iteration=-1
# 		rng=np.random.default_rng()
		for ep in range(epochs):
			for batch,(x,y) in enumerate(self.trloader):
				self.model.train()
				iteration+=1
				back,fore=self.model(x.to(self.device))
				loss=self.lossf(fore,y[...,0].to(self.device))+backcoef*self.lossf(back,torch.zeros_like(x[...,0]).to(self.device))
# 				loss=self.lossf(fore,y[...,0].to(self.device))+backcoef*self.lossf(back,x[...,0].to(self.device))
				self.opt.zero_grad()
				loss.backward()
				self.opt.step()
				
				if record[0]=='i' and iteration%(int(record[1:]))==0:
					self.validate(ep,batch,iteration,loss.item())
					
			if record[0]=='e':
				self.validate(ep,batch,iteration,loss.item())
# =============================================================================
# 				if iteration%mesInterval==0:
# 					test_err,lastsample=self.evaluate(valloader,self.lossf,lastsample=True)
# 					stepinfo={'ep':ep,'iteration':iteration,'step':iteration}
# 					trlog={'loss':loss.item(),'x':x,'y':y[...,0],'f':fore.detach().cpu()}
# 					vallog={'loss':test_err.item(),'x':lastsample[0],'y':lastsample[1],
# 							 'f':lastsample[2]}
# # 					pickindices=rng.choice(min(len(x),len(lastsample[0])),size=4,replace=False)
# 					self.trainlogging(stepinfo,trlog,vallog,)
# =============================================================================
# =============================================================================
# 					step=iteration
# 					print(f'epoch:{ep} iteration:{iteration} | training loss={loss.item():f} | test_loss={test_err.item():f}')
# 					self.writer.add_scalar('loss/train',loss.item(),step)
# 					self.writer.add_scalar('loss/validate',test_err.item(),step)
# 					
# 					trainbgf=(x,y[...,0],fore.detach().cpu())
# 					for ploti,tx,ty,tf,vx,vy,vf in zip(range(4),*trainbgf,*lastsample):
# 						self.writer.add_figure(f'train/all{ploti}',self.plotall(tx,ty,tf),step)
# 						self.writer.add_figure(f'train/fore{ploti}',self.plotfore(ty,tf),step)
# 						self.writer.add_figure(f'validate/all{ploti}',self.plotall(vx,vy,vf),step)
# 						self.writer.add_figure(f'validate/fore{ploti}',self.plotfore(vy,vf),step)
# 					self.writer.flush()
# =============================================================================

	def validate(self,ep,batch,itrn,trainloss):
		test_err=self.evaluate(self.valloader,self.lossf).item()
		stepstr=f'epoch/batch/iteration : {ep}/{batch}/{itrn}'
		print(f'{stepstr} | train loss={trainloss:f} | test err={test_err:f}')
		
		self.writer.add_scalar('loss/train',trainloss,itrn)
		self.writer.add_scalar('loss/validate',test_err,itrn)
		
		trainsample_x,trainsample_y=[torch.from_numpy(i) for i in self.trloader.dataset.getvisualbatch()]
		trainsample_f=self.inference(trainsample_x).cpu()
		for idx,x,y,f in zip(self.trloader.dataset.visualindices,trainsample_x,trainsample_y,trainsample_f):
			self.writer.add_figure(f'train/all_{idx}',self.plotall(x,y,f),itrn)
			self.writer.add_figure(f'train/fore_{idx}',self.plotfore(y,f),itrn)
			
		valsample_x,valsample_y=[torch.from_numpy(i) for i in self.valloader.dataset.getvisualbatch()]
		valsample_f=self.inference(valsample_x).cpu()
		for idx,x,y,f in zip(self.valloader.dataset.visualindices,valsample_x,valsample_y,valsample_f):
			self.writer.add_figure(f'validate/all_{idx}',self.plotall(x,y,f),itrn)
			self.writer.add_figure(f'validate/fore_{idx}',self.plotfore(y,f),itrn)
		
		self.writer.flush()
		
		
	def trainlogging(self,stepinfo,trlog,vallog,pickindices=range(4)):
		step=stepinfo["step"]
		stepstr=f'epoch:{stepinfo["ep"]} iteration:{stepinfo["iteration"]}'
		print(f'{stepstr} | training loss={trlog["loss"]:f} | test_loss={vallog["loss"]:f}')
		self.writer.add_scalar('loss/train',trlog["loss"],step)
		self.writer.add_scalar('loss/validate',vallog["loss"],step)
		
		for ploti in pickindices:
			tx,ty,tf=[trlog[i][ploti] for i in ('x','y','f')]
			vx,vy,vf=[vallog[i][ploti] for i in ('x','y','f')]
			self.writer.add_figure(f'train/all{ploti}',self.plotall(tx,ty,tf),step)
			self.writer.add_figure(f'train/fore{ploti}',self.plotfore(ty,tf),step)
			self.writer.add_figure(f'validate/all{ploti}',self.plotall(vx,vy,vf),step)
			self.writer.add_figure(f'validate/fore{ploti}',self.plotfore(vy,vf),step)
		self.writer.flush()
				
	def inference(self,data):
		self.model.eval()
		with torch.no_grad():
			back,fore=self.model(data.to(self.device))
			return fore
		
	def evaluate(self,dataloader,metric,lastsample=False):
		error=0
		for batch,(x,gt) in enumerate(dataloader,1):
# 		used,gt=pairdata
			result=self.inference(x)
			error+=metric(result,gt[...,0].to(self.device))
			
		if lastsample:
			return error/batch,(x,gt[...,0],result.cpu())
		return error/batch
	
	def plotall(self,back,gt,fore):
		bl,fl=len(back),len(fore)
		
		fig, ax1 = plt.subplots()
		ax1.plot(range(0,bl),back,label='back',color='b')
		ax1.plot(range(bl,bl+fl),gt,label='ground truth',color='g')
		ax1.plot(range(bl,bl+fl),fore,label='forecast',color='r')
		
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
	
class ARGS():
	rng=np.random.default_rng()
	def __init__(self):
		parser=argparse.ArgumentParser()
		#dataset setting
		parser.add_argument('-dp','--datapath',type=str,default='household_power_consumption.txt')
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
# =============================================================================
# 	wholedataset=Dataset(args.datapath,
# 					  args.forecast_length,
# 					  args.backcast_length,
# 					  usesubs=args.use_cols,
# 					  align=args.align,
# 					  normalize=args.normalized_method)
# =============================================================================

	trainset,validateset=wholedataset.splitbyblock()
	trainset.setvisualindices(args.rng,args.samplesize)
	validateset.setvisualindices(args.rng,args.samplesize)
# =============================================================================
# 	dataset={i:j for i,j in zip(('train','validate')
# 					 ,wholedataset.divide(args.valid_ratio,args.samplesize))}
# =============================================================================
	
# =============================================================================
# 	sample={i:j for i,j in zip(('train','validate')
# 					 ,wholedataset.divide(args.valid_ratio,args.samplesize))}
# =============================================================================

	trainloader=torch.utils.data.DataLoader(trainset,args.train_batch,shuffle=True)
	valloader=torch.utils.data.DataLoader(validateset,args.valid_batch,shuffle=False)

# =============================================================================
# 	loaders={ds:torch.utils.data.DataLoader(dataset[ds],bt,sf)
# 			  for ds,bt,sf in zip(('train','validate'),
# 						 (args.train_batch,args.valid_batch),(True,False))}
# =============================================================================
	
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
	exp=Trainer(net,trainloader,valloader,nn.MSELoss(),opt,device=args.dev,tb_log_dir=args.tb_log_dir)
	exp.train(epochs=args.epochs,
		   backcoef=args.backcoef,
		   record=args.record)