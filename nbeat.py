# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:12:25 2021

@author: egoist
"""

import argparse
import readIHEPC
from readIHEPC import IHEPC
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
	def __init__(self,model,trloader,valloader,lossf,opt,device,tb_log_dir,lossratio):
		self.model=model.to(device)
		self.trloader=trloader
		self.valloader=valloader
		self.lossf=lossf
		self.opt=opt
		self.device=device
		self.writer=SummaryWriter(tb_log_dir)
		# self.useback2train=useback2train
		# self.useback2eval=useback2eval
		self.lossratio=lossratio
		
	def train(self,epochs,record):
		iteration=-1
		for ep in range(epochs):
			for batch,(x,y) in enumerate(self.trloader):
				iteration+=1
				x,y=[i.squeeze(-1) for i in (x,y)]
				back,fore=self.inference(x,trmode=True,gd=True)

				# lossall=self.evaluate(x,y,back,fore,useback=True)
				# lossfore=self.evaluate(x,y,back,fore,useback=False)
				loss=self.evaluate2(x,y,back,fore)

				# self.update(lossall if self.useback2train else lossfore)
				self.update(sum([i*j for i,j in zip(loss,self.lossratio)]))

				
				if record[0]=='i' and iteration%(int(record[1:]))==0:
					self.validate(ep,batch,iteration,loss)
					
			if record[0]=='e':
				self.validate(ep,batch,iteration,loss)

	def update(self,loss):
		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

	def validate(self,ep,batch,itrn,trainloss):
		# vall,vfore=np.mean([[self.evaluate(*[i.squeeze(-1) for i in (x,y)],*self.inference(x,trmode=False,gd=False),useback=ub).cpu() 
		# 					for ub in (True,False)]
		# 		for x,y in self.valloader],axis=0)
		verr=np.mean([self.evaluate2(*[i.squeeze(-1) for i in (x,y)],
									*self.inference(x,trmode=False,gd=False),tocpu=True) for x,y in self.valloader],axis=0)

		stepstr=f'epoch/batch/iteration : {ep}/{batch}/{itrn}'
		trainstr=f'train back={trainloss[0].item():f} | train fore={trainloss[1].item():f} | train all={trainloss[2].item():f}'
		valstr=f'valiadate back={verr[0]:f} | valiadate fore={verr[1]:f} | valiadate all={verr[2]:f}'
		print(f'{stepstr} ][ {trainstr} ][ {valstr}')
		
		self.writer.add_scalar('train/all',trainloss[2].item(),itrn)
		self.writer.add_scalar('train/back',trainloss[0].item(),itrn)
		self.writer.add_scalar('train/fore',trainloss[1].item(),itrn)
		self.writer.add_scalar('validate/all',verr[2],itrn)
		self.writer.add_scalar('validate/back',verr[0],itrn)
		self.writer.add_scalar('validate/fore',verr[1],itrn)
		
		trainsample_x,trainsample_y=[torch.from_numpy(i) for i in self.trloader.dataset.getvisualbatch()]
		trainsample_b,trainsample_f=[i.cpu() for i in self.inference(trainsample_x,trmode=False,gd=False)]
		for idx,x,y,b,f in zip(self.trloader.dataset.visualindices,trainsample_x,trainsample_y,trainsample_b,trainsample_f):
			self.writer.add_figure(f'train_{idx}/all',self.plotall(x,y,b,f),itrn)
			self.writer.add_figure(f'train_{idx}/fore',self.plotfore(y,f),itrn)
			self.writer.add_figure(f'train_{idx}/back',self.plotback(x,b),itrn)
			
		valsample_x,valsample_y=[torch.from_numpy(i) for i in self.valloader.dataset.getvisualbatch()]
		valsample_b,valsample_f=[i.cpu() for i in self.inference(valsample_x,trmode=False,gd=False)]
		for idx,x,y,b,f in zip(self.valloader.dataset.visualindices,valsample_x,valsample_y,valsample_b,valsample_f):
			self.writer.add_figure(f'validate_{idx}/all',self.plotall(x,y,b,f),itrn)
			self.writer.add_figure(f'validate_{idx}/fore',self.plotfore(y,f),itrn)
			self.writer.add_figure(f'validate_{idx}/back',self.plotback(x,b),itrn)
		
		self.writer.flush()

	def inference(self,data,trmode,gd):
		data=data.to(self.device)
		if trmode is True:
			self.model.train()
		else:
			self.model.eval()

		if gd is True:
			return self.model(data)
		with torch.no_grad():
			return self.model(data)
	
	# def evaluate(self,x,y,b,f,useback):
	# 	if useback is True:
	# 		f=torch.cat((b,f),-1)
	# 		y=torch.cat((x,y),-1)
	# 	return self.lossf(f,y.to(self.device)) #+useback*self.lossf(b,x.to(self.device))

	def evaluate2(self,x,y,b,f,tocpu=False):
		loss_back=self.lossf(b,x.to(self.device))
		loss_fore=self.lossf(f,y.to(self.device))
		loss_all =self.lossf(torch.cat((b,f),-1),torch.cat((x,y),-1).to(self.device))
		loss=[i.cpu() if tocpu else i for i in (loss_back,loss_fore,loss_all)]
		return loss
	
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

	def count_params(self,cond='all'):
		cond_f={'all':lambda x:True,
				'trainable':lambda x:x.requires_grad}.get(cond)
		return sum(p.numel() for p in self.model.parameters() if cond_f(p))

	
class ARGS():
	def __init__(self): #TODO parse arg by tb_log_dir?
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
		parser.add_argument('-tdim','--thetas_dim',type=str,default='8,8')
		parser.add_argument('-swis','--share_weights_in_stack',type=bool,default=False)
		parser.add_argument('-hlu','--hidden_layer_units',type=int,default=128)
		parser.add_argument('-pm','--predictModule',type=lambda s:None if s=='' else s,default=None)
		parser.add_argument('-pml','--predict_module_layer',type=lambda s:None if s=='' else self.tonumlist(s),default=None)
		parser.add_argument('-spm','--share_predict_module',type=lambda s:None if s=='' else bool(s),default=None)
		parser.add_argument('-pmnl','--predict_module_num_layers',type=lambda s:None if s=='' else int(s),default=None)

		#training setting
		parser.add_argument('-d','--cudadevice',type=int,default=0)
		parser.add_argument('-rs','--rngseed',type=int,default=None)
		parser.add_argument('-e','--epochs',type=int,default=35)
		parser.add_argument('-tbd','--tb_log_dir',type=str,default=None)
		parser.add_argument('-r','--record',type=str,default='e') #i100
		parser.add_argument('-tb','--train_batch',type=int,default=512)
		# parser.add_argument('-vr','--valid_ratio',type=int,default=0.1)
		parser.add_argument('-vb','--valid_batch',type=int,default=512)
		parser.add_argument('-ss','--samplesize',type=int,default=8)
		parser.add_argument('-lr','--lossratio',type=self.tofloatlist,default=[0,0,1]) #
		# parser.add_argument('-ub2t','--useback2train',type=bool,default=False)
		# parser.add_argument('-ub2e','--useback2eval',type=bool,default=False)
		
		self.args=parser.parse_args()
		print(f'use args : {self.args}')
		self.dev=self.checkDevice(self.args.cudadevice)
		self.globalrng=np.random.default_rng(seed=self.args.rngseed)
		
	def __getattr__(self,key):
		attr=getattr(self.args,key)
		
		if key=='stack_types':
			return self.getstacktype(attr)
		elif key=='thetas_dim':
			return self.tonumlist(attr)
		return attr
	
	@staticmethod
	def getstacktype(s):
		stacktype={'g':NBeatsNet.GENERIC_BLOCK,'s':NBeatsNet.SEASONALITY_BLOCK,
				't':NBeatsNet.TREND_BLOCK,'c':NBeatsNet.GENERIC_CNN}
		return [stacktype.get(i) for i in s]
	
	@staticmethod
	def tonumlist(s):
		return [int(i) for i in s.split(',')]
	
	@staticmethod
	def tofloatlist(s):
		return [float(i) for i in s.split(',')]
	
	@staticmethod
	def checkDevice(cudadevice):
		print('=============================')
		print(f'python ver. {sys.version}')
		print(f'pytorch ver. {torch.__version__}')
		print(f'cuda ver. {torch.version.cuda}')
		print(f'cuda avail : {(cuava:=torch.cuda.is_available())}')
		print(f'use device: {(dev:=torch.device(f"cuda:{cudadevice}" if cuava else "cpu"))}')
		print('=============================')
		return dev
	
if __name__=='__main__':
	args=ARGS()
	
	dataset = IHEPC(datapath=args.datapath,
					use_cols=args.use_cols,
					timeunit=args.timeunit,
					align=args.align,
					normalized_method=args.normalized_method,
					nanThreshold=args.nanThreshold,
					forecast_length=args.forecast_length,
					backcast_length=args.backcast_length,
					globalrng=args.globalrng,
					samplesize=args.samplesize)

	# rawdata=readIHEPC.IHEPCread(datasetpath=args.datapath,
	# 							usesubs=args.use_cols)
	# wholedataset=rawdata.parse(seqLength=args.timeunit*(args.forecast_length+args.backcast_length),
	# 							timeunit=args.timeunit,
	# 							align=args.align,
	# 							normalize=args.normalized_method,
	# 							nanRT=args.nanThreshold)
	# wholedataset.setbackfore(args.backcast_length)

	# trainset,validateset=wholedataset.splitbyblock()
	# trainset.setrng(seed=args.globalrng.integers(1000000)) #generate a num as seed to control sample
	# trainset.setvisualindices(args.samplesize)
	# validateset.setvisualindices(args.samplesize)

	trainloader=torch.utils.data.DataLoader(dataset.trainset,args.train_batch,shuffle=True)
	valloader=torch.utils.data.DataLoader(dataset.validateset,args.valid_batch,shuffle=False)
	
	net = NBeatsNet(
		device=args.dev,
		stack_types=args.stack_types,
		nb_blocks_per_stack=args.nb_blocks_per_stack,
		forecast_length=args.forecast_length,
		backcast_length=args.backcast_length,
		thetas_dim=args.thetas_dim,
		share_weights_in_stack=args.share_weights_in_stack,
		hidden_layer_units=args.hidden_layer_units,
		predictModule=args.predictModule,
		predict_module_layer=args.predict_module_layer,
		share_predict_module=args.share_predict_module,
		predict_module_num_layers=args.predict_module_num_layers)
	
	opt=torch.optim.Adam(net.parameters())
	exp=Trainer(net,trainloader,valloader,nn.MSELoss(),opt,
				device=args.dev,
				tb_log_dir=args.tb_log_dir,
				# useback2train=args.useback2train,
				# useback2eval=args.useback2eval,
				lossratio=args.lossratio)
	print(f'params: {exp.count_params()}')
	exp.train(epochs=args.epochs,
		#    backcoef=args.backcoef,
		   record=args.record)
	...