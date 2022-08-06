'''
Author: Egoist
Date: 2021-10-07 02:13:45
LastEditors: Egoist
LastEditTime: 2022-08-06 12:14:25
FilePath: /smp/api.py
Description: 

'''

# %%
import os
import json
import requests
import numpy as np
import pandas as pd
import re
import pickle
from datetime import date as Date,timedelta

# from funcs import drawmdlist,drawhd,time2date,queryDays

class Connection():
    def __init__(self,path='access/connection.json'):
        self.load_connection_data(path)

    def load_connection_data(self,path):
        with open(path, 'r') as f:
            info=json.load(f)
        self.ip=info['ip']
        self.proxies=info['proxies']

def time2date(time):
    if isinstance(time,Date):
        return time
    elif isinstance(time,(tuple,list)):
        return Date(*time)
    elif isinstance(time,str):
        t=[int(i) for i in time.split('-')]
        return Date(*t)

def queryDays(start,end):
    def getmonthrange(year,startTime,endTime,addend=True):
        begin=startTime.month if startTime.year==year else 1
        end=endTime.month if year==endTime.year else 12
        return (begin,end+1) if addend else (begin,end)

    def getdayrange(year,month,startTime,endTime,addend=True):
        begin=startTime.day if  startTime.year==year and startTime.month==month else 1
        def maxday():
            if month in (1,3,5,7,8,10,12):
                return 31
            elif month in (4,6,9,11):
                return 30
            elif year%400==0 or (year%4==0 and year%100!=0):
                return 29
            else:
                return 28
            
        end=endTime.day if year==endTime.year and month==endTime.month else maxday()
        return (begin,end+1) if addend else (begin,end)

    start,end=[time2date(i) for i in (start,end)]
    return ((y,m,d) for y in range(start.year,end.year+1) 
                       for m in range(*getmonthrange(y,start,end)) 
                           for d in range(*getdayrange(y,m,start,end)))

class ReadResidents():
    def __init__(self,file='access/110resident.xls'):
        self.df=pd.read_excel(file)
        self.NFAlist=[self.NFAcode(addr) for addr in self.df['房屋住址']]
        
    def address2NFA(self,address):
        _,no,floor,area=re.split('巷|號|樓',address)
        no=int(no)
        floor={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,
         '十一':11,'十二':12,'十三':13,'十四':14,'十五':15,'十六':16,'十七':17,
         '十八':18,'十九':19,'二十':20,'二十一':21,}.get(floor)
        area=0 if len(area)==0 else int(area[1:])
        return no,floor,area
    
    def NFAcode(self,address):
        no,floor,area=self.address2NFA(address)
        return f'N{no:02}-F{floor:02}-A{area:02}'
    
    def getinfo(self,idx):
        series=self.df.iloc[idx]
        return {'NFAcode':self.NFAlist[idx],
            'base':0 if self.NFAlist[idx][:3]=='N16' else 1,
            'address':series['房屋住址'],
            'roomtype':series['房型'],
            '坪數':series['坪數'],
            'residents':series['居住人數']}
    
    def __len__(self):
        return len(self.NFAlist)
    
class Meterdata():
    def __init__(self,NFA,baseidx,info,perioddata):
        self.NFA=NFA
        self.baseidx=baseidx
        self.info=info
        self.start=perioddata['start']
        self.end=perioddata['end']
        self.dataList=perioddata['data']
        
    def __len__(self):
        return len(self.dataList)
    
    def update(self,api,end):
        start=self.end+ timedelta(days=1) if self.end is not None else self.start
        end=time2date(end)
        if (end-start).days >=0:
            apidata=api.getperiod(self.NFA,start,end)
            if apidata['success']:
                self.end=end
                self.dataList=self.dataList+apidata['data']
                print(f'{self.NFA} update to {end} success')
            elif apidata['end'] is not None:
                self.end=apidata['end']
                self.dataList=self.dataList+apidata['data']
                print(f'{self.NFA} only update to {self.end}')
            else:
                print(f'{self.NFA} update failure')
        else:
            #print(f'endtime {end} is before start')
            ...
        
class TMAPI():
    def __init__(self,num,coninfo):
        self.baseidx=num
        self.proxy=coninfo.proxies
        self.ip=coninfo.ip[num]
        self.response=requests.get(f'http://{self.ip}/ebuilding/meter.asmx/getPowersMeters',
                             proxies=self.proxy)
        if self.response.status_code == requests.codes.ok:
            self.text=self.response.text
            self.infoList=self.response.json()
            self.descDict={dic['description']:idx for idx,dic in enumerate(self.infoList)}
            self.NFAdict={self.getNFAcode(idx):idx for idx in range(len(self))}
            print(f'request success ...a total of {len(self.infoList)} records got.')
        else:
            self.text=self.infoList=self.descDict=None
            print('request failure')
        
    def getNFAbyapiIdx(self,idx):
        desc=self.infoList[idx]['description']
        if desc=='':
            return None
        else:
            _,area,floor=re.split('A|B|F',desc) 
            area,floor=(int(i) for i in (area,floor))
            no=16 if self.baseidx==0 else {'A':17,'B':19}.get(desc[0])
            if no==16:
                area=[None,8,9,0,1,2,3,5,6,7][area]
            elif no==17:
                area=0 if floor==1 else [None,7,8,9,10,11,12,13,0,1,2,3,5,6][area]
            elif no==19:
                area=[None,9,8,7,6,5,3,2,1,0,16,15,13,12,11,10][area]
            return no,floor,area
        
    def __len__(self):
        return len(self.infoList)
    
    def getapiIdxbyNFA_desc(self,identifier):
        if isinstance(identifier,int):
            return identifier
        elif len(identifier.split('-'))==1:
            return self.descDict.get(identifier)
        else:
            return self.NFAdict.get(identifier)
        
    def getNFA(self,identifier):
        idx=self.getapiIdxbyNFA_desc(identifier)
        return self.getNFAbyapiIdx(idx)
    
    def getNFAcode(self,identifier):
        nfa=self.getNFA(identifier)
        if nfa is None:
            return ''
        no,floor,area=nfa
        return f'N{no:02}-F{floor:02}-A{area:02}'
        
    def getinfo(self,identifier):
        idx=self.getapiIdxbyNFA_desc(identifier)
        infodict=self.infoList[idx]
        preDict={'base':f'base{self.baseidx}',
                   'apiIdx':idx,
                   'NFAcode':self.getNFAcode(idx)}
        return {**preDict,**infodict}
        
    def get1d(self,identifier,td,time):
        td={'d':'Day','w':'Week','m':'Month','y':'Year'}.get(td)
        time=str(time2date(time))
        argd={'valueID':self.getinfo(identifier)['valueID'],
                'timeDivision':td,'time':time}
        r=requests.get(f'http://{self.ip}/ebuilding/meter.asmx/getMainPowerMeterByTimeDivision',
                 params=argd,proxies=self.proxy)
        if r.status_code == requests.codes.ok:
            try:
                return r.json()
            except Exception as e:
                print(e)
        else:
            print(f'get1d request {identifier} failure')
            
    def getperiod(self,identifier,start,end):
        perioddata={'start':time2date(start),'end':None,'data':[],'success':None,'log':None}
        for year, month, day in queryDays(start,end):
            data=self.get1d(identifier, 'd', f'{year}-{month}-{day}')
            if data is not None and len(data)>0:
                perioddata['data'].extend([i['y2'] for i in data])
# 				perioddata['start']=Date(year, month, day) if perioddata['start'] is None else perioddata['start']
                perioddata['end']=Date(year, month, day)
            else:
                print(f'get {identifier} period data on {year}-{month}-{day} failure')
                perioddata['success']=False
                perioddata['log']=(data,Date(year, month, day))
                return perioddata
        else:
            print(f'get {identifier} data success')
            perioddata['success']=True
        return perioddata
            
class SummaryTable():
    def __init__(self,start,end,coninfo):
        self.baseapi=[TMAPI(i,coninfo) for i in (0,1)]
        self.residents=ReadResidents()
        self.start=time2date(start)
        self.end=time2date(end)
        self.meters=[]
        # self.fails={}
        for idx in range(len(self.residents)): #len(self.residents)
            info1=self.residents.getinfo(idx)
            NFAcode=info1['NFAcode']
            baseidx=info1['base']
            info2=self.baseapi[baseidx].getinfo(NFAcode)
            apidata=self.baseapi[baseidx].getperiod(NFAcode,start,end)

            self.meters.append(Meterdata(NFAcode,baseidx,{**info1,**info2},apidata))
                
    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self, f)
            
    def update(self,end):
        end=time2date(end)
        self.end=end if (end-self.end).days >0 else self.end
        for meter in self.meters:
            if meter.end is None or (self.end-meter.end).days >=0:
                meter.update(self.baseapi[meter.baseidx],end)
            
    def todf(self):
        df1={i:[] for i in self.meters[0].info}
        days=[(y,m,d) for y,m,d in queryDays(self.start,self.end)]
        df2={'time':[f'{y}-{m:02}-{d:02} {h:02}:00' for y,m,d in days for h in range(24)]}
        
        for meter in self.meters:
            if meter.end==self.end:
                NFAcode=meter.info['NFAcode']
                for key,value in meter.info.items():
                    df1[key].append(value)
                
                df2[NFAcode]=meter.dataList
        return pd.DataFrame(df1),pd.DataFrame(df2)

def main(startdate=(2022,5,1),
         enddate=(2022,5,1),
         outdir='tempdata'):
    coninfo=Connection()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    table=SummaryTable(startdate,enddate,coninfo)
    table.update(enddate)
    table.update(enddate)
    df1,df2=table.todf()
    # df1.to_csv(f'{outdir}/info.csv',index=False,encoding='utf-8-sig')
    df2.to_csv(f'{outdir}/data_{startdate[0]%100}{startdate[1]:02}.csv',encoding='utf-8-sig')
    table.save(f'{outdir}/pdtable_{startdate[0]%100}{startdate[1]:02}.pkl')
    ...
# =============================================================================
# 	with open('st.pkl', 'rb') as f:
# 		st2 = pickle.load(f)
# 	st2.update((2021,10,31))
# =============================================================================

if __name__=='__main__':
    main()
