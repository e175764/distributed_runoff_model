import numpy as np
import pandas as pd
import csv
import contextlib
import glob
import itertools
import datetime as dt
from tqdm import tqdm
from numba import jit,prange
     
def test(file,name):
    root_df=pd.read_csv(file,
                        encoding="shift-jis",
                        header=1,
                        usecols=name
                        )
    data=root_df.drop(root_df.index[[0,1,2,3,4,5]])
    data=data.iloc[::-1]
    data=data[name].reset_index(drop=True)
    return data

def test2(file,name):
    root_df=pd.read_csv(file,
                        encoding="shift-jis",
                        header=1,
                        usecols=name
                        )
    data=root_df.drop(root_df.index[[0,1]])
    data=data.iloc[::-1]
    data=data.reset_index(drop=True)
    return data

def make_riv_data(root_path,names):
    path=root_path+"river/"
    year_file=sorted(glob.glob(path+"*"))
    all_file=[]
    
    for i in year_file:
        all_file.append(sorted(glob.glob(i+"/*")))
    files=list(itertools.chain.from_iterable(all_file))
    datas=pd.DataFrame()
    for i in files:
        temp=test(i,names)
        datas=pd.concat([datas,temp],axis=0)
    datas=datas.reset_index(drop=True)
    datas=datas.rename(columns={'観測所':'time'})
    for name in datas.columns:
        if name!="time":
            datas[name]=datas[name].mask(datas[name].str.contains('未収集',na=False,regex=False))
            datas[name]=datas[name].mask(datas[name].str.contains('欠測',na=False,regex=False))
    #datas[names[1]].interpolate(axis=0,inplace=True,limit_direction="both")
    datas[names[1]]=datas[names[1]].fillna(method="bfill")
    #print(datas[names[1]].isnull().sum())
    return datas
    
def make_rain_data(root_path,names):
    path=root_path+"rain/"
    year_file=sorted(glob.glob(path+"*"))
    all_file=[]
    for i in year_file:
        all_file.append(sorted(glob.glob(i+"/*")))
    files=list(itertools.chain.from_iterable(all_file))
    datas=pd.DataFrame()
    for i in files:
        temp=test2(i,names)
        datas=pd.concat([datas,temp],axis=0)
    datas=datas.reset_index(drop=True)
    datas=datas.rename(columns={'観測所':'time'})
    for name in datas.columns:
        if name!="time":
            datas[name]=datas[name].mask(datas[name].str.contains('未収集',na=False,regex=False))
            datas[name]=datas[name].mask(datas[name].str.contains('欠測',na=False,regex=False))
    for name in names:
        if name!="観測所":
            #datas[name]=datas[name].interpolate()
            datas[name]=datas[name].fillna(value=0)
            #print(datas[name].isnull().sum())
    return datas
                  

def make_input_data(water,rain,name,lt,start,stop):
    test_x=[]
    test_y=[]
    train_x=[]
    train_y=[]
    temp_water=[]
    temp_rain=[]
    temp_x=[]
    height=[]
    rain=rain.drop(columns="time").astype(float)
    #rain["木部"]=rain["木部"].shift(6)
    #rain=rain.shift(12)
    water2=water[name].astype(float)-water[name].astype(float).shift(lt*6)#shift何時間予測
    change=water[name].astype(float)-water[name].astype(float).shift(1)#何時間分の変化を使うか
    for i in tqdm(range(len(water))):
        temp_x=np.array([])
        if 30 <= i <= len(rain)-(lt*6+1):#+18->3時間前までの水位変化を入力にする:現在時刻18
            temp_water=change[i-18+1:i+1].values#1個はNan
            #temp_water=np.concatenate([temp_water, (water[name][i-18:i].values)],0)
            temp_water=np.append(temp_water, float(water[name][i]))
            temp_rain=rain[i-(30-lt*6):i+(lt*6)].values
            temp_x=np.concatenate([temp_water,temp_rain.flatten()],0)
            #if start <= pd.to_datetime(water["time"][i-18]) and pd.to_datetime(water["time"][i+18])<=stop:
            if start > pd.to_datetime(water["time"][i-18]) :
                train_x.append(temp_x)
                train_y.append(water2[i+lt*6])  
            elif pd.to_datetime(water["time"][i+18]) <= stop:
                test_x.append(temp_x)
                test_y.append(water2[i+lt*6])
                height.append(water[name][i])
            else:
                break
    return test_x,test_y,train_x,train_y,height

if __name__=="__main__" :    
    path="/Users/e175764/Desktop/Okazaki-lab/data/"
    riv_datas=make_riv_data(path,["観測所","因原"])
    rain_datas=make_rain_data(path,["観測所","川本＿河","邑智"])

    #test_x,test_y,train_x,train_y,height = make_input_data2(riv_datas,rain_datas,"町田",3,dt.datetime(2011,7,2),dt.datetime(2011,7,5))
    #print(rain_datas)
    make_input_data(riv_datas,rain_datas,"因原",3,dt.datetime(2018,7,5),dt.datetime(2018,7,12))
    

    
    #test("/Users/e175764/Desktop/Okazaki-lab/data/river/suii_10min_2019/suii_10min_201903.csv",["観測所","町田"])
    #test2("/Users/e175764/Desktop/Okazaki-lab/data/rain/uryou_10m_2009/uryou_10m_200902.csv",["観測所","津和野＿河","名賀"])