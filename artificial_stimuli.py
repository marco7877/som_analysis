#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:22:34 2020

@author: marco
"""

def main():
    import numpy as np
    import glob
    import pandas as pd
    extention="*.txt"
    n=20
    filedirection="/media/marco/MarcoHDD/github/LipReading/ga/"
    files=glob.glob(filedirection+extention)
    print("Reading files from "+filedirection)
    filenumber=0
    saveplain=True
    normaldist=True
    if saveplain==True:
        documentofinal="ArtificialDescriptorGa2_gaussian.csv"
    for text in files:
        data_x = np.loadtxt(text,dtype=float, delimiter=",")
        if filenumber==0:
            obs=len(data_x)
            dataframe=pd.DataFrame(data_x)
            filenumber+=1
        else:
            temporaldata=np.loadtxt(text,dtype=float,delimiter=",")
            dataframe[filenumber]=pd.Series(temporaldata)
            #dataframe=dataframe.append(temporaldata)
            filenumber+=1
    ####calculating mean and meadian###
    df_mean=dataframe.mean(axis=1)
    df_sd=dataframe.std(axis=1) 
    print("calculating "+str(n)+" new stimuli from the original ones")
    for i in range(0,obs):
        if i==0:
            if normaldist==True:
               array=np.random.normal(df_mean[i],df_sd[i],size=n)
            else:
                times=np.random.uniform(-.1,.1,n)
                list_=[]
                for j in range(0,n):
                    list_.append(df_mean[i]*times[j])
                array=np.array(list_)
        else:
            if normaldist==True:
               temp_array=np.random.normal(df_mean[i],df_sd[i],size=n)
               array=np.vstack((array,temp_array))
            else:
                times=np.random.uniform(-.1,.1,n)
                list_=[]
                for j in range(0,n):
                    list_.append(df_mean[i]*times[j])
                temp_array=np.array(list_)
                array=np.vstack((array,temp_array))
    array=pd.DataFrame(array)
    dataMatrix=pd.concat([array,dataframe],axis=1,sort=False)
    dataMatrix=dataMatrix.T
    if saveplain==True:
        import os
        print("saving artificial stimuli from gaussian distribution")
        output_path="./output/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        document=output_path+documentofinal
        dataMatrix.to_csv(document, index=False,header=False)
if __name__=="__main__":
    main()