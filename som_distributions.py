#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:04:21 2020

@author: marco
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
def charge_som(som_text):
    area,dim=data_som.shape
    lado=np.sqrt(area)
    som=np.empty(shape=(int(lado),int(lado),int(dim)),dtype=float)
    index=0
    for i in range(int(lado)):
        for j in range(int(lado)):
            som[i][j]=np.array(data_som[index])
            index+=1
    return som,dim,lado

def dist(v1, v2):
    return math.sqrt(sum((v1 - v2) ** 2))
def closest_node(stimuli, map, m_rows):
    result=(0,0)
    small_dist = dist(stimuli,map[0][0])#número inicial de distancia pequeña
    for i in list(range(0,int(m_rows))):
      for j in list(range(0,int(m_rows))):
        ed = dist(map[i][j], stimuli)
        if ed < small_dist:# compra distancia euclideana para saber su fitness
          small_dist = ed# si sí, la menor distancia se reescribe
          result = (i, j)# si sí, resultado se sobrescriba con el nodo + fit
    return result
def normal(x,g):
    return x/g
def axis_dict(labelsList,somaxissize):
    dictname={}
    for labels in (labelsList):
        dictname[labels]={}
        for coord in list(range(0,int(somaxissize))):
            dictname[labels][coord]=0
    return dictname
def plothist(axis_dictionary,labelslist,axis):
    for i in labelslist:
        plt.bar(list(axis_dictionary[i].keys()),axis_dictionary[i].values(),color='g')
        legend="histrogram for axis "+str(axis)+" for label "+str(i)
        plt.title(legend)
        plt.show()
def dict2list(dictionary,key):
    mylist=list(dictionary[key].values())
    return mylist

def plotdist(list_ocurr,plotname):    
    mu = np.mean(list_ocurr)
    sigma = np.mean(list_ocurr)
    statmin=min(list_ocurr)
    statmax=max(list_ocurr)
    zmin = ( statmin - mu ) / sigma
    zmax = ( statmax - mu ) / sigma
    x = np.arange(zmin, zmax, 0.001) 
    x_all = np.arange(-10, 10, 0.001)
    y = norm.pdf(x,0,1)
    y2 = norm.pdf(x_all,0,1)
    fig, ax = plt.subplots(figsize=(9,6))
    plt.style.use('fivethirtyeight')
    ax.plot(x_all,y2)

    ax.fill_between(x,y,0, alpha=0.3, color='b')
    ax.fill_between(x_all,y2,0, alpha=0.1)
    ax.set_xlim([-4,4])
    ax.set_xlabel(plotname+' # of Standard Deviations Outside the Mean')
    ax.set_yticklabels([])
    ax.set_title('Normal Gaussian Curve')
    #plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
    plt.show()

def z_comparition (lt):
    ocurr_array=np.array(lt)
    median=np.median(ocurr_array)
    mean=np.mean(ocurr_array)
    sd=np.std(ocurr_array)
    z=(median-mean)/sd
    return z
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.scatter(x, y)
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    
def scatter_z(x, y,syl):
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    plt.title("scatter z "+str(syl))
    plt.show()
def ocurrences(data_stimuli,map,lado,labels_size,labels):
    result_x=axis_dict([1,2],lado)
    result_y=axis_dict([1,2],lado)
    ocurr_listx1=[]
    ocurr_listX2=[]
    ocurr_listy1=[]
    ocurr_listy2=[]
    for i in list(range(0,labels_size)):
        winner=closest_node(data_stimuli[i], map, lado)
        key=labels[i]
        x,y=winner
        if key==1:
           ocurr_listx1.append(x)
           ocurr_listy1.append(y)
        else:
           ocurr_listX2.append(x)
           ocurr_listy2.append(y)
        result_x[key][x]+=1
        result_y[key][y]+=1
    return result_x, result_y,ocurr_listx1, ocurr_listy1, ocurr_listX2, ocurr_listy2

def divergencekl(listocurr_p,listocurr_q):
    ocurr_p=listocurr_p
    ocurr_q=listocurr_q
    n_p=len(ocurr_p)
    n_q=len(ocurr_q)
    for i in list(range(0,n_p)):
        if ocurr_p[i] !=0: 
            ocurr_p[i]=ocurr_p[i]/n_p
        else:
            ocurr_p[i]=0
    for i in list(range(0,n_p)):
        if ocurr_q[i] !=0: 
            ocurr_q[i]=ocurr_q[i]/n_q
        else:
            ocurr_q[i]=0
    for i in list(range(0,n_p)):
        if ocurr_p[i]==0:
            ocurr_p[i]=1
    for i in list(range(0,n_q)):
        if ocurr_q[i]==0:
            ocurr_q[i]=1
    ocurr_p=np.log2(ocurr_p)
    ocurr_q=np.log2(ocurr_q)
    dkl=0
    for i in list(range(0,n_p)):
        dkl+=ocurr_p[i]-ocurr_q[i]
    return dkl
def sctter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=True)
    ax_histy.tick_params(axis="y", labelleft=True)
    ax.scatter(x, y)
    ax_histx.hist(x, bins=10)
    ax_histy.hist(y, bins=10, orientation='horizontal')
    return ax_histx, ax_histy,ax

###############################main############################################
data_file = "/media/marco/MarcoHDD/github/SOM/SOMoutput/SOM_1000_noise2_gaussianRaw_.txt"
data_som = np.loadtxt(data_file, delimiter=",",
                      dtype=np.floating)
map,dim,lado=charge_som(data_som)
data_stimulifile = "/media/marco/MarcoHDD/github/LipReading/output/ArtificialDescriptorBa2_gaussian.csv"
data_stimuli = np.loadtxt(data_stimulifile, delimiter=",", usecols=range(0,dim),
                    dtype=np.floating)# vector por sujeto
data_stimulifileo = "/media/marco/MarcoHDD/github/LipReading/output/Descriptor_baga_original.csv"
data_stimulio = np.loadtxt(data_stimulifileo, delimiter=",", usecols=range(0,dim),
                    dtype=np.floating)# vector por sujeto
data_stimulifilea = "/media/marco/MarcoHDD/github/LipReading/output/Descriptor_baga_artificial.csv"
data_stimulia = np.loadtxt(data_stimulifilea, delimiter=",", usecols=range(0,dim),
                    dtype=np.floating)# vector por sujeto
normalize=False
if normalize==True:
    maxim=np.amax(data_stimuli)
    data_stimuli=normal(data_stimuli,maxim)
    


labels = np.loadtxt(data_stimulifile, delimiter=",", usecols=(dim),dtype=np.int)
_x_=labels.size
labelso = np.loadtxt(data_stimulifileo, delimiter=",", usecols=(dim),dtype=np.int)
_xo_=labelso.size
labelsa = np.loadtxt(data_stimulifilea, delimiter=",", usecols=(dim),dtype=np.int)
_xa_=labelsa.size
result_x, result_y,ocurr_listx1, ocurr_listy1, ocurr_listX2, ocurr_listy2= ocurrences(data_stimuli,map,lado,_x_,labels)
result_xo, result_yo,ocurr_listx1o, ocurr_listy1o, ocurr_listX2o, ocurr_listy2o= ocurrences(data_stimulio,map,10,_xo_,labelso)
result_xa, result_ya,ocurr_listx1a, ocurr_listy1a, ocurr_listX2a, ocurr_listy2a= ocurrences(data_stimulia,map,10,_xa_,labelsa)



plothist(result_x,[1,2],"x")
plothist(result_y,[1,2],"y")    
x_ba=dict2list(result_x,1)
x_ga=dict2list(result_x,2)
y_ba=dict2list(result_y,1)
y_ga=dict2list(result_y,2)
x_bao=dict2list(result_xo,1)
x_gao=dict2list(result_xo,2)
y_bao=dict2list(result_yo,1)
y_gao=dict2list(result_yo,2)
x_baa=dict2list(result_xa,1)
x_gaa=dict2list(result_xa,2)
y_baa=dict2list(result_ya,1)
y_gaa=dict2list(result_ya,2)
plotdist(ocurr_listx1,"x_ba")
plotdist(ocurr_listy1,"y_ba")
plotdist(ocurr_listX2,"x_ga")
plotdist(ocurr_listy2,"x_ga")
zx1=z_comparition(ocurr_listx1)
zx2=z_comparition(ocurr_listX2)
zy1=z_comparition(ocurr_listy1)
zY2=z_comparition(ocurr_listy2)

scatter_z(zx1,zy1,'ba')
scatter_z(zx2,zy1,'ga')

def plot_sctter(ocurr_listX2, ocurr_listy2,name):
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.06
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.3]
    rect_histy = [left + width + spacing, bottom, 0.3, height]
# start with a square Figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx)
    ax_histy = fig.add_axes(rect_histy)
# use the previously defined function
    ax_histx, ax_histy,ax=sctter_hist(ocurr_listX2o, ocurr_listy2o, ax, ax_histx, ax_histy)
    plt.title("scatter plot for "+str(name))
    plt.show()

klbax=divergencekl(x_bao,x_ba)
klbay=divergencekl(y_bao,y_ba)
klgax=divergencekl(x_gao,x_ga)
klgay=divergencekl(y_gao,y_ga)
print(klbax,klbay,klgax,klgay)

plot_sctter(ocurr_listX2o, ocurr_listy2o, name="empirical ga stimuli")
plot_sctter(ocurr_listx1o, ocurr_listy1o, name="empirical ba stimuli")
name="empirical ga stimuli"
