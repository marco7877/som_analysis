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
def closest_node(stimuli, map, m_rows, m_cols):
    result=(0,0)
    small_dist = dist(stimuli,map[0][0])#número inicial de distancia pequeña
    for i in range(int(m_rows)):
      for j in range(int(m_cols)):
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
normalize=False
data_file = "/media/marco/MarcoHDD/github/SOM/SOMoutput/SOM_1000_noise2_gaussianRaw_.txt"
data_som = np.loadtxt(data_file, delimiter=",",
                      dtype=np.floating)
map,dim,lado=charge_som(data_som)
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
data_stimulifile = "/media/marco/MarcoHDD/github/LipReading/output/ArtificialDescriptorBa2_gaussian.csv"
data_stimuli = np.loadtxt(data_stimulifile, delimiter=",", usecols=range(0,dim),
                    dtype=np.floating)# vector por sujeto

def z_comparition (lt):
    ocurr_array=np.array(lt)
    median=np.median(ocurr_array)
    mean=np.mean(ocurr_array)
    sd=np.std(ocurr_array)
    z=(median-mean)/sd
    return z
if normalize==True:
    maxim=np.amax(data_stimuli)
    data_stimuli=normal(data_stimuli,maxim)
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
    plt.title("scatter z "+syl)
    plt.show()
    
labels = np.loadtxt(data_stimulifile, delimiter=",", usecols=(dim),dtype=np.int)
_x_=labels.size
result_x=axis_dict([1,2],lado)
result_y=axis_dict([1,2],lado)
ocurr_x={1:[],2:[]}
ocurr_y={1:[],2:[]}
ocurr_listx1=[]
ocurr_listX2=[]
ocurr_listy1=[]
ocurr_listy2=[]
for i in list(range(0,_x_)):
    winner=closest_node(data_stimuli[i], map, lado, lado)
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
    

plothist(result_x,[1,2],"x")
plothist(result_y,[1,2],"y")    
x_ba=dict2list(result_x,1)
x_ga=dict2list(result_x,2)
y_ba=dict2list(result_y,1)
y_ga=dict2list(result_y,2)
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


'''
hist, bin_edges=np.histogram(ocurr_x,bins=10)
import matplotlib.pyplot as plt
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.show()
'''
def sctter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.scatter(x, y)
    binwidth = 0.4
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
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
sctter_hist(ocurr_listx1, ocurr_listy1, ax, ax_histx, ax_histy)
plt.title("som ba activation")
plt.show()