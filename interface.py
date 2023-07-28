from distutils.command.upload import upload
from tkinter import filedialog
import numpy as np 
import pandas as pd
import tkinter as tk 
from tkinter import *
from tkinter import filedialog
from setuptools import Command
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler 
import cluster as cl 
from PIL import ImageTk,Image
my_w=tk.Tk()
my_w.geometry("1200x800")
my_w.configure(background='#dee2e6')
titre=Label(my_w,text="Clustering Iteratif",background='#dee2e6',fg='#5c162e',font=("Javanese Text",43))
titre.place(x=360,y=0)
l1=Label(my_w,text="Les Clusters:",background='#dee2e6',fg='#5c162e',font=("Javanese Text",17))
l1.place(x=50,y=204)
l2=Label(my_w,text="Distances Intra:",background='#dee2e6',fg='#5c162e',font=("Javanese Text",17))
l2.place(x=35,y=330)
l3=Label(my_w,text="Distances Inter:",background='#dee2e6',fg='#5c162e',font=("Javanese Text",17))
l3.place(x=35,y=456)
l4=Label(my_w,text="Nombres de Clusters:",background='#dee2e6',fg='#5c162e',font=("Javanese Text",17))
l4.place(x=110,y=550)
l5=Label(my_w,text="Les Clusters:",background='#dee2e6',fg='#5c162e',font=("Javanese Text",17))
l5.place(x=50,y=643)
b1=tk.Button(my_w,text="Get Clusters",width=10,background='#5c162e',fg='#ffffff',font=("italique",12),command=lambda:do_clus(file))
b1.place(x=1044,y=340)
b2=tk.Button(my_w,text="Statistics",width=10,background='#5c162e',fg='#ffffff',font=("italique",12),command=lambda:plot(file))
b2.place(x=550,y=735)
t1=tk.Text(my_w,height=4,width=100,background='#f6f5f5')
t1.place(x=200,y=200)
t2=tk.Text(my_w,height=4,width=100,background='#f6f5f5')
t2.place(x=200,y=320)
t3=tk.Text(my_w,height=4,width=100,background='#f6f5f5')
t3.place(x=200,y=440)
b3=tk.Button(my_w,text="Get Data",width=10,background='#5c162e',fg='#ffffff',font=("italique",12),command=lambda:Choix_File())
b3.place(x=550,y=130)
b4=tk.Button(my_w,text="keamns",width=10,background='#5c162e',fg='#ffffff',font=("italique",12),command=lambda:K_means(file,nb_cluster))
b4.place(x=1044,y=660)
t4=tk.Text(my_w,height=4,width=100,background='#f6f5f5')
t4.place(x=200,y=640)
b5=tk.Button(my_w,text="Valider",width=10,background='#5c162e',fg='#ffffff',font=("italique",12),command=lambda:get_n())
b5.place(x=700,y=561)
t5=tk.Text(my_w,height=1.7,width=32,background='#f6f5f5',font=(14))
t5.place(x=350,y=560)
def get_n():
    global nb_cluster
    n = t5.get("1.0","end-1c")
    nb_cluster = int(n)
    return nb_cluster
def K_means(file,n):
    df=pd.read_excel('data.xlsx')
    X= df["x"].values.tolist()
    Y= df["y"].values.tolist()
    C = cl.kmeans(df,n)
    for k,v in C.items():
        t4.insert(tk.END,k,":",v)
        t4.insert(tk.END,"\n")  
def Choix_File():
    global file
    file = filedialog.askopenfilename(filetypes=[("Excel file",'.xlsx')])
    return file
def do_clus(file):
    df=pd.read_excel(file)
    X= df["x"].values.tolist()
    Y= df["y"].values.tolist()
    data = list(zip(X,Y))
    Cluster = {}
    data = list(zip(X,Y))
    cus=cl.ClusteringIteratif(data,Cluster)
    for k,v in cus.items():
        t1.insert(tk.END,k,":",v)
        t1.insert(tk.END,"\n")
    t2.insert(tk.END,cl.les_distance_intra(Cluster))
    t3.insert(tk.END,cl.distance_inter(cl.Centre_gravite_cluster(Cluster)))
def plot(file):
    figure(figsize=(15,7.5))
    df=pd.read_excel(file)
    X= df["x"].values.tolist()
    Y= df["y"].values.tolist()
    data = list(zip(X,Y))
    p1=plt.subplot(3,2,1)
    plt.scatter(X,Y,color="#7c162e")
    plt.plot()
    Cluster={}
    #Cluster=cl.Detection_Aberrant(X,Y)
    data = list(zip(X,Y))
    cluster=cl.ClusteringIteratif(data,Cluster)
    l=cl.les_distance_intra(cluster)
    l2=cl.distance_inter(cl.Centre_gravite_cluster(cluster))
    C = cl.kmeans(df,nb_cluster)
    lk = cl.distance_inter(cl.Centre_gravite_cluster(C))
    lkk=cl.les_distance_intra(C)
    p2=plt.subplot(3,2,3)
    for i in range(len(l)):
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.bar(l[i][0],l[i][1],color=col)
    p6=plt.subplot(3,2,5)
    for i in range(len(lkk)):
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.bar(lkk[i][0],lkk[i][1],color=col)
    xx=[]
    yy=[]
    for i in range(len(l2)):
      xx.append(l2[i][0])
      yy.append(l2[i][1])
    po=[]
    for i in range(len(xx)):
        po.append(','.join(xx[i]))

    x1 = np.arange(len(po))
    p3=plt.subplot(3,2,4)
    a=plt.bar(x1,yy,color='#84a59d',width=0.4)
    plt.xticks(x1, po)
    xxk=[]
    yyk=[]
    for i in range(len(lk)):
        xxk.append(lk[i][0])
        yyk.append(lk[i][1])
    pok=[]
    for i in range(len(xxk)):
        pok.append(','.join(xxk[i]))
    x1k = np.arange(len(pok))
    p5=plt.subplot(3,2,6)
    b=plt.bar(x1k,yyk,color='#f28482',width=0.4)
    plt.xticks(x1k, pok)
    for item in cluster.values():
        print(item)
        if len(item)>=2:
            x=[]
            y=[]
            for i in range(len(item)):
                x.append(item[i][0])
                y.append(item[i][1])
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.subplot(3,2,2)
            plt.scatter(x,y,color=col)
        else :
            x=[]
            y=[]
            for i in range(len(item)):
                x=item[i][0]
                y=item[i][1]
                p4=plt.subplot(3,2,2)
                plt.scatter(x,y,c=np.random.rand(3,))
    p1.set_title("Nuage des points")
    p2.set_title("Distance Intra Cluster")
    p3.set_title("Distance Inter Cluster")
    p4.set_title("Les Clusters")
    p5.set_title("Distance Inter K-means")
    p6.set_title("Distance Intra K-means")
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    plt.show()  
my_w.mainloop()