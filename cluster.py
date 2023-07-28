import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler 
df=pd.read_excel('data.xlsx')
X= df["x"].values.tolist()
Y= df["y"].values.tolist()
data = list(zip(X, Y))
Cluster = {}
def kmeans(df,n):
    km=KMeans(n_clusters=n)
    y_prediction=km.fit_predict(df[['x','y']])
    for i in range(len(y_prediction)):
        y_prediction[i]+=1
    df['cluster'] = y_prediction
    df.sort_values('cluster', inplace=True) 
    C = {}
    j = 1
    for i in df['cluster'].unique():
        df1 = df.loc[df['cluster'] == i]
        X= df1["x"].values.tolist()
        Y= df1["y"].values.tolist()
        l1 = list(zip(X,Y))
        C['C'+str(j)]= l1
        j+=1
    return C
def Detection_Aberrant(X,Y):
    cluster = {}
    moyennex = np.mean(X)
    ecartx = np.std(X)
    moyenney = np.mean(Y)
    ecarty = np.std(Y) 
    les_x = []
    les_y = []   
    for i in range(len(X)):
        zx = (X[i]-moyennex)/ecartx
        zy = (Y[i]-moyenney)/ecarty
        les_x.append(zx)
        les_y.append(zy)   
    liste_a_sup = []
    for i in range(len(les_x)):
        if les_x[i]<-3 or les_x[i]>3 or les_y[i]<-3 or les_y[i] > 3:
            liste_a_sup.append(i)     
    for i in range(len(liste_a_sup)):
        cluster['C'+str(i+1)] = []
        cluster['C'+str(i+1)].append([X[liste_a_sup[i]],Y[liste_a_sup[i]]])
        X.pop(liste_a_sup[i])
        Y.pop(liste_a_sup[i])
    return cluster
#Cluster = Detection_Aberrant(X,Y)
def distance_eucliduinne(data):
    distance = {}
    for i in range(len(data)):
        for j in range(len(data)):
            if i < j:
                ind = (data[i],data[j])
                dist = math.dist(data[i], data[j])           
                distance[ind]= dist
    return distance                      
data = list(zip(X,Y))
def DistanceMoyenne(data):
    dis = []
    distance = distance_eucliduinne(data)
    for item in distance.values():
        dis.append(item)  
    return np.mean(dis)
def isEmpty(dictionary):
    for element in dictionary:
        if element:
            return False
    return True
def dist(x,y):
    return math.sqrt((y[0]-x[0])**2+((y[0]-x[0]))**2)
def Centre_gravite_cluster(cluster):
    CG = []
    for i in cluster.values():
        if len(i)>=2:
            x=0
            y=0
            for j in i:
                x+=j[0]
                y+=j[1]
            x*=(1/len(i))
            y*=(1/len(i)) 
            CG.append([x,y])
        else:
            CG.append(i[0])
    return CG
def distance_inter(L1):
    Dist = {}
    Distances = []
    for i in range(0,len(L1)):
        for j in range(0,len(L1)):
            if i<j:            
                ind = 'C'+str(i+1),'C'+str(j+1)
                Dist[ind]=math.dist(L1[i],L1[j]) 
    for k,v in Dist.items():
        Distances.append([k,v])
    return Distances
def distance_intra(L1):
    distance = []
    for i in range(len(L1)):
        for j in range(len(L1)):
            if i < j:
                dist = math.dist(L1[i], L1[j])           
                distance.append(dist)
    for i in distance:
        return np.mean(distance)      
def les_distance_intra(cluster):
    i=0
    les_distances = []
    for item in cluster:
        i+=1
        if len(cluster[item]) > 1:
            les_distances.append(['C'+str(i),distance_intra(cluster[item])])
    return les_distances
def ClusteringIteratif(data,cluster):
    ClusterKeyList = []
    CentreGravite = {}
    if not isEmpty(cluster):
        j = len(cluster.keys())
        for i in range(1,j+1):
            ClusterKeyList.append(i)
        for item in cluster:
            CentreGravite[item] = cluster[item]
    else:
        j = 0 
        ClusterKeyList.append(j)       
    distance_moyenne = DistanceMoyenne(data)  
    a_sup = []  
    for i in data:  
        ajou = False                 
        for item in CentreGravite.copy():                
            if dist(i, CentreGravite[item][0]) < distance_moyenne/2:
                elmne = list(i)
                CentreGravite[item].append(elmne)
                cluster[item].append(elmne)      
                ajou = True    
        if ajou == False:
            j = int(ClusterKeyList[-1]) +1
            cle = 'C'+str(j)
            cluster[cle] = []
            el = list(i)
            cluster[cle].append(el)     
            CentreGravite[cle]= []
            CentreGravite[cle].append(list(i))
            ClusterKeyList.append(j)     
            for item in CentreGravite:
                if len(CentreGravite[item])>1:
                    CentreGravite[item] = Centre_gravite_cluster({item:CentreGravite[item]})    
    return cluster
Cluster = ClusteringIteratif(data,Cluster)
print(Cluster)

def affichage_cluster(cluster):
    print('Les clusters sont:\n')
    for k,v in cluster.items():
        print(k,':',v)
affichage_cluster(Cluster)

print('\nLes distances intra sont:\n')
print(les_distance_intra(Cluster))

print('\nLes distances inter sont:\n')
print((distance_inter(Centre_gravite_cluster(Cluster))))









