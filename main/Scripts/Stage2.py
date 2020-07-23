
import numpy as np
import time
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import pickle

def Sort(sub_li): #Sorting a list according to ascending order
    sub_li.sort(key = lambda x: x[1]) 
    return sub_li 

def Stage2(config,stage2_load_condition): 
    load_path=config['load_path']
    patchlist = np.load('patchlist'+load_path+'.npy', allow_pickle=True)
    patch_size=config['window_size']
   
    if stage2_load_condition==1:
        print("Loading model from file")
        model = pickle.load(open('model'+load_path+'.sav', 'rb'))
        n_clusters=config['n_clusters']
        labels=model.labels_
        centroids = model.cluster_centers_
        
    else:
        if config['clustering_model']=='KMEANS':
            time_start = time.time()
            n_clusters=config['n_clusters']
            model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm="auto")
            model=model.fit(patchlist.reshape(len(patchlist),-1))
            labels=model.labels_
            centroids = model.cluster_centers_
            print("Kmeans Done. Time elapsed: {} minutes".format((time.time()-time_start)/60))
            print(time.time()-time_start)
        
        elif config['clustering_model']=='MEANSHIFT':
            time_start = time.time()
            bandwidth=config['bandwidth']
            model = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True, min_bin_freq=1, n_jobs=1, seeds= None)
            model=model.fit(patchlist.reshape(len(patchlist),-1))
            labels = model.labels_
            centroids = model.cluster_centers_
            labels_unique = np.unique(labels)
            n_clusters = len(labels_unique)
            config.update( {'n_clusters' : n_clusters} )
            print ("MeanShift Done. Time elapsed: {} minutes".format((time.time()-time_start)/60))
            print(time.time()-time_start)
    
    k=0
    clusterdata=[]
    for clus in range(n_clusters):
        for i in patchlist[np.where(labels==clus)]:
            k=k+1
        temp=[clus+1,k]
        clusterdata.append(temp)   
        k=0 #comment to check cumulative density 
    clusterdata=Sort(clusterdata)
    
    temp=[]
    clustack=[]
    for clus in range(n_clusters):
        for i in patchlist[np.where(labels==clus)]:
            temp.append(i)
        clustack.append(np.array(temp))
        temp=[]
    clustack=np.array(clustack)
    
    dist=[]
    for Clusterno in range(n_clusters):
        temp=[]
        Cent=centroids[Clusterno].reshape(patch_size,patch_size)
        k=0
        for m in clustack[Clusterno]:
            distance = np.linalg.norm(m-Cent)
            distance=[int(k),int(distance)]
            temp.append(distance)
            k=k+1
        dist.append(temp)
    for clus in range(n_clusters):
        dist[clus]=np.array(Sort(dist[clus]))
    
    Rows=10  #how many rows and columns in collage
    Columns=10
    
    #Making Collage
    cluster_no=4 # cluster number
    Background_white = [255,255,255]
    final_collage = Image.new('RGBA', (1000, 1000), (255, 255,255))
    k=0
    
    for i in range(0,Columns*100,100):
        for j in range(0,Rows*100,100):
            temp_image=cv2.copyMakeBorder(clustack[cluster_no-1][int(dist[cluster_no-1][k][0])],0,1,0,1,cv2.BORDER_CONSTANT,value=Background_white)
            temp_image = Image.fromarray(temp_image, 'L')
            temp_image = temp_image.resize((100,100), resample=Image.NEAREST)
            k=k+1
            final_collage.paste(temp_image, (j,i))
    
    final_collage.show()
    
    
    
    np.save('cluster_density'+load_path+'.npy', clusterdata)
    np.save('patch_with_clusters'+load_path+'.npy', clustack)
    np.save('distance_info_for_clusters'+load_path+'.npy', dist)
    pickle.dump(model, open('model'+load_path+'.sav', 'wb'))
    print("Stage 2 Completed Succesfully!")