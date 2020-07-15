import math
import numpy as np
from PIL import Image
import cv2
import pickle
import json


def cos_square(x): 
    ans=math.cos(x)*math.cos(x)
    return ans

def sin_square(x): 
    ans=math.sin(x)*math.sin(x)
    return ans

def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()


def Stage4(config):
    load_path=config['load_path']
    model = pickle.load(open('model'+load_path+'.sav', 'rb'))
    no_of_sampleimages=config['sampleimages']
    test_skip=config["test_skip"]
    symbolic_list=np.load('testdata_symboliclist'+load_path+'.npy')
    patchlength=config['patchl']
    patchheight=config['patchh']
    patch_size=config['window_size']
    neighbor_range=config['neighbor_range']
    centroids = model.cluster_centers_
   

    test_images=[]
    image=[]
    for num in range(no_of_sampleimages):
        image=symbolic_list[num]
        patchl=0
        patchh=0
        temp=[]
        for i in range(0,patchheight,test_skip):
            for j in range(0,patchlength,test_skip):
                temp.append(image[i][j])
                patchl=patchl+1
            patchh=patchh+1
        
        patchl=int(patchl/patchh)
        temp=np.array(temp)
        temp=temp.reshape(patchh,patchl)
        test_images.append(temp)
    
    arrayorder=[]
    k=0
    for y2 in range(patchh):
        for x2 in range(patchl):
            arrayorder.append(k)
            k=k+1
    arrayorder=np.array(arrayorder)
    arrayorder=arrayorder.reshape(patchh,patchl)
    
    #make patchh,patchl 5 if we change the dimension of input matrix
    weights=[]
    for y1 in range(patchh):
        for x1 in range(patchl):
            temp=[]
            for y2 in range(patchh):
                for x2 in range(patchl):
                    if y2>y1 and x1==x2:
                        x=90 #pure north
                    elif y2<y1 and x1==x2:
                        x=-90 #pure south
                    elif y2==y1 and x2==x1:
                        x=0
                    elif x1>x2 and y1==y2:
                        x=404   #pure west
                    elif x2>x1 and y1==y2:
                        x=303  #pure east
                    else:
                        slope = ((y2-y1)/(x2-x1))
                        x=math.atan(slope)
                    temp.append([int(x1),int(y1),int(x2),int(y2),x])
            weights.append(temp)
    #getting angles for each element[point1,point2==>angle]
    
    values=[]
    l=0
    for y1 in range(patchh):
        for x1 in range(patchl):
            k=0
            element=[]
            for y2 in range(patchh):
                for x2 in range(patchl):
                    angle=weights[l][k][4] #radians
                    if angle==90:
                        ans=[1,0,0,0] #north
                    elif angle==-90:
                        ans=[0,1,0,0] #south
                    elif angle==404:
                        ans=[0,0,1,0] #east
                    elif angle==303:
                        ans=[0,0,0,1] #west
                    elif x1==x2 and y1==y2:
                        ans=[0,0,0,0] #samepoint       
                    elif x1>x2 and y1>y2:
                        ans=[0,sin_square(angle),0,cos_square(angle)] #southwest
                    elif x1>x2 and y1<y2:
                        ans=[sin_square(angle),0,0,cos_square(angle)] #northwest
                    elif x1<x2 and y1>y2:
                        ans=[0,sin_square(angle),cos_square(angle),0] #southeast
                    elif x1<x2 and y1<y2:
                        ans=[sin_square(angle),0,cos_square(angle),0] #northeast
                    else:
                        ans=4040404 #error
                    element.append([x1,y1,x2,y2,ans]) #all relations for one point
                    k=k+1 #increment to next point
            #print(4040404 in element[4])
            values.append(element)  
            l=l+1 #increment to next pivot point
    
    if neighbor_range!="NA":
        final=[]
        c=0
        for y1 in range(patchh):
            for x1 in range(patchl):
                ans=cell_neighbors(arrayorder,y1,x1,neighbor_range)
                standby=[]
                for i in range(len(ans)):
                    standby.append(values[c][ans[i]])
                final.append(standby) #final gives only those points within neighbor distance
                c=c+1
    else:
        final=values
    
    X_reconstructed=test_images[1].reshape(patchl*patchh,)
    #X_reconstructed=test_images[1].reshape(patchl*patchh,)
    Background_white = [255,255,255]
    
    final_image = Image.new('RGBA', (patchh*100, patchl*100), (255, 255, 255, 255))
    
    k=0
    for i in range(0,patchh*100,100):
        for j in range(0,patchl*100,100):
            Centroid=centroids[X_reconstructed[k]-1].reshape(patch_size,patch_size)
            temp_image=cv2.copyMakeBorder(Centroid,0,0,0,0,cv2.BORDER_CONSTANT,value=Background_white)
            temp_image = Image.fromarray(temp_image)
            temp_image = temp_image.resize((100,100), resample=Image.NEAREST)
            k=k+1
            final_image.paste(temp_image, (j,i))
                   
    final_image.show()
    
    
    np.save('AllHardcoded_directionalmatrix'+load_path+'.npy', values)
    np.save('New_symbolic_list'+load_path+'.npy', test_images)
    np.save('Selected_Hardcoded_directionalmatrix'+load_path+'.npy', final)
    config.update( {'stage4_patchl' : patchl,
                    'stage4_patchh' : patchh} )
    with open('config.json', 'w') as f:
            json.dump(config, f)
    print("Stage 4 Completed Succesfully!")