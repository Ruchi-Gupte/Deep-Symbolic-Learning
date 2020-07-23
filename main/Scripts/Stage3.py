import numpy as np
import time    
import pickle
from PIL import Image
import cv2

def Stage3(config,stage3_load_condition):
    load_path=config['load_path']
    model = pickle.load(open('model'+load_path+'.sav', 'rb'))
    test_data=np.load('Array_of_test_images'+load_path+'.npy',allow_pickle = True )
    
    centroids = model.cluster_centers_
    no_of_sampleimages=config['sampleimages']
    widthi=config['image_length']
    heighti=config['image_height']
    padding=config['padding']
    patch_size=config['window_size']
    shift=config['Stride']
    patchl=config['patchl']
    patchh=config['patchh']
    
    #assigning symbolic ID for all test patches
    if stage3_load_condition==1:
        print("Loading existing symbolic list")
        test_symboliclist=np.load('testdata_symboliclist'+load_path+'.npy')
        
    else:
        test_symboliclist=[]
        time_start = time.time()
        for testno in range(no_of_sampleimages):
            X_reconstructed=[]
            print(testno)
            image=test_data[testno]  #to select an image number
            for i in range(0,heighti+padding[0]+padding[1]-(patch_size-1),shift):
                for j in range(0,widthi+padding[2]+padding[3]-(patch_size-1),shift):
                    BLOCK=image[i:i+patch_size,j:j+patch_size]
                    v=model.predict(BLOCK.reshape(1, -1))
                    X_reconstructed.append(v[0]+1)
            test_symboliclist.append(X_reconstructed)
        print("Symbolic List Created. Time elapsed: {} minutes".format((time.time()-time_start)/60))
    
    
    
    for testno in range(no_of_sampleimages):
        test_symboliclist[testno]=np.array(test_symboliclist[testno])
        test_symboliclist[testno]=test_symboliclist[testno].reshape(patchl,patchh)
    
    X_reconstructed=test_symboliclist[1].reshape(patchl*patchh,)
    #X_reconstructed=test_images[1].reshape(patchl*patchh,)
    Background_white = [255,255,255]
    
    final_image = Image.new('RGBA', (patchl*100, patchh*100), (255, 255, 255, 255))
    
    k=0
    for i in range(0,patchh*100,100):
        for j in range(0,patchl*100,100):
            Centroid=centroids[X_reconstructed[k]-1].reshape(patch_size,patch_size)
            temp_image=cv2.copyMakeBorder(Centroid,0,1,0,1,cv2.BORDER_CONSTANT,value=Background_white)
            temp_image = Image.fromarray(temp_image)
            temp_image = temp_image.resize((100,100), resample=Image.NEAREST)
            k=k+1
            final_image.paste(temp_image, (j,i))
                   
    final_image.show()
    
    np.save('testdata_symboliclist'+load_path+'.npy', test_symboliclist)
    print("Stage 3 Completed Succesfully!")
