import numpy as np
import json

def Stage1(config): 
    data = np.load('Array_of_training_images1.npy')
    widthi=heighti=config['image_size']
    no_of_sampleimages=config['sampleimages']
    patch_size=config['window_size']
    shift=config['Stride']
    padding=config['padding']
    
    X_im=[]
    for k in range(no_of_sampleimages):
        patchl=0
        patchh=0
        image=data[k]
        for i in range(0,heighti+padding[0]+padding[1]-(patch_size-1),shift):
            for j in range(0,widthi+padding[2]+padding[3]-(patch_size-1),shift):
                BLOCK=image[i:i+patch_size,j:j+patch_size]
                X_im.append(BLOCK) 
                patchl=patchl+1
            patchh=patchh+1        
    patchl=int(patchl/patchh)
    X_im = np.array(X_im)
    
    
    config.update( {'patchl' : patchl, #patch length
                    'patchh' : patchh} ) #patch height
    with open('config.json', 'w') as f:
            json.dump(config, f)
    np.save('patchlist1.npy', X_im)
    print("Stage 1 Completed Succesfully!")