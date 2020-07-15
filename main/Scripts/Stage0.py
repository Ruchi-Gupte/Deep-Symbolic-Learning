import numpy as np
import glob, idx2numpy, cv2, sklearn
#import random


def Stage0(config): 
    load_path=config['load_path']
    padding=config['padding'] #Padding per image incase of uneven strides
    no_of_sampleimages=config['sampleimages']
    type_dataset=config['dataset_format']  #The path and file type of dataset
    
    
    if type_dataset=='folder': #Incase dataset is a set of images in a folder
        final_image_list = []
        for img in glob.glob(config['genral_path_incase_of_folder']+'/*.jpg*'):
            temp_image= cv2.imread(img,0)
            final_image_list.append(temp_image)
        #random.shuffle(final_image_list) #Incase you want to shuffle dataset
        testdata, traindata = sklearn.model_selection.train_test_split(final_image_list, train_size=0.4, test_size=0.6)

    elif type_dataset=='ubyte': #In case of MNIS, images are stored in ubyte folder
        traindata = idx2numpy.convert_from_file(config['training_image_path']) 
        testdata = idx2numpy.convert_from_file(config['test_image_path']) 
    
    else:
        print("Error in Reading Images in Stage 0")
    
    traindata = traindata[:no_of_sampleimages] #Taking only certain set of images
    testdata = testdata[:no_of_sampleimages] 
    
    final_train_data=[0]*no_of_sampleimages #Inserting padding to each image
    for k in range(no_of_sampleimages):
        final_train_data[k]=np.pad(traindata[k], [(padding[0], padding[1]), (padding[2], padding[3])], mode='constant', constant_values=0)
        
    final_test_data=[0]*no_of_sampleimages  #Inserting padding to each image
    for k in range(no_of_sampleimages):
        final_test_data[k]=np.pad(traindata[k], [(padding[0], padding[1]), (padding[2], padding[3])], mode='constant', constant_values=0)
    
    np.save('Array_of_training_images'+load_path+'.npy', final_train_data)
    np.save('Array_of_test_images'+load_path+'.npy', final_test_data)
    print("Stage 0 Completed Succesfully!")