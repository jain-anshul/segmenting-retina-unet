import os
import h5py
import numpy as np
from PIL import Image

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#train
original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/1st_manual/"

#test
original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test = "./DRIVE/test/1st_manual/"

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_datasets_training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print "original image: " +files[i]
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print "ground truth name: " + groundTruth_name
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            

    print "imgs max: " +str(np.max(imgs))
    print "imgs min: " +str(np.min(imgs))
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0)
    print "ground truth are correctly withih pixel value range 0-255 (black-white)"
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    return imgs, groundTruth


imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,"train")
imgs_val = imgs_train[18:]
groundTruth_val = groundTruth_train[18:]
print imgs_val.shape

imgs_train = imgs_train[:18]
groundTruth_train = groundTruth_train [:18]
print imgs_train.shape

print "saving train datasets"
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")

print "saving validation datasets"
write_hdf5(imgs_val, dataset_path + "DRIVE_dataset_imgs_validation.hdf5")
write_hdf5(groundTruth_val, dataset_path + "DRIVE_dataset_groundTruth_validation.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_imgs_test,groundTruth_imgs_test,"test")
print "saving test datasets"
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
