import numpy as np
import ConfigParser

import sys
sys.path.insert(0, './utils/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training


#========= Load settings from Config file
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
algorithm = 'nn'
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)

patches_imgs_val, patches_masks_val = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'val_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'val_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('validation settings', 'N_subimgs')),
    inside_FOV = config.getboolean('validation settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)

n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

patches_imgs_train = np.reshape(patches_imgs_train,(patches_imgs_train.shape[0],patch_height*patch_width))
patches_imgs_val = np.reshape(patches_imgs_val, (patches_imgs_val.shape[0],patch_height*patch_width))
print patches_imgs_train.shape

from sklearn.svm import SVC

clf = SVC()
clf.fit(patches_imgs_train, patches_masks_train)

print(clf.predict(patches_imgs_))
print(patches_masks_train)
