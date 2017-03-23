import numpy as np
import ConfigParser

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras.utils import np_utils

import sys
sys.path.insert(0, './utils/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    dnn = Flatten()(conv1) 
    conv2 = Dense(2)(dnn)

    conv2 = core.Activation('softmax')(conv2)

    model = Model(input=inputs, output=conv2)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#========= Load settings from Config file
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
algorithm = 'cnn'
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
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
print "Check: final output of the network:"
print model.output_shape
model.summary()
plot(model, to_file='./'+name_experiment+'/'+algorithm+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/nn'+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+algorithm+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


patches_masks_train = np_utils.to_categorical(patches_masks_train, 2)
patches_masks_val = np_utils.to_categorical(patches_masks_val, 2)

for i in range(N_epochs):
    print '\n\n\n'
    print 'Iternation number: ',(i+1)
    print 'TRAIN DATA'
    model.fit(patches_imgs_train, patches_masks_train, nb_epoch=1, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])
    y_pred = model.predict(patches_imgs_train, batch_size=32, verbose=1)
    fa, fr, ta, tr = class_accuracy(y_pred[:, 1], patches_masks_train[:, 1])
    print '\n',"FA FR TA TR", fa, fr, ta, tr

    print '\n','VALIDATION DATA'
    score = model.evaluate(x=patches_imgs_val, y=patches_masks_val, batch_size=32, verbose=1)
    print score[1], score[0]

    y_pred = model.predict(patches_imgs_val, batch_size=32, verbose=1)
    fa, fr, ta, tr = class_accuracy(y_pred[:, 1], patches_masks_val[:, 1])
    print '\n',"FA FR TA TR", fa, fr, ta, tr

model.save_weights('./'+name_experiment+'/'+algorithm+'/'+name_experiment +'_last_weights.h5', overwrite=True)
