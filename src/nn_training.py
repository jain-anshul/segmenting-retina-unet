import numpy as np
import ConfigParser

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras.utils import np_utils

import scikitplot.plotters as skplt
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './utils/')
from help_functions import *
np.random.seed(1337)

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    
    dnn = Flatten()(inputs)
    dnn1 = Dense(256)(dnn) 
    dnn2 = Dense(2)(dnn1)

    dnn2 = core.Activation('softmax')(dnn2)

    model = Model(input=inputs, output=dnn2)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

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
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
print "Check: final output of the network:"
print model.output_shape
model.summary()
plot(model, to_file='./'+name_experiment+'/'+algorithm+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+algorithm+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+algorithm+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

patches_masks_train = np_utils.to_categorical(patches_masks_train, 2)
patches_masks_val = np_utils.to_categorical(patches_masks_val, 2)



run_flag = True
weights = []
# Check if it is the first iteration
first_iter = True

# Setting the final accuracy to 0 just for the start
final_acc = 0.0
count_neg_iter = 0
iter_count = 1
nb_neg_cycles = 3
lr = 0.01

while run_flag:

    if first_iter:
        first_iter = False
    else:
        model.set_weights(np.asarray(weights))

    sgd = SGD(lr=lr)

    print lr, " learning rate"
    print iter_count, " iteration"
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    print 'TRAIN DATA'
    model.fit(patches_imgs_train, patches_masks_train, nb_epoch=1, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpointer])
    y_pred = model.predict(patches_imgs_train, batch_size=32, verbose=1)
    fa, fr, ta, tr = class_accuracy(y_pred[:, 1], patches_masks_train[:, 1])
    print '\n',"FA FR TA TR", fa, fr, ta, tr

    print '\n','VALIDATION DATA'
    score = model.evaluate(x=patches_imgs_val, y=patches_masks_val, batch_size=32, verbose=1)
    print score[1], score[0]

    y_pred = model.predict(patches_imgs_val, batch_size=32, verbose=1)
    fa, fr, ta, tr = class_accuracy(y_pred[:, 1], patches_masks_val[:, 1])
    print '\n',"FA FR TA TR", fa, fr, ta, tr

    val_accuracy = score[1]

    print val_accuracy, " - val accuracy"
    print final_acc, " - final_accuracy"
    if val_accuracy - final_acc > 0.0005:
        iter_count += 1
        # Update the weights if the accuracy is greater than .001
        weights = model.get_weights()
        print ("Updating the weights")
        # Updating the final accuracy
        final_acc = val_accuracy
        # Setting the count to 0 again so that the loop doesn't stop before reducing the learning rate n times
        # consecutively
        count_neg_iter = 0
    else:
        # If the difference is not greater than 0.005 reduce the learning rate
        lr /= 2.0
        print ("Reducing the learning rate by half")
        count_neg_iter += 1

        # If the learning rate is reduced consecutively for nb_neg_cycles times then the loop should stop
        if count_neg_iter > nb_neg_cycles:
            run_flag = False
            model.set_weights(np.asarray(weights))


model.save_weights('./'+name_experiment+'/'+algorithm+'/'+name_experiment +'_last_weights.h5', overwrite=True)


y_pred = model.predict(patches_imgs_val, batch_size=32, verbose=1)

print '\n', roc_auc_score(patches_masks_val[:,1], y_pred[:,1])
print y_pred[:,1].shape, patches_masks_val[:,1].shape
skplt.plot_roc_curve(patches_masks_val[:,1], y_pred, curves=('each_class'))
plt.savefig('./'+name_experiment+'/'+algorithm+'/'+name_experiment +'_roc.png')
