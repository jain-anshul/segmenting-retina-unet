import numpy as np
import ConfigParser
from matplotlib import pyplot as plt

from keras.models import model_from_json
from keras.models import Model

import sys
sys.path.insert(0, './utils/')
from help_functions import *
from extract_patches import get_data_testing_single_image
algorithm = sys.argv[1]

# ========= CONFIG FILE TO READ FROM =======
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')

# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

print test_imgs_orig.shape,full_img_width, full_img_height

# # the border masks provided by the DRIVE
# DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
# test_border_masks = load_hdf5(DRIVE_test_border_masks)

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './log/log_balanced/'+name_experiment+'/'+algorithm+'/'

# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

print N_visual, path_experiment

visualize(group_images(test_imgs_orig[0:20,:,:,:],5),path_experiment + 'test_original')

gtruth= path_data + config.get('data paths', 'test_groundTruth')
img_truth= load_hdf5(gtruth)

visualize(group_images(img_truth[0:20,:,:,:],5),path_experiment + 'test_gtruth')

# Load the saved model
model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
model.load_weights(path_experiment + name_experiment + '_best_weights.h5')

for index in range(test_imgs_orig.shape[0]):
	print(index+1," image")
	# ============ Load the data and divide in patches
	patches_imgs_test = None
	new_height = None
	new_width = None
	masks_test = None
	patches_masks_test = None
	patches_imgs_test, patches_masks_test = get_data_testing_single_image(
	    DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
	    DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
	    patch_height=patch_height,
	    patch_width=patch_width,
	    index=index
	)

	print patches_imgs_test.shape, patches_masks_test.shape

	# ================ Run the prediction of the patches ==================================

	# Calculate the predictions
	predictions = model.predict(patches_imgs_test, batch_size=32, verbose=1)
	print "predicted images size :"
	print predictions.shape

	# ===== Convert the prediction arrays in corresponding images

	pred_img = conv_to_imgs(pred=predictions,img_h=img_truth.shape[2],img_w=img_truth.shape[3],mode='threshold', patch_h=patch_height, patch_w=patch_width, path_experiment = path_experiment, index=index)
