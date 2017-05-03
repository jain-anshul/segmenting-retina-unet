import numpy as np
import ConfigParser
from matplotlib import pyplot as plt

from keras.models import model_from_json
from keras.models import Model

# from sklearn.metrics import roc_auc_score

import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from keras.utils import np_utils



import sys
sys.path.insert(0, './utils/')
from help_functions import *
from extract_patches import get_data_val_ROC_testing

np.random.seed(1337)

# ========= CONFIG FILE TO READ FROM =======
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')

# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')



algorithm = "cnn"

name_experiment_list = ["log_normalisation_patches-fft-real_normalisation", "log_normalisation_patches-fft-real-imag_normalisation","log_normalisation_patches-fft-real-imag-raw_normalisation" ]
# name_experiment_list = ["log_normalisation_patches-gabor(2,2)-real_normalisation","log_normalisation_patches-gabor(2,2)-real_normalisation_reduced-dataset","log_normalisation_patches-gabor(2,4)-real_normalisation_reduced-dataset", "log_normalisation_patches-gabor(4,2)-real_normalisation_reduced-dataset","log_normalisation_patches-gabor(2,2)-real-imag_normalisation_reduced-dataset","log_normalisation_patches-fft-abs_normalisation","log_normalisation_patches-fft-imag_normalisation", "log_normalisation_patches-fft-real-imag-raw_normalisation_reduced-dataset"]

# plt.figure(0).clf()
# plt.title('ROC curve')
# plt.xlabel("FPR (False Positive Rate)")
# plt.ylabel("TPR (True Positive Rate)")
# plt.legend(loc="lower right")

for name_experiment in name_experiment_list:
	print '\n\n\n', name_experiment
	log_path_experiment = './log/log_balanced/'+name_experiment+'/'+algorithm+'/'

	patches_imgs_val, patches_masks_val = get_data_val_ROC_testing(
	    DRIVE_train_imgs_original = path_data + config.get('data paths', 'val_imgs_original'),
	    DRIVE_train_groudTruth = path_data + config.get('data paths', 'val_groundTruth'),  #masks
	    patch_height = int(config.get('data attributes', 'patch_height')),
	    patch_width = int(config.get('data attributes', 'patch_width')),
	    N_subimgs = int(config.get('validation settings', 'N_subimgs')),
	    inside_FOV = config.getboolean('validation settings', 'inside_FOV'), #select the patches only inside the FOV  (default == True)
	    path_experiment = log_path_experiment,
	    name_experiment = name_experiment
	)
	
	# Load the saved model
	model = model_from_json(open(log_path_experiment + name_experiment + '_architecture.json').read())
	model.load_weights(log_path_experiment + name_experiment + '_best_weights.h5')
	
	y_pred = model.predict(patches_imgs_val, batch_size=32, verbose=1)

	patches_masks_val = np_utils.to_categorical(patches_masks_val, 2)

	


	fpr, tpr, thresholds = roc_curve(y_true = patches_masks_val[:,1], y_score=y_pred[:,1], drop_intermediate=False)
	AUC_ROC = roc_auc_score(patches_masks_val[:,1], y_pred[:,1])
	# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
	print "\nArea under the ROC curve: " + str(AUC_ROC)
	# roc_curve = plt.figure()
	plt.figure()
	plt.plot(fpr, tpr, '-', label=name_experiment[25:]+'(AUC = %0.4f)' % AUC_ROC)
	plt.title('ROC curve')
	plt.xlabel("FPR (False Positive Rate)")
	plt.ylabel("TPR (True Positive Rate)")
	plt.legend(loc="lower right")
	plt.savefig('./roc_curve/' +name_experiment + "_ROC.png")

	# fpr, tpr, thresholds = roc_curve(y_true = patches_masks_val[:,1], y_score=y_pred[:,1], drop_intermediate=False)
	# AUC_ROC = roc_auc_score(patches_masks_val[:,1], y_pred[:,1])
	# # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
	# print "\nArea under the ROC curve: " + str(AUC_ROC)
	# # roc_curve = plt.figure()
	# plt.plot(fpr, tpr, '-',label=' (AUC = %0.4f)' %  AUC_ROC)
	
# plt.savefig('./roc_curve/' + "comparative_ROC.png")


