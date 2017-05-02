import numpy as np
import ConfigParser
from matplotlib import pyplot as plt

from keras.models import model_from_json
from keras.models import Model

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

name_experiment_list = ["log_normalisation_clahe"]

for name_experiment in name_experiment_list:

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

	log_path_experiment = './log/log_balanced/'+name_experiment+'/'+algorithm+'/'
	model.load_weights(log_path_experiment + name_experiment + '_best_weights.h5')
	y_pred = model.predict(patches_imgs_val, batch_size=32, verbose=1)

	print '\n', 'ROC AREA: ', roc_auc_score(patches_masks_val[:,1], y_pred[:,1])
	print y_pred[:,1].shape, patches_masks_val[:,1].shape
	skplt.plot_roc_curve(patches_masks_val[:,1], y_pred, curves=('each_class'))
	plt.savefig('./test/' + name_experiment +'_roc.png')




