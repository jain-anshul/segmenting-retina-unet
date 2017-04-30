import h5py
import numpy as np
np.random.seed(1337)
from PIL import Image
from matplotlib import pyplot as plt

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

def class_accuracy(y_predicted, y_value, threshold=0.5):
    # Calculates and returns the False Alarm Rate, False Reject Rate, True Alarm Rate, True Reject Rate.

    # Hypothesis
    false_reject = 0
    false_alarm = 0
    true_alarm = 0
    true_reject = 0
    print y_value.shape, y_predicted.shape

    # Total positive examples would be the sum of y_val because it would contain a 1 for every possible +ve example
    # and 0 for -ve example
    total_positive_examples = sum(y_value)
    total_negative_examples = len(y_value) - total_positive_examples

    for i in range(0, len(y_predicted)):
        # Checking for the hypothesis
        if y_predicted[i] >= threshold and y_value[i] == 0:
            false_alarm += 1
        elif y_predicted[i] < threshold and y_value[i] == 1:
            false_reject += 1
        elif y_predicted[i] >= threshold and y_value[i] == 1:
            true_alarm += 1
        elif y_predicted[i] < threshold and y_value[i] == 0:
            true_reject += 1
        else:
            print "hello", y_predicted[i], y_value[i]
    print true_reject, false_alarm
    print false_reject, true_alarm

    return (false_alarm / float(total_negative_examples), false_reject / float(total_positive_examples),
            true_alarm / float(total_positive_examples), true_reject / float(total_negative_examples))


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    img = None
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg

def conv_to_imgs(pred, img_h, img_w, patch_h, patch_w, path_experiment, mode='original'):
    assert (len(pred.shape) == 2)  # 3D array: (Npatches,2)
    assert (pred.shape[1] == 2)  # check the classes are 2
    pred_image = np.empty((pred.shape[0]))  # (Npatches,height*width)
    img_descp = mode
    threshold = 0.4
    if mode == "original":
        for i in range(pred.shape[0]):
            pred_image[i] = pred[i, 1]
    elif mode == "threshold":
        img_descp += "_" + str(threshold)
        for i in range(pred.shape[0]):
            if pred[i, 1] >= threshold:
                pred_image[i] = 1
            else:
                pred_image[i] = 0
    else:
        print "mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'"
        exit()
    pred_image = np.reshape(pred_image, (1, (img_h - (patch_h-1)), (img_w - (patch_w-1))))
    final_image = np.zeros((1,img_h,img_w))
    final_image[:, int(patch_h/2):int(img_h-patch_h/2), int(patch_w/2):int(img_w-patch_w/2)] = pred_image
    print pred_image.shape, final_image.shape
    visualize(np.transpose(final_image, (1, 2, 0)), path_experiment + 'test_prediction_' + img_descp)
    return final_image
