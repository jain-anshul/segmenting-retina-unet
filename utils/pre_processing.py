import numpy as np
np.random.seed(1337)
from PIL import Image
import cv2

import bob.ip.gabor as gabor
from help_functions import *
from bob.sp import fft

def my_PreProc_patches(data):
    assert(len(data.shape)==4)

    # image.shape[1] >1 in using gabor wavelet,  so cannot have fixed number of channels
    #assert (data.shape[1]==1)

    data = fourier_transform_real_imag_raw_image(data)
    # data = gabor_DWT_real_imag(imgs = data, number_of_scales = 4, number_of_directions = 2)

    # data = gabor_DWT_real(imgs = data, number_of_scales = 2, number_of_directions = 2)
    # data = fourier_transform_real_imag(data)
    for i in range(data.shape[0]):
        data[i] = image_normalize(data[i])
        # data[i][:2] = image_normalize(data[i][:2])
        # data[i][2] = image_normalize(data[i][2])
    print("\n\nTraining patches normalised successfully, shape is ",data.shape)


    return data

def my_PreProc_patches_ROC_testing(data, name_experiment):
    assert(len(data.shape)==4)
    if name_experiment=="log_normalisation_patches-fft-real_normalisation":
        data = fourier_transform_real(data)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-fft-real-imag_normalisation":
        data = fourier_transform_real_imag(data)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-fft-real-imag-raw_normalisation":
        data = fourier_transform_real_imag_raw_image(data)
        for i in range(data.shape[0]):
            data[i][:2] = image_normalize(data[i][:2])
            data[i][2] = image_normalize(data[i][2])
    elif name_experiment=="log_normalisation_patches-gabor(2,2)-real_normalisation":
        data = gabor_DWT_real(imgs = data, number_of_scales = 2, number_of_directions = 2)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-gabor(2,2)-real_normalisation_reduced-dataset":
        data = gabor_DWT_real(imgs = data, number_of_scales = 2, number_of_directions = 2)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-gabor(2,4)-real_normalisation_reduced-dataset":
        data = gabor_DWT_real(imgs = data, number_of_scales = 2, number_of_directions = 4)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-gabor(4,2)-real_normalisation_reduced-dataset":
        data = gabor_DWT_real(imgs = data, number_of_scales = 4, number_of_directions = 2)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-gabor(2,2)-real-imag_normalisation_reduced-dataset":
        data = gabor_DWT_real_imag(imgs = data, number_of_scales = 2, number_of_directions = 2)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-fft-abs_normalisation":
        data = fourier_transform_abs(data)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-fft-imag_normalisation":
        data = fourier_transform_imag(data)
        for i in range(data.shape[0]):
            data[i] = image_normalize(data[i])
    elif name_experiment=="log_normalisation_patches-fft-real-imag-raw_normalisation_reduced-dataset":
        data = fourier_transform_real_imag_raw_image(data)
        for i in range(data.shape[0]):
            data[i][:2] = image_normalize(data[i][:2])
            data[i][2] = image_normalize(data[i][2])
    # data = gabor_DWT_real_imag(imgs = data, number_of_scales = 4, number_of_directions = 2)
    print("\n\nTraining patches normalised successfully, shape is ",data.shape)

    return data

def my_PreProc_ROC_testing(data, name_experiment):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)


    if name_experiment=="log_normalisation_clahe":
        train_imgs = clahe_equalized(train_imgs)
    
    # train_imgs = gabor_DWT_real_imag(imgs = train_imgs, number_of_scales = 2, number_of_directions = 2)

    #train_imgs = fourier_transform_real(train_imgs)
    #train_imgs = dataset_normalized(train_imgs)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    print("\n\nTraining images normalised successfully, shape is ",train_imgs.shape)
    return train_imgs


def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    # train_imgs = gabor_DWT_real_imag(imgs = train_imgs, number_of_scales = 2, number_of_directions = 2)
    #train_imgs = fourier_transform_real(train_imgs)
    #train_imgs = dataset_normalized(train_imgs)
    #train_imgs = clahe_equalized(train_imgs)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    print("\n\nTraining images normalised successfully, shape is ",train_imgs.shape)
    return train_imgs


def image_normalize(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img-img_mean)/img_std

    return img_normalized

def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays

    # image.shape[1] >1 in using gabor wavelet,  so cannot have fixed number of channels
    #assert (imgs.shape[1]==1)  #check the channel is 1

    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / \
            (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def fourier_transform_real(imgs):
    print(imgs.shape)
    for i in range(imgs.shape[0]):
        freq_img = fft(imgs[i][0].astype(np.complex128))
        imgs[i][0] = np.real(freq_img)

    return imgs

def fourier_transform_imag(imgs):
    for i in range(imgs.shape[0]):
        freq_img = fft(imgs[i][0].astype(np.complex128))
        imgs[i][0] = np.imag(freq_img)

    return imgs

def fourier_transform_abs(imgs):
    for i in range(imgs.shape[0]):
        freq_img = fft(imgs[i][0].astype(np.complex128))
        imgs[i][0] = np.abs(freq_img)

    return imgs

def fourier_transform_real_imag(imgs):
    print imgs.shape
    transformed_patch = np.empty((imgs.shape[0],imgs.shape[1]*2,imgs.shape[2],imgs.shape[3]))
    for i in range(imgs.shape[0]):
        freq_img = fft(imgs[i][0].astype(np.complex128))
        
        transformed_patch[i][0] = np.real(freq_img)
        transformed_patch[i][1] = np.imag(freq_img)
    print("Adding real+imaginary part", transformed_patch.shape)
    return transformed_patch


def gabor_DWT_real(imgs, number_of_scales, number_of_directions):
    gwt = gabor.Transform(number_of_scales = number_of_scales, number_of_directions = number_of_directions)

    transformed_img = np.empty((imgs.shape[0],imgs.shape[1]*number_of_scales*number_of_directions, imgs.shape[2],imgs.shape[3] ))
    for i in range(imgs.shape[0]):
        transformed_img[i] = gwt(imgs[i][0])
        transformed_img[i] = np.real(transformed_img[i])


    return transformed_img

def gabor_DWT_real_imag(imgs, number_of_scales, number_of_directions):
    gwt = gabor.Transform(number_of_scales = number_of_scales, number_of_directions = number_of_directions)
    transformed_img = np.empty((imgs.shape[0],imgs.shape[1]*number_of_scales*number_of_directions*2, imgs.shape[2],imgs.shape[3] ))
    for i in range(imgs.shape[0]):
        temp = gwt(imgs[i][0])
        transformed_img[i][:number_of_directions*number_of_scales] = np.real(temp)
        transformed_img[i][number_of_directions*number_of_scales:] = np.imag(temp)
    return transformed_img

def fourier_transform_real_imag_raw_image(imgs):
    print imgs.shape
    transformed_patch = np.empty((imgs.shape[0],imgs.shape[1]*3,imgs.shape[2],imgs.shape[3]))
    for i in range(imgs.shape[0]):
        freq_img = fft(imgs[i][0].astype(np.complex128))
        
        transformed_patch[i][0] = np.real(freq_img)
        transformed_patch[i][1] = np.imag(freq_img)
        transformed_patch[i][2] = imgs[i][0]
    print("Adding real+imaginary+raw part", transformed_patch.shape)
    return transformed_patch
