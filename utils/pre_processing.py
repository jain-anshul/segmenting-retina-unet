import numpy as np
np.random.seed(1337)
from PIL import Image
import cv2

import bob.ip.gabor as gabor
from help_functions import *
from bob.sp import fft

def my_PreProc_patches(data):
    assert(len(data.shape)==4)
    
    # data = fourier_transform_real_imag(data)
    # for i in range(data.shape[0]):
    #   data[i] = image_normalize(data[i])
    
    return data


def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = gabor_DWT(imgs = train_imgs, number_of_scales = 2, number_of_directions = 2)
    #train_imgs = fourier_transform_real(train_imgs)
    train_imgs = dataset_normalized(train_imgs)
    #train_imgs = clahe_equalized(train_imgs)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


def image_normalize(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img-img_mean)/img_std

    return img_normalized

def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
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


def gabor_DWT(imgs, number_of_scales, number_of_directions):
    gwt = gabor.Transform(number_of_scales = number_of_scales, number_of_directions = number_of_directions)
    transformed_img = np.empty((imgs.shape[0],imgs.shape[1]*number_of_scales*number_of_directions, imgs.shape[2],imgs.shape[3] ))
    for i in range(imgs.shape[0]):
        transformed_img[i] = gwt(imgs[i][0])
        transformed_img[i] = np.real(transformed_img[i])

    return transformed_img

