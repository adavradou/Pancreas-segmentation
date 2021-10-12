"""
Created on Wed Sep 15 2021

@author: Agapi Davradou

This module loads the dataset.

"""
import pdb

from argparser import args
import os
import cv2
import numpy as np
import SimpleITK as sitk
import glob
import re


def img_resize(imgs, img_rows, img_cols):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])

    for mm, img in enumerate(imgs):
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
    return new_imgs

def preprocess_img(img):
    """
    Z-score image normalization
    """
    mu = np.mean(img)
    sigma = np.std(img)

    return (img - mu) / sigma


def data_to_array(img_rows, img_cols, dataset):
    print("-" * 30)
    print("Converting dataset to .npy format ...")
    print("-" * 30)

    foldernames = []
    t_list = []

    if (dataset.name == 'train'):
        foldernames = glob.glob(args.input_path + '/train/*' + "/*/", recursive = True)
    elif (dataset.name == 'test'):
        foldernames = glob.glob(args.test_path + '/test/*' + "/*/", recursive=True)

    foldernames = [x for x in foldernames if "MACOS" not in x]  # Remove MACOS folders
    foldernames = sorted(foldernames)

    for folder in foldernames:
        imageName = ""
        labelName = ""
        #pdb.set_trace()
        files = glob.glob(folder + '/*nii*', recursive = True)
        for f in files:
            if not re.findall("L.nii$", f):
                images = re.findall("[0-9]+.nii.gz", f)
                labels = re.findall("mask", f)
                if images:
                    imageName = f
                if labels:
                    labelName = f
        if imageName and labelName: #Check that both the image and the label are found in the folder
            t_list.append(imageName)
            t_list.append(labelName)
            print("\nfolder: " + folder)

    images = []
    masks = []

    for filename in t_list:
        itkimage = sitk.ReadImage(filename)
        itkimage = sitk.Flip(itkimage, [False, True, False])  # Flip to show correct.
        imgs = sitk.GetArrayFromImage(itkimage)

        if 'mask' in filename.lower():
            imgs = img_resize(imgs, img_rows, img_cols)
            masks.append(imgs)

        else:
            imgs = img_resize(imgs, img_rows, img_cols)
            images.append(imgs)

    images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols, 1)
    masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
    masks = masks.astype(int)

    # Smooth images using CurvatureFlow
    images = smooth_images(images)
    #images = preprocess_img(images)

    if (dataset.name == 'train'):
        np.save(args.output_path + '/X_train.npy', images)
        np.save(args.output_path + '/y_train.npy', masks)
        print("-" * 30)
        print("Saved train.npy")
        print("-" * 30)

    elif (dataset.name == 'test'):
        return images, masks


def load_data():
    print("-" * 30)
    print("Loading data ...")

    X_train = np.load(args.output_path + '/X_train.npy')
    y_train = np.load(args.output_path + '/y_train.npy')

    print("Data loading finished.")
    print("-" * 30)

    return X_train, y_train


def elastic_transform(image, x=None, y=None, alpha=256*3, sigma=256*0.07):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    shape = image.shape
    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha
    dy = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha

    if (x is None) or (y is None):
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    map_x =  (x+dx).astype('float32')
    map_y =  (y+dy).astype('float32')

    return cv2.remap(image.astype('float32'), map_y,  map_x, interpolation=cv2.INTER_NEAREST).reshape(shape)


def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)


    return imgs