#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 2021

@author: Agapi Davradou
"""

from __future__ import division, print_function
import os
from functools import partial
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from models import *
from metrics import dice_coef, dice_coef_loss
from augmenters import *
from sklearn.model_selection import KFold
import gc


def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


def data_to_array(img_rows, img_cols):
    print("-" * 30)
    print("Converting dataset to .npy format ...")
    print("-" * 30)

    train_list = os.listdir('../data/train/')
    train_list = filter(lambda x: '.nii.gz' in x, train_list)
    train_list = sorted(train_list)

    images = []
    masks = []
    for filename in train_list:

        itkimage = sitk.ReadImage('../data/train/' + filename)
        itkimage = sitk.Flip(itkimage, [False, True, False])  # Flip to show correct.
        imgs = sitk.GetArrayFromImage(itkimage)

        if 'label' in filename.lower():
            imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
            masks.append(imgs)

        else:
            imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
            images.append(imgs)

    images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols, 1)
    masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
    masks = masks.astype(int)

    # Smooth images using CurvatureFlow
    images = smooth_images(images)

    np.save('../data/X_train.npy', images)
    np.save('../data/y_train.npy', masks)

    print("-" * 30)
    print("Saved train.npy")
    print("-" * 30)


def load_data():
    print("-" * 30)
    print("Loading data ...")

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')

    print("Data loading finished.")
    print("-" * 30)

    return X_train, y_train


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


def plot_graphs(model_hist, foldNumber):
    acc = model_hist.history['dice_coef']
    val_acc = model_hist.history['val_dice_coef']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    # Plot accuracy graph
    plt.clf()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.title('Model accuracy')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.savefig('accuracy_fold' + str(foldNumber) +'_.png')

    # Plot loss graph
    plt.clf()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.savefig('loss_fold' + str(foldNumber) +'_.png')


def keras_fit_generator(img_rows=96, img_cols=96, batch_size=8, regenerate=True):
    if regenerate:
        data_to_array(img_rows, img_cols)

    kf = KFold(n_splits=4)

    X_train, y_train = load_data()
    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]

    print("Dataset shape: " + str(X_train.shape))

    # Save a slice of an image and mask on disk, respectively
    plt.imsave("X_train_" + str(kf) + ".png", X_train[100, :, :, 0], cmap='gray')
    plt.imsave("y_train_" + str(kf) + ".png", y_train[100, :, :, 0], cmap='gray')

    # Provide the same seed and keyword arguments to the fit and flow methods
    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows * 1.5, sigma=img_rows * 0.07)
    # Create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[1, 1.2],
        fill_mode='constant',
        preprocessing_function=elastic)

    fold_var = 1

    for train_index, val_index in kf.split(y_train):
        print("-" * 30)
        print("Fold " + str(fold_var) + " running")

        training_data = X_train[train_index]
        mask_data = y_train[train_index]
        validation_data = X_train[val_index]
        validation_mask_data = y_train[val_index]

        n_imgs_train = training_data.shape[0]
        n_imgs_valid = validation_data.shape[0]

        print("Train dataset shape: " + str(n_imgs_train))
        print("Validation dataset shape: " + str(n_imgs_valid))
        print("-" * 30)

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen_val = ImageDataGenerator(**data_gen_args)
        mask_datagen_val = ImageDataGenerator(**data_gen_args)

        seed = 2
        image_datagen.fit(X_train[train_index], seed=seed)
        mask_datagen.fit(y_train[train_index], seed=seed)
        image_datagen_val.fit(X_train[val_index], seed=seed)
        mask_datagen_val.fit(y_train[val_index], seed=seed)

        image_generator = image_datagen.flow(training_data, batch_size=batch_size, seed=seed)
        mask_generator = mask_datagen.flow(mask_data, batch_size=batch_size, seed=seed)
        image_generator_val = image_datagen_val.flow(validation_data, batch_size=batch_size, seed=seed)
        mask_generator_val = mask_datagen_val.flow(validation_mask_data, batch_size=batch_size, seed=seed)

        train_generator = (pair for pair in zip(image_generator, mask_generator))
        valid_generator = (pair for pair in zip(image_generator_val, mask_generator_val))

        model = UNet((img_rows, img_cols, 1), start_ch=6, depth=5, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
        #    model.load_weights('../data/weights.h5')

        model.summary()
        model_checkpoint = ModelCheckpoint(
            get_model_name(fold_var), monitor='val_loss', save_best_only=True)
        c_backs = [model_checkpoint]
        c_backs.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15))

        model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])

        model_history = model.fit_generator(
            train_generator,
            steps_per_epoch=n_imgs_train // batch_size,
            epochs=1000,
            verbose=2,
            shuffle=True,
            validation_data=valid_generator,
            validation_steps=n_imgs_valid // batch_size,
            callbacks=c_backs,
            use_multiprocessing=True)

        # List all data in history
        print(model_history.history.keys())

        plot_graphs(model_history, fold_var)

        tf.keras.backend.clear_session()
        del training_data, mask_data, image_generator, mask_generator, image_generator_val, mask_generator_val, train_generator, valid_generator
        gc.collect() # Invoke Garbage Collector
        fold_var += 1


if __name__ == '__main__':
    import time

    start = time.time()
    keras_fit_generator(img_rows=256, img_cols=256, regenerate=False, batch_size=128)

    end = time.time()

    print("-" * 30)
    print('Elapsed time:', round((end - start) / 60, 2))
    print("-" * 30)