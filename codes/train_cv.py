"""
Created on Sun Mar 21 2021
@author: Agapi Davradou

This module contains the main code for training the model.

Example to run from terminal: python train_cv.py train --epochs 1000 --regenerate True --preprocess_label False
"""

from __future__ import division, print_function
from functools import partial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import *
from metrics import dice_coef, dice_coef_loss
from sklearn.model_selection import KFold
from dataloader import *
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import enum


class Dataset(enum.Enum):
    train = 1
    test = 2


def plot_graphs(model_hist, fold_number):
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
    plt.savefig(args.output_path + '/accuracy_fold' + str(fold_number) +'_.png')

    # Plot loss graph
    plt.clf()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.savefig(args.output_path + '/loss_fold' + str(fold_number) +'_.png')



def create_generator(gen_args, x_data, y_data, batch):
    x_datagen = ImageDataGenerator(**gen_args)
    y_datagen = ImageDataGenerator(**gen_args)

    seed = args.seed
    x_datagen.fit(x_data, seed=seed)
    y_datagen.fit(y_data, seed=seed)

    image_generator = x_datagen.flow(x_data, batch_size=batch, seed=seed)
    mask_generator = y_datagen.flow(y_data, batch_size=batch, seed=seed)

    data_generator = (pair for pair in zip(image_generator, mask_generator))

    return data_generator



def keras_fit_generator(img_rows=96, img_cols=96, batch_size=8, regenerate=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if regenerate:
        data_to_array(img_rows, img_cols, Dataset.train)

    kf = KFold(n_splits=args.folds)

    X_train, y_train = load_data()
    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]

    logger.info("Dataset shape: " + str(X_train.shape))

    # Save a slice of an image and mask on disk, respectively
    plt.imsave(args.output_path + "/X_train_" + str(kf) + ".png", X_train[100, :, :, 0], cmap='gray')
    plt.imsave(args.output_path + "/y_train_" + str(kf) + ".png", y_train[100, :, :, 0], cmap='gray')

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
        logger.info("-" * 30)
        logger.info("Fold " + str(fold_var) + " running")

        training_data = X_train[train_index]
        mask_data = y_train[train_index]
        validation_data = X_train[val_index]
        validation_mask_data = y_train[val_index]

        n_imgs_train = training_data.shape[0]
        n_imgs_valid = validation_data.shape[0]

        logger.info("Train dataset shape: " + str(n_imgs_train))
        logger.info("Validation dataset shape: " + str(n_imgs_valid))
        logger.info("-" * 30)

        train_generator = create_generator(data_gen_args, training_data, mask_data, batch_size)
        valid_generator = create_generator(data_gen_args, validation_data, validation_mask_data, batch_size)

        unet_model = unet(img_shape=(img_rows, img_cols, 1),
                          start_ch=args.start_ch,
                          dropout=args.dropout,
                          maxpool=args.maxpool,
                          residual=args.residual,
                          model_name=fold_var,
                          print_model=args.print_model)

        model = unet_model.create_model()
        #    model.load_weights(args.input_path + '/weights.h5')
        model_callbacks = unet_model.get_callbacks()

        model.compile(optimizer=Adam(lr=args.learningrate), loss=dice_coef_loss, metrics=[dice_coef])

        model_history = model.fit_generator(
            train_generator,
            steps_per_epoch=n_imgs_train // batch_size,
            epochs=args.epochs,
            verbose=args.verbose,
            validation_data=valid_generator,
            validation_steps=n_imgs_valid // batch_size,
            callbacks=model_callbacks,
            use_multiprocessing=False)

        # List all data in history
        logger.info(model_history.history.keys())

        plot_graphs(model_history, fold_var)

        tf.keras.backend.clear_session()
        del training_data, mask_data, train_generator, valid_generator
        gc.collect() # Invoke Garbage Collector
        fold_var += 1


if __name__ == '__main__':
    import time

    start = time.time()
    logger = log(path = args.output_path, file="train.log") #Set a logger file
    keras_fit_generator(img_rows=args.image_size, img_cols=args.image_size, regenerate=args.regenerate, batch_size=args.batch_size)
    end = time.time()

    logger.info("-" * 30)
    logger.info('Elapsed time:', str(round((end - start) / 60, 2)))
    logger.info("-" * 30)