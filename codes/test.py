"""
Created on Sun Mar 21 2021
@author: Agapi Davradou

This module contains is used to evaluate the model.

Example to run from terminal: python test.py test --weights model_3.h5
"""

from __future__ import division, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from dataloader import *
from argparser import args
import gc
import glob
import cv2
import numpy as np
import SimpleITK as sitk
from metrics import *
from models import *
from skimage.measure import find_contours
from tensorflow.keras.optimizers import Adam
import enum


class Dataset(enum.Enum):
    train = 1
    test = 2


def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


def get_model(img_rows, img_cols):
    unet_model = unet(img_shape=(img_rows, img_cols, 1),
                           start_ch=args.start_ch,
                           dropout=args.dropout,
                           maxpool=args.maxpool,
                           residual=args.residual,
                           print_model=args.print_model)

    model = unet_model.create_model()
    model.load_weights(args.output_path + '/' + args.weights_name)
    model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def predict_test(fileList, X_test, y_test, folder=args.test_path + '/test/', dest=args.test_path + '/predictions/', plot=False):
    if not os.path.isdir(dest):
        os.mkdir(dest)

    img_rows = X_test.shape[1]
    img_cols = img_rows
    model = get_model(img_rows, img_cols)
    y_pred = model.predict(X_test, verbose=1, batch_size=128)

    start_ind = 0
    end_ind = 0

    for filename in fileList:
        itkimage = sitk.ReadImage(filename)
        img = sitk.GetArrayFromImage(itkimage)
        start_ind = end_ind
        end_ind += len(img)
        pred = resize_pred_to_val(y_pred[start_ind:end_ind], img.shape)
        pred = np.squeeze(pred)
        mask = sitk.GetImageFromArray(pred)
        mask.SetOrigin(itkimage.GetOrigin())
        mask.SetDirection(itkimage.GetDirection())
        mask.SetSpacing(itkimage.GetSpacing())
        sitk.WriteImage(mask, dest + '/' + os.path.basename(filename)[:-7] + '_segmentation.nii.gz')

    if plot:
        make_test_plots(X_test, y_test, y_pred)


def make_test_plots(X, y, y_pred, n_best=20, n_worst=20):
    # PLotting the results'
    img_rows = X.shape[1]
    img_cols = img_rows
    axis = tuple(range(1, X.ndim))
    scores = numpy_dice(y, y_pred, axis=axis)
    sort_ind = np.argsort(scores)[::-1]
    indice = np.nonzero(y.sum(axis=axis))[0]  # Keep only images with pancreas.

    # Add some best and worst predictions
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count += 1
        if count > n_best:
            break

    segm_pred = y_pred[img_list].reshape(-1, img_rows, img_cols)
    img = X[img_list].reshape(-1, img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    n_cols = 4
    n_rows = int(np.ceil(len(img) / n_cols))

    fig = plt.figure(figsize=[4 * n_cols, int(4 * n_rows)])
    gs = gridspec.GridSpec(n_rows, n_cols)

    for mm in range(len(img)):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], cmap='gray')
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    path = args.test_path + '/images/best_predictions' + '.png'

    fig.savefig(path, bbox_inches='tight', dpi=300)

    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count += 1
        if count > n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1, img_rows, img_cols)
    img = X[img_list].reshape(-1, img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    n_cols = 4
    n_rows = int(np.ceil(len(img) / n_cols))

    fig = plt.figure(figsize=[4 * n_cols, int(4 * n_rows)])
    gs = gridspec.GridSpec(n_rows, n_cols)

    for mm in range(len(img)):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], cmap='gray')
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    path = args.test_path + '/images/worst_predictions_' + '.png'

    fig.savefig(path, bbox_inches='tight', dpi=300)



def resize_pred_to_val(y_pred, shape):
    row = shape[1]
    col = shape[2]

    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm, :, :] = cv2.resize(y_pred[mm, :, :, 0], (row, col), interpolation=cv2.INTER_NEAREST)

    return resized_pred.astype(int)



def optimize(segm_3D_array):
    total_slices = segm_3D_array.shape[0]
    #print(total_slices)

    # iterate through slices
    for current_slice in range(0, total_slices - 1):

        first_slice = segm_3D_array[current_slice - 1, :, :]
        middle_slice = segm_3D_array[current_slice, :, :]
        last_slice = segm_3D_array[current_slice + 1, :, :]
        #print(str(current_slice) + " / " + str(total_slices))

        for x in range(0, middle_slice.shape[0]):
            for y in range(0, middle_slice.shape[1]):
                if (middle_slice[x, y] == 0 and first_slice[x, y] == 1 and last_slice[x, y] == 1):
                    middle_slice[x, y] = 1

        for x in range(0, middle_slice.shape[0]):
            for y in range(0, middle_slice.shape[1]):
                if (middle_slice[x, y] == 1 and first_slice[x, y] == 0 and last_slice[x, y] == 0):
                    middle_slice[x, y] = 0

        segm_3D_array[current_slice, :, :] = middle_slice

    return segm_3D_array



def check_predictions(true_label, prediction):
    return numpy_dice(true_label, prediction)


def read_cases(flip, folder, name):
    filenames = glob.glob(folder + name, recursive=True)

    for filename in filenames:
        itkimage = sitk.ReadImage(filename)
        if flip:
            itkimage = sitk.Flip(itkimage, [False, True, False])
        img = sitk.GetArrayFromImage(itkimage)

        return img




if __name__ == '__main__':

    logger = log(path = args.output_path, file="test.log") #Set a logger file

    logger.info("-" * 30)
    logger.info("Reading test data ...")
    logger.info("-" * 30)

    test_data_list = [f for f in glob.glob(args.test_path + '/test/*' + '/*' + '/*nii*', recursive=True)]

    label_list = list(filter(lambda x: 'mask' in x, test_data_list))
    test_list = list(filter(lambda x: x not in label_list, test_data_list))
    test_data_list = sorted(test_data_list)
    test_list = sorted(test_list)
    label_list = sorted(label_list)
    logger.info("test_data_list: " + str(test_data_list))
    logger.info("test_list: " + str(test_list))
    logger.info("label_list: " + str(label_list))

    images, masks = data_to_array(args.image_size, args.image_size, Dataset.test) # Read test data

    plt.imsave(args.output_path + "/test_image.png", images[100, :, :, 0], cmap='gray')
    plt.imsave(args.output_path + "/label_image.png", masks[100, :, :, 0], cmap='gray')
    logger.info("Test shape: " + str(images.shape))
    logger.info("Label shape: " + str(masks.shape))

    logger.info("-" * 30)
    logger.info("Predicting segmentation ...")
    logger.info("-" * 30)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    predict_test(test_list, images, masks, plot=args.plot_results)


    logger.info("-" * 30)
    logger.info("Calculating Dice score ...")
    logger.info("-" * 30)
    del images, masks
    gc.collect()  # Invoke Garbage Collector

    pred_list = os.listdir(args.test_path + '/predictions/')
    pred_list = list(filter(lambda x: 'nii.gz' in x, pred_list))
    pred_list = sorted(pred_list)

    dice_scores = []
    dice_scores_opt = []

    for count, filename in enumerate (pred_list):

        logger.info("\nvolume filename is: " + filename)

        y_pred = read_cases(flip = False, folder = args.test_path + '/predictions/', name = filename) #Read predicted segmentation
        logger.info("Prediction shape: " + str(y_pred.shape))
        plt.imsave(args.output_path + "/y_pred.png", y_pred[100, :, :], cmap='gray')

        label_filename = label_list[count]
        logger.info("label filename is " + label_filename)

        y_test = read_cases(flip = True, folder='', name = label_filename)  #Read test labels
        logger.info("Label reread shape: " + str(y_test.shape))
        plt.imsave(args.output_path + "/y_test2.png", y_test[100, :, :], cmap='gray')

        if (y_pred.shape != y_test.shape):
            raise NameError('Prediction and label shapes for filename %1 do not match.', filename)

        logger.info("Dice score before optimization: " + str(check_predictions(y_test, y_pred)))
        dice_scores.append(check_predictions(y_test, y_pred))

        # Apply custom optimization
        y_pred_optimized = optimize(y_pred)

        logger.info("Dice score after optimization: " + str(check_predictions(y_test, y_pred_optimized)))
        #current_dice_score_opt = check_predictions(y_test, y_pred_optimized)
        dice_scores_opt.append(check_predictions(y_test, y_pred_optimized))

    logger.info("std of dice_scores: " + str(np.std(dice_scores, dtype=np.float32)))
    logger.info("mean of dice_scores: " + str(np.mean(dice_scores, dtype=np.float32)))
    logger.info("std of dice_scores_opt: " + str(np.std(dice_scores_opt, dtype=np.float32)))
    logger.info("mean of dice_scores_opt: " + str(np.mean(dice_scores_opt, dtype=np.float32)))