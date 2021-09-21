"""
Created on Sun Mar 21 2021
@author: Agapi Davradou

This module contains the main code for training the model.
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



def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


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
        itkimage = sitk.ReadImage(folder + filename)
        img = sitk.GetArrayFromImage(itkimage)
        start_ind = end_ind
        end_ind += len(img)
        pred = resize_pred_to_val(y_pred[start_ind:end_ind], img.shape)
        pred = np.squeeze(pred)
        mask = sitk.GetImageFromArray(pred)
        mask.SetOrigin(itkimage.GetOrigin())
        mask.SetDirection(itkimage.GetDirection())
        mask.SetSpacing(itkimage.GetSpacing())
        sitk.WriteImage(mask, dest + '/' + filename[:-7] + '_segmentation.nii.gz')

        if plot:
            make_test_plots(X_test, y_test, y_pred, volumename=filename)


def make_test_plots(X, y, y_pred, n_best=20, n_worst=20, volumename='test'):
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

    path = args.test_path + '/images/best_predictions_' + volumename[:-7] + '.png'

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

    path = args.test_path + '/images/worst_predictions_' + volumename[:-7] + '.png'

    fig.savefig(path, bbox_inches='tight', dpi=300)


def get_model(img_rows, img_cols):
    model = UNet((img_rows, img_cols, 1), start_ch=6, depth=5, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    model.load_weights(args.weight_path + '/' + args.weights_name)
    model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def resize_pred_to_val(y_pred, shape):
    row = shape[1]
    col = shape[2]

    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm, :, :] = cv2.resize(y_pred[mm, :, :, 0], (row, col), interpolation=cv2.INTER_NEAREST)

    return resized_pred.astype(int)



def optimize(segm_3D_array):
    total_slices = segm_3D_array.shape[0]
    print(total_slices)

    # iterate through slices
    for current_slice in range(0, total_slices - 1):

        first_slice = segm_3D_array[current_slice - 1, :, :]
        middle_slice = segm_3D_array[current_slice, :, :]
        last_slice = segm_3D_array[current_slice + 1, :, :]
        print(str(current_slice) + " / " + str(total_slices))

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
    print('Accuracy:', numpy_dice(true_label, prediction))


def read_test_data(folder=args.test_path + '/test/', img_rows=256, img_cols=256, volume_name='.nii.gz'):
    fileList = os.listdir(folder)
    fileList = filter(lambda x: volume_name in x, fileList)
    fileList = sorted(fileList)
    n_imgs = []
    images = []
    for filename in fileList:
        itkimage = sitk.ReadImage(folder + filename)
        itkimage = sitk.Flip(itkimage, [False, True, False])  # Flip to show correct.
        imgs = sitk.GetArrayFromImage(itkimage)
        #        imgs = imgs.astype(int)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=False)  # img_rows

        images.append(imgs)
        n_imgs.append(len(imgs))

    images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols, 1)  # img_rows
    images = smooth_images(images)
    return images, fileList



def read_cases(the_list=None, flip=True, folder='../data/train/', name='/*0080*.nii.gz'):
    filenames = glob.glob(folder + name, recursive=True)  # Add pancreas to exclude label files.

    for filename in filenames:
        itkimage = sitk.ReadImage(filename)
        if flip:
            itkimage = sitk.Flip(itkimage, [False, True, False])  # Flip images, because they are shown reversed.
        img = sitk.GetArrayFromImage(itkimage)

        return img


if __name__ == '__main__':
    volume_name = args.volume_name

    print("-" * 30)
    print("Reading test data ...")
    print("-" * 30)
    X_test, x_list = read_test_data(args.test_path + '/test/', args.image_size, args.image_size, volume_name) #Read test data
    y_test, y_list = read_test_data(args.test_path + '/test_labels/', args.image_size, args.image_size, volume_name) #Read test labels
    y_test = y_test.astype(int)

    plt.imsave("X_test.png", X_test[100, :, :, 0], cmap='gray')
    plt.imsave("y_test.png", y_test[100, :, :, 0], cmap='gray')
    print("Test shape: " + str(X_test.shape))
    print("Label shape: " + str(y_test.shape))

    print("-" * 30)
    print("Predicting segmentation ...")
    print("-" * 30)
    #predict_test(x_list, X_test, y_test, plot=args.plot_results)

    print("-" * 30)
    print("Calculating Dice score ...")
    print("-" * 30)
    del X_test, y_test
    gc.collect()  # Invoke Garbage Collector

    y_test = read_cases(flip = True, folder=args.test_path + '/test_labels/', name = '/*' + volume_name + '*.nii.gz')  #Read test labels
    print("Label reread shape: " + str(y_test.shape))
    plt.imsave("y_test2.png", y_test[100, :, :], cmap='gray')
    y_pred = read_cases(flip = False, folder = args.test_path + '/predictions/', name = '/*' + volume_name + '*.nii.gz') #Read predicted segmentation
    print("Prediction shape: " + str(y_pred.shape))
    print("Number of 1s: " +  str((y_pred == 1).sum()))
    print("Number of 0s: " + str((y_pred == 0).sum()))
    plt.imsave("y_pred.png", y_pred[100, :, :], cmap='gray')

    print("Dice score before optimization: ")
    check_predictions(y_test, y_pred)

    # Apply custom optimization
    y_pred_optimized = optimize(y_pred)

    print("Dice score after optimization: ")
    check_predictions(y_test, y_pred_optimized)