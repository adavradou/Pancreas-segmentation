
from __future__ import division, print_function
from collections import defaultdict
import os, pickle, sys
import shutil
from functools import partial
#from itertools import izip
import datetime
import tensorflow as tf
import cv2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist

from models import *
from metrics import dice_coef, dice_coef_loss, Active_Contour_Loss
from augmenters import *
from itertools import product


def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )
		
    return new_imgs

def data_to_array(img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)) )

    train_list =  os.listdir('../data/train/')
    train_list = filter(lambda x: '.nii.gz' in x, train_list)
    train_list= sorted(train_list)

    images = []
    masks = []
    for filename in train_list:        

        itkimage = sitk.ReadImage('../data/train/'+filename)
        itkimage = sitk.Flip(itkimage, [False, True, False]) #Flip to show correct.                        
        imgs = sitk.GetArrayFromImage(itkimage)

        if 'label' in filename.lower():
            imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
            masks.append( imgs )

        else:
            imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
            images.append(imgs )

    images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
    print("number of images:")
    print(len(images))
    masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
    print("number of annotations:")
    print(len(masks))		  
    masks = masks.astype(int)

    #Smooth images using CurvatureFlow
    images = smooth_images(images)
    print("smoothing finished!")   
               

    np.save('../data/X_train.npy', images)
    np.save('../data/y_train.npy', masks)
        
    print("Saved train.npy")    
	  
    valid_list =  os.listdir('../data/valid/')
    valid_list = filter(lambda x: '.nii.gz' in x, valid_list)
    valid_list= sorted(valid_list)

    images = []
    masks = []
    for filename in valid_list:


        itkimage = sitk.ReadImage('../data/valid/'+filename)
        itkimage = sitk.Flip(itkimage, [False, True, False]) #Flip to show correct.                        
        imgs = sitk.GetArrayFromImage(itkimage)

        if 'label' in filename.lower():
            imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
            masks.append( imgs )

        else:
            imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
            images.append(imgs )

    images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
    print("number of images:")
    print(len(images))
    masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
    print("number of annotations:")
    print(len(masks))		  
    masks = masks.astype(int)

    #Smooth images using CurvatureFlow
    images = smooth_images(images)
    print("smoothing finished!")
        


    np.save('../data/X_val.npy', images)
    np.save('../data/y_val.npy', masks)
        
    print("Saved val.npy")          


def load_data():

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')
			  
    print("loaded stuff")


    return X_train, y_train, X_val, y_val





def keras_fit_generator(img_rows=96, img_cols=96, batch_size=8, regenerate=True):
			  
#    if regenerate: #UNCOMMENT THE FIRST TIME TO CREATE THE .npy FILES!
#        data_to_array(img_rows, img_cols) #UNCOMMENT THE FIRST TIME TO CREATE THE .npy FILES!

    X_train, y_train, X_val, y_val = load_data()
    
    
    print("Train dataset shape: " + str(X_train.shape))
    print("Validation dataset shape: " + str(X_val.shape))
    
    plt.imsave("X_train.png", X_train[100,:,:,0], cmap='gray')
    plt.imsave("X_val.png", X_val[100,:,:,0], cmap='gray')
    plt.imsave("y_val.png", y_val[100,:,:,0], cmap='gray')
    plt.imsave("y_train.png", y_train[100,:,:,0], cmap='gray')

    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]     
    
    n_imgs_train = X_train.shape[0]
    n_imgs_valid = X_val.shape[0]

    # Provide the same seed and keyword arguments to the fit and flow methods
    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07 )
    # we create two instances with the same arguments
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
      
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2
    image_datagen.fit(X_train, seed=seed)
    mask_datagen.fit(y_train, seed=seed)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
    #train_generator = zip(image_generator, mask_generator)
       
    train_generator = (pair for pair in zip(image_generator, mask_generator))
    
    
    image_datagen_val = ImageDataGenerator(**data_gen_args)
    mask_datagen_val = ImageDataGenerator(**data_gen_args)

    seed = 2
    image_datagen_val.fit(X_val, seed=seed)
    mask_datagen_val.fit(y_val, seed=seed)
    image_generator_val = image_datagen_val.flow(X_val, batch_size=batch_size, seed=seed)
    mask_generator_val = mask_datagen_val.flow(y_val, batch_size=batch_size, seed=seed)        
    
    valid_generator = (pair for pair in zip(image_generator_val, mask_generator_val))
   
    


    model = UNet((img_rows, img_cols,1), start_ch=6, depth=5, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
#    model.load_weights('../data/weights.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint(
        '../data/weights.h5', monitor='val_loss', save_best_only=True)
    c_backs = [model_checkpoint]
    c_backs.append( EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15) )

    model.compile(  optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])


    model_history = model.fit_generator(
                        train_generator,
                        steps_per_epoch=n_imgs_train//batch_size,
                        epochs=1000,
                        verbose=2,
                        shuffle=True,
                        validation_data=valid_generator,
                        validation_steps = n_imgs_valid//batch_size,
                        callbacks=c_backs,
                        use_multiprocessing=True)	
    

    # list all data in history
    print(model_history.history.keys())
		
    acc = model_history.history['dice_coef']
    val_acc = model_history.history['val_dice_coef']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

	#Plot accuracy graph
    plt.plot(acc)
    plt.plot(val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.title('Model accuracy')
    #plt.legend()
    plt.legend(['Train', 'Validation'], loc='center right')	
    plt.savefig('accuracy.png')

	#Plot loss graph
    plt.clf()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
	#plt.legend()
    plt.legend(['Train', 'Validation'], loc='center right')	
    plt.savefig('loss.png')    
	

    

if __name__=='__main__':

    import time


    start = time.time()
    keras_fit_generator(img_rows=256, img_cols=256, regenerate=True, batch_size=128)


    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2 ) )


