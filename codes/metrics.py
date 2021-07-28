"""
Created on Sun Feb 21 2020
@author: Agapi Davradou

This module contains the metrics used for training and testing.
"""

from __future__ import division, print_function
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf

   

def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = tf.cast(y_true_f, dtype=tf.float32) 
    y_pred_f = tf.cast(y_pred_f, dtype=tf.float32)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred) #Changed to 1-dice_coef instead of -dice_coef, in order to avoid negative loss.


