from __future__ import division, print_function

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
from scipy.ndimage import morphology
import tensorflow as tf

   
    
def Active_Contour_Loss(y_true2, y_pred2): 

    y_true2 = K.cast(y_true2, dtype = 'float64') #.reshape
    y_pred2 = K.cast(y_pred2, dtype = 'float64') #author  
    

    y_true = tf.transpose(y_true2, [0, 3, 1, 2])
    y_pred = tf.transpose(y_pred2, [0, 3, 1, 2])
    
    
    # print("y true is: " + str(y_true.get_shape()))
    # print("y pred is: " + str(y_pred.get_shape()))
    

    """
	lenth term
    """

    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

    delta_x = x[:,:,1:,:-2]**2
    delta_y = y[:,:,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y) 

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

    """
    region term
    """

    C_1 = np.ones((256, 256))
    C_2 = np.zeros((256, 256))

    region_in = K.abs(K.sum( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
    region_out = K.abs(K.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

    # lambdaP = 0.01 # lambda parameter could be various.
    
    #Hyperparameters. The loss is sensitive to unbalanced classes problem on different datasets.
    a = 0.01
    b = 0.01
	
    # loss =  lenth + lambdaP * (region_in + region_out) 
    loss =  lenth + (a * region_in + b * region_out) 
    
    # print("LOSS: " + str(loss) + "length: " + str(lenth) + "region in: " + str(region_in) + "region out: " + str(region_out))

    return loss


def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #Added it because of error: 
    #cannot compute Mul as input #1(zero-based) was expected to be a int64 tensor but is a float tensor

    #K.cast(K.argmax(y_true, axis=1),dtype='float64')
    #y_true_f = K.cast(K.flatten(y_true),dtype='int64')
    #y_pred_f = K.cast(K.flatten(y_pred),dtype='int64')
    #y_true_f = tf.convert_to_tensor(y_true_f, dtype=tf.float32)
    #y_pred_f = tf.convert_to_tensor(y_pred_f, dtype=tf.float32)
    
    y_true_f = tf.cast(y_true_f, dtype=tf.float32) 
    y_pred_f = tf.cast(y_pred_f, dtype=tf.float32)


    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred) #Changed to 1-dice_coef instead of -dice_coef, in order not to have negative loss.


def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )


def rel_abs_vol_diff(y_true, y_pred):

    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)


def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)



def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    
    #S = input_1 - morphology.binary_erosion(input_1, conn)
    #Changed it because if error: 
    #TypeError: numpy boolean subtract, the `-` operator, is not supported, 
    #use the bitwise_xor, the `^` operator, or the logical_xor function instead.
    
    input_1 = input_1.astype(np.int)    
    input_1_new = morphology.binary_erosion(input_1, conn)    
    input_1_new = input_1_new.astype(np.int)
    
    S = input_1 - input_1_new
   
    
    #Sprime = input_2 - morphology.binary_erosion(input_2, conn)
    
    input_2 = input_2.astype(np.int)    
    input_2_new = morphology.binary_erosion(input_2, conn)   
    input_2_new = input_2_new.astype(np.int)

    Sprime = input_2 - input_2_new

    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds
