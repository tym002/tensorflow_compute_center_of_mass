#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:06:44 2021

@author: tianyu
"""

from keras import backend as K
import tensorflow as tf

def COM(feature,nx,ny,nz=None):
    '''
    COM computes the center of mass of the input 4D or 5D image 
    To use COM in a tensorflow model, use layers.Lambda
    Arguments: 
        feature: input image of 5D tensor with format [batch,x,y,z,channel]
                    or 4D tensor with format [batch,x,y,channel]
        nx,ny,nz: dimensions of the input image, if using 4D tensor, nz = None
    '''
    
    map1 = feature
    n_dim = map1.get_shape().ndims
    
    if n_dim == 5:
        x = K.sum(map1, axis =(2,3))
    else:
        x = K.sum(map1, axis = 2)

    r1 = tf.range(0,nx, dtype = 'float32')
    r1 = K.reshape(r1, (1,nx,1))
    
    x_product = x*r1
    x_weight_sum = K.sum(x_product,axis = 1,keepdims=True)+0.00001
    x_sum = K.sum(x,axis = 1,keepdims=True)+0.00001
    cm_x = tf.divide(x_weight_sum,x_sum)
    
    if n_dim == 5:
        y = K.sum(map1, axis =(1,3))
    else:
        y = K.sum(map1, axis = 1)

    r2 = tf.range(0,ny, dtype = 'float32')
    r2 = K.reshape(r2, (1,ny,1))
    
    y_product = y*r2
    y_weight_sum = K.sum(y_product,axis = 1,keepdims=True)+0.00001
    y_sum = K.sum(y,axis = 1,keepdims=True)+0.00001
    cm_y = tf.divide(y_weight_sum,y_sum)
    
    if n_dim == 5:
        z = K.sum(map1, axis =(1,2))
    
        r3 = tf.range(0,nz, dtype = 'float32')
        r3 = K.reshape(r3, (1,nz,1))
        
        z_product = z*r3
        z_weight_sum = K.sum(z_product,axis = 1,keepdims=True)+0.00001
        z_sum = K.sum(z,axis = 1,keepdims=True)+0.00001
        cm_z = tf.divide(z_weight_sum,z_sum)
    
        center_mass = tf.concat([cm_x,cm_y,cm_z],axis=1)
    else:
        center_mass = tf.concat([cm_x,cm_y],axis=1)

    return center_mass
