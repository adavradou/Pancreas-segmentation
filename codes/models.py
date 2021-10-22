"""
Created on Sun Feb 21 2020
@author: Agapi Davradou

This module contains the U-Net model's definition code.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import  UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from argparser import args
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from dataloader import *
from metrics import *
from tensorflow.keras.optimizers import Adam


class unet(object):
	def __init__(self, img_shape=(args.image_size, args.image_size, 1),
				start_ch=args.start_ch,
				out_ch=args.out_ch,
				depth=args.depth,
				batchnorm=args.batchnorm,
				dropout=args.dropout,
				maxpool=args.maxpool,
				upconv=args.upconv,
				residual=args.residual,
				inc_rate=args.inc_rate,
				activation=args.activation,
				model_name=args.model_name,
				print_model = args.print_model):

		self.img_shape = img_shape
		self.start_ch = start_ch
		self.out_ch = out_ch
		self.depth = depth
		self.batchnorm = batchnorm
		self.dropout = dropout
		self.maxpool = maxpool
		self.upconv = upconv
		self.residual = residual
		self.inc_rate = inc_rate
		self.activation = activation
		self.model_name = model_name
		self.print_model = print_model


	def create_model(self):
		return self.UNet()

	def get_model_name(self, k):
		return args.output_path + '/model_' + str(k) + '.h5'


	def conv_block(self, m, dim, acti, bn, res, do=0):

		init = VarianceScaling(scale=1.0/9.0 )
		n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init )(m)
		n = BatchNormalization()(n) if bn else n
		n = Dropout(do)(n) if do else n
		n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(n)
		n = BatchNormalization()(n) if bn else n

		return concatenate([n, m], axis=3) if res else n

	def level_block(self, m, dim, depth, inc, acti, do, bn, mp, up, res):
		if depth > 0:
			n = self.conv_block(m, dim, acti, bn, res)
			m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
			m = self.level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
			if up:
				m = UpSampling2D()(m)
				m = Conv2D(dim, 2, activation=acti, padding='same')(m)
			else:
				m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
			n = concatenate([n, m], axis=3)
			m = self.conv_block(n, dim, acti, bn, res)
		else:
			m = self.conv_block(m, dim, acti, bn, res, do)
		return m


	def UNet(self):

		logger = log(path=args.output_path, file="train.log" + __name__)

		i = Input(shape=self.img_shape)
		o = self.level_block(i, self.start_ch, self.depth, self.inc_rate, self.activation, self.dropout, self.batchnorm, self.maxpool, self.upconv, self.residual)
		o = Conv2D(self.out_ch, 1, activation='sigmoid')(o)

		model = Model(inputs=i, outputs=o)
		if self.print_model:
			model.summary(print_fn=logger.info)

		return model


	def get_callbacks(self):

		model_checkpoint = ModelCheckpoint(self.get_model_name(self.model_name), monitor='val_loss', save_best_only=True)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)
		csv_logger = CSVLogger(args.output_path + '/train_log.csv', append=True, separator=';')

		return [model_checkpoint, early_stopping, csv_logger]



