from keras.datasets import mnist
import keras
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, BatchNormalization, MaxPooling2D, Conv2DTranspose, Conv2D, InputLayer, Flatten, Reshape, UpSampling2D, Dropout, ReLU
from keras.models import Model
from keras import backend as K
from keras import metrics, Sequential


def build_CNNautoencoder(img_shape, intermediate_dim):
    # The encoder
    kernel_dim = 5
    input_img = Input(shape=(img_shape))
    h = Conv2D(10, kernel_size = kernel_dim, activation="relu", strides=1, padding="same")(input_img)
    h = MaxPooling2D(pool_size=(2,2), padding='same')(h)
    h = Conv2D(20, kernel_size = kernel_dim, activation="relu", strides=1, padding="same")(h)
    enc_output = MaxPooling2D(pool_size=(2,2), padding='same')(h)    
    x = Flatten()(enc_output)
    
    # the decoder
    latent_inputs = Input(shape=(x.shape[0],), name='z_sampling')
    h =Conv2DTranspose(20, kernel_size=kernel_dim,activation='relu',strides=1,padding='same')(enc_output)
    h = UpSampling2D((2, 2))(h)
    h =Conv2DTranspose(10, kernel_size=kernel_dim,activation='relu',strides=1,padding='same')(h)
    h = UpSampling2D((2, 2))(h)    
    dec_output =Conv2DTranspose(1, kernel_size=kernel_dim,strides=1,padding='same')(h)
    autoencoder = Model(input_img,dec_output)
    encoder = Model(input_img, x, name='encoder')
    decoder = Model(latent_inputs, dec_output, name='decoder')


    return encoder, decoder, autoencoder

def build_VAE(img_shape, intermediate_dim):
    # The encoder
    epsilon_std = 1.0
    kernel_dim = 11
    alpha = 1
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    input_img = Input(shape=(img_shape))
    h = Conv2D(10, kernel_size = (kernel_dim,kernel_dim), activation="relu", strides=1, padding="same")(input_img)
    h = MaxPooling2D(pool_size=(2,2), padding='same')(h)
    h = Conv2D(20, kernel_size = (kernel_dim,kernel_dim), activation="relu", strides=1, padding="same")(h)
    enc_output = MaxPooling2D(pool_size=(2,2), padding='same')(h)
    shape = K.int_shape(enc_output)
    x = Flatten()(enc_output)
    z_mean = Dense(latent_dim, name = 'z_mean')(x)
    z_log_var = Dense(latent_dim, name = 'z_log_var')(x)

    z = Lambda(sampling, output_shape=(latent_dim,), name = 'z')([z_mean, z_log_var])

    encoder = Model(input_img, [z_mean, z_log_var, z], name = 'encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x=Conv2DTranspose(32, kernel_size=(kernel_dim,kernel_dim),activation='relu',strides=1,padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x=Conv2DTranspose(64, kernel_size=(kernel_dim,kernel_dim),activation='relu',strides=1,padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    dec_output=Conv2DTranspose(1, kernel_size=(kernel_dim,kernel_dim), activation='relu',strides=1,padding='same')(x)
    decoder = Model(latent_inputs, dec_output, name='decoder')


    outputs = decoder(encoder(input_img)[2])
    vae = Model(input_img, outputs, name='vae')
    reconst_loss = mean_squared_error(K.flatten(input_img), K.flatten(outputs))
    reconst_loss *= X_train.shape[1] * X_train.shape[2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconst_loss + alpha*kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae

def visualize(img,encoder,autoencoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    code = encoder.predict(img[None])[0]
    reco = autoencoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img.reshape([img.shape[0],img.shape[1]]))

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//4,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(reco.reshape([img.shape[0],img.shape[1]]))
    plt.show()
