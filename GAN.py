import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential


def load_data():
    # Define data directory and file extension
    data_dir = 'img'
    file_ext = '.png'
    data_ext = '.txt'

    # Load image data from directory
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(file_ext):
            img = Image.open(os.path.join(data_dir, file))
            data.append(np.array(img))
        if file.endswith(data_ext):
            with open('text.txt', 'r', encoding='utf-8') as file:
                lines = file.readlines()

    return data,lines

def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(7 * 7 * 512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((7, 7, 512)))

    model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(1, kernel_size=3, strides=(2, 4), padding='same', activation='tanh'))
    return model

# Tesseract OCR
# define the discriminator network
def build_discriminator(img_shape):
    
    model = Sequential()
    
    # input layer
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # print the summary of the discriminator network
    model.summary()

    # define the input image
    img = Input(shape=img_shape)
    
    # determine the validity of the input image
    validity = model(img)

    # define the discriminator model
    discriminator = Model(img, validity)

    return discriminator