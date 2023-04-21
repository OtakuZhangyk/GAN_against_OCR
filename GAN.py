import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import pytesseract
import pandas as pd
from io import StringIO
import time
from skimage.metrics import structural_similarity as compare_ssim
import textdistance

# get list of filenames of train images
def load_data(data_dir = 'img49_1024'):
    # Define data directory and file extension
    file_ext = '.png'

    # Load image data from directory
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(file_ext):
            img = Image.open(os.path.join(data_dir, file))
            data.append(np.array(img))
    
    data = np.array(data)
    # expand a channel dimension to images, from (num of images,50,1080) to (num of images,50,1080,1)
    data = np.expand_dims(data, axis=3)
    return data

# apply noise array generated by generator to image
def apply_noise(image, noise):
    # Normalize image to range [-1, 1]
    normalized_image = (image / 127.5) - 1

    # Add noise to the normalized image
    noisy_image = normalized_image + noise

    # Clip values to range [-1, 1]
    noisy_image = np.clip(noisy_image, -1, 1)

    # Convert the noisy image back to the range [0, 255]
    noisy_image = ((noisy_image + 1) * 127.5).astype(np.uint8)

    return noisy_image

# Call Tesseract OCR to analyze image and return confidence and result in 2D np array
def OCR_(img_data):
    start_time = time.time()
    # add config tsv to return tsv format string including result and confidence
    tsv_string = pytesseract.image_to_data(img_data, config = 'tsv')
    end_time = time.time()
    # transform into file stream
    tsv_file = StringIO(tsv_string)
    # load using pandas
    df = pd.read_csv(tsv_file, sep='\t')

    # get data with conf != -1 and get their conf and text
    filtered_data = df[df['conf'] != -1][['conf', 'text']]

    # Convert the filtered data to a NumPy array
    result_np_array = filtered_data.to_numpy()

    # [conf text]
    return result_np_array, end_time - start_time

# Levenshtein Distance (Edit Distance), normalized to 0~1, 1 for same
def str_similarity(string1, string2):
    distance = textdistance.levenshtein.distance(string1, string2)
    max_length = max(len(string1), len(string2))
    similarity = 1 - (distance / max_length)
    return similarity

# evaluate the performance of the generator on each pair of real and fake image
def evaluate(real_np, real_time, fake_np, fake_time, img_sim):
    # Define the weights for each factor
    w_str_sim = -1.0
    w_confidence = 1.0
    w_time = 0.1
    w_img_sim = 1.0

    real_str_np = real_np[:,1]
    fake_str_np = fake_np[:,1]
    real_conf_np = real_np[:,0]
    fake_conf_np = fake_np[:,0]

    # string similiarity(accuracy)
    real_str = ' '.join(i for i in real_str_np)
    fake_str = ' '.join(i for i in fake_str_np)
    str_sim = str_similarity(real_str, fake_str)

    # confidence drop(negative confidence if wrong), nromalized
    if str_sim == 1.0:
        conf_drop = np.mean(real_conf_np) - np.mean(fake_conf_np)
    else:
        for i in range(min(real_str_np.shape[0], fake_str_np.shape[0])):
            if real_str_np[i] != fake_str_np[i]:
                fake_conf_np[i] = -fake_conf_np[i]
        conf_drop = np.mean(real_conf_np) - np.mean(fake_conf_np)
    conf_drop /= 2.0

    # time increase, normalized
    time_incre = (fake_time - real_time) / fake_time

    # evaluation equation
    eval_ = w_str_sim * str_sim + w_confidence * conf_drop + w_time * time_incre + w_img_sim * img_sim

    eval_ /= w_str_sim + w_confidence + w_time + w_img_sim

    return eval_

# transform np array to PIL image object
def numpy_to_pil_image(np_array):
    # Convert the array values to the range [0, 255]
    np_array = (np_array * 127.5 + 127.5).astype(np.uint8)

    # Remove the channel dimension if it's 1
    if np_array.shape[2] == 1:
        np_array = np_array.reshape(np_array.shape[0], np_array.shape[1])

    # Create a PIL.Image object from the NumPy array
    pil_image = Image.fromarray(np_array)
    return pil_image

def standard_gen_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)

# take latent_dim noise vector and generate 50*1080*1 grey scale image
def build_generator(latent_dim, img_shape):
    model = Sequential()

    flattened_img_size = img_shape[0] * img_shape[1] * img_shape[2]

    model.add(Dense(7 * 7 * 512, input_dim=latent_dim + flattened_img_size))
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

    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
    # reshape to image shape
    model.add(Reshape((49, 1024, 1)))
    return model

# get ssim of real and fake image
def ssim(image1, image2):
    return compare_ssim(image1, image2, multichannel=True)

# Define the training loop
def train(epochs, batch_size, latent_dim):
    X_train = load_data()

    generator = build_generator(latent_dim, (49, 1024, 1))

    learning_rate = 0.0002
    beta_1 = 0.5

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

    # Training loop
    for epoch in range(epochs):
        # Select a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        # Generate a batch of noise_arrays
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict([noise, real_images])
        print(fake_images.shape)
        return

        batch_eval_list = []
        # Evaluate real and fake images using unknown_func
        for i in range(real_images.shape[0]):
            fake_image = apply_noise(real_images[i], noise_arrays[i])

            real_np, real_time = OCR_(numpy_to_pil_image(real_images[i]))
            fake_np, fake_time = OCR_(numpy_to_pil_image(fake_image))
            sim = ssim(real_images[i], fake_image)
            eval_ = evaluate(real_np, real_time, fake_np, fake_time, sim)
            batch_eval_list.append(eval_)

        batch_eval_np = np.array(batch_eval_list)
        real_labels = np.ones((batch_size, 1))

        # Calculate loss and gradients for the generator
        with tf.GradientTape() as gen_tape:
            gen_loss = standard_gen_loss(real_labels, batch_eval_np)
            
            # Calculate gradients
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)

        # Update the generator's weights
        optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        # Print progress
        print("Epoch: %d, Generator Loss: %f" % (epoch, gen_loss))

def main():
    # Set the training parameters and train the GAN
    latent_dim = 100
    epochs = 100
    batch_size = 32
    train(epochs, batch_size, latent_dim)

if __name__ == "__main__":
    main()