from PIL import Image
import pytesseract
import pandas as pd
from io import StringIO
import numpy as np
from math import sqrt

from GAN import load_data

img_filename0 = './res/0-0.png'
img_filename1 = './res/0-2.png'

#data,lines = load_data()
tsv_string = pytesseract.image_to_data(Image.open(img_filename0), config = 'tsv')
tsv_file = StringIO(tsv_string)
df = pd.read_csv(tsv_file, sep='\t')

filtered_data = df[df['conf'] != -1][['conf', 'text']]
filtered_data = filtered_data[filtered_data['text'] != ' ']

# Convert the filtered data to a NumPy array
result_np_array = filtered_data.to_numpy()

#print("\nNumPy Array:")
print(result_np_array)
# print(' '.join(i for i in result_np_array[:,1]))

tsv_string = pytesseract.image_to_string(Image.open(img_filename0))

print(tsv_string)

print('')
print('')

#data,lines = load_data()
tsv_string = pytesseract.image_to_data(Image.open(img_filename1), config = 'tsv')
tsv_file = StringIO(tsv_string)
df = pd.read_csv(tsv_file, sep='\t')

filtered_data = df[df['conf'] != -1][['conf', 'text']]
filtered_data = filtered_data[filtered_data['text'] != ' ']

# Convert the filtered data to a NumPy array
result_np_array = filtered_data.to_numpy()

#print("\nNumPy Array:")
print(result_np_array)
# print(' '.join(i for i in result_np_array[:,1]))

tsv_string = pytesseract.image_to_string(Image.open(img_filename1))

print(tsv_string)

image0 = np.array(Image.open(img_filename0))
image1 = np.array(Image.open(img_filename1))
image1 -= image0
image1 = abs(image1)
img = (image1 / 127.5) - 1
img = img.flatten()
dist = abs(np.linalg.norm(img))
print(sqrt(dist) - 1.2)





