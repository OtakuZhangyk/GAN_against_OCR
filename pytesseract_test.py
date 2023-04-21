from PIL import Image
import pytesseract
import pandas as pd
from io import StringIO

from GAN import load_data

#data,lines = load_data()
tsv_string = pytesseract.image_to_data(Image.open('img/0.png'), config = 'tsv')
tsv_file = StringIO(tsv_string)
df = pd.read_csv(tsv_file, sep='\t')

filtered_data = df[df['conf'] != -1][['conf', 'text']]

# Convert the filtered data to a 2D list
result_list = filtered_data.values.tolist()

# Convert the filtered data to a NumPy array
result_np_array = filtered_data.to_numpy()

print("2D List:")
print(result_list)

print("\nNumPy Array:")
print(result_np_array)
print(result_np_array[:,1])
print(result_np_array[:,1].shape)
print(result_np_array[:,1].shape[0])
print(' '.join(i for i in result_np_array[:,1]))