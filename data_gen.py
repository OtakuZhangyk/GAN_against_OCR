import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re

# Example string
my_string = "This is a test string with, non-English chars: 漢字, ひらがな and a number: 123.4"

# Use regular expressions to remove non-alphabetic characters, spaces, commas, and dots
pattern = re.compile('[^a-zA-Z\s,.0-9]')
my_string = pattern.sub('', my_string)

# Print the cleaned string
print(my_string)
exit()

# Define the image size and text parameters
img_size = (1024, 64)
font_size = 26
font_color = (0, 0, 0)  # Black
bg_color = (255, 255, 255)  # White
max_chrs_per_image = 70

floder_name = 'img64_1024_1'


# Open text file and read lines
with open('texts.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create list of words
words = []
for line in lines:
    # Ignore empty lines
    if not line.strip():
        continue
    words.extend(line.strip().split())

out_txt = ''

text = ''
img_num = 0
while words:
    w = words.pop(0)
    while len(text) + len(w) < 70:
        text += ' ' + w
        w = words.pop(0)
    new_text = w

    # Create a blank image
    img = Image.new('RGB', img_size, color=bg_color)

    # Get a font
    font = ImageFont.truetype('arial.ttf', font_size)

    # Get a drawing context
    draw = ImageDraw.Draw(img)

    # Get the text size
    text_size = draw.textsize(text, font=font)

    # Calculate the text position
    text_pos = ((img_size[0] - text_size[0]) // 2, (img_size[1] - text_size[1]) // 2)

    # Draw the text on the image
    draw.text(text_pos, text, fill=font_color, font=font)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Convert the image from RGB to grayscale
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    img_filename = f'./{floder_name}/{img_num}.png'
    
    cv2.imwrite(img_filename, gray_img)

    print(f'Saved image {img_filename}')
    #exit()
    out_txt += text + '\n'

    img_num += 1
    text = new_text

with open(f'./{floder_name}/splitted_text.txt', 'w', encoding='utf-8') as file:
    file.write(out_txt)