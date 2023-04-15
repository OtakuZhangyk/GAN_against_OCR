import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define the image size and text parameters
img_size = (1080, 50)
font_size = 24
font_color = (0, 0, 0)  # Black
bg_color = (255, 255, 255)  # White
max_words_per_image = 12

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

# Create images with 10-20 words per image
for i in range(0, len(words), max_words_per_image):
    # Get words for this image
    image_words = words[i:i+max_words_per_image]
    text = ' '.join(image_words)


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

    img_filename = f'./img/{i//max_words_per_image}.png'
    
    cv2.imwrite(img_filename, gray_img)

    print(f'Saved image {img_filename}')
    #exit()
    out_txt += text + '\n'

with open('./img/splitted_text.txt', 'w', encoding='utf-8') as file:
    file.write(out_txt)