import random
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as pyplot



target_size=7
cursor_size=3
display_width=25
display_height=25
scale=30

iterations = 3
img_raw_matrix = np.array([])

for i in range(0,iterations):
    target_pos = (random.randint(target_size,display_height-target_size-1),random.randint(target_size,display_width-target_size-1))
    cursor_pos = target_pos
    while target_pos == cursor_pos:
        cursor_pos = (random.randint(cursor_size,display_height-cursor_size-1),random.randint(cursor_size,display_width-cursor_size-1))



    img = Image.new('RGB', (display_height,display_width))
    draw = ImageDraw.Draw(img)

    target_y1 = target_pos[0]
    target_x1 = target_pos[1]

    cursor_y1 = target_y1 + target_size/2 - cursor_size/2
    cursor_x1 = target_x1 + target_size/2 - cursor_size/2

    # cursor_y1 = cursor_pos[0]
    # cursor_x1 = cursor_pos[1]

    draw.rectangle((target_y1,target_x1,target_y1+target_size,target_x1+target_size), fill=(0,0,0), outline=(255,255,255))
    draw.rectangle((cursor_y1,cursor_x1,cursor_y1+cursor_size,cursor_x1+cursor_size), fill=(0,0,0), outline=(255,255,255),width=100)

    img = img.convert('1')
    img_raw_matrix = np.append(img_raw_matrix, img)

img_raw_matrix.reshape(display_width*display_height, iterations)


target_size*=scale
cursor_size*=scale
display_height*=scale
display_width*=scale
target_x1 *=scale
target_y1*=scale
cursor_x1*=scale
cursor_y1*=scale
border_size=scale

big_img = Image.new('RGB', (display_height,display_width), (0,255,0))
draw = ImageDraw.Draw(big_img)
draw.rectangle((target_y1,target_x1,target_y1+target_size,target_x1+target_size), fill=(255,255,255), outline=(255,255,255))


draw.rectangle((target_y1+scale,target_x1+scale,target_y1+target_size-scale,target_x1+target_size-scale), fill=(0,0,0), outline=(255,255,255))
draw.rectangle((cursor_y1,cursor_x1,cursor_y1+cursor_size,cursor_x1+cursor_size), fill=(255,255,255), outline=(255,255,255),width=100)


# img = img.resize((80,50)).convert('1')
raw_img = (np.asarray(img)).flatten()
raw_img = [int(k) for k in raw_img]

big_img.show()