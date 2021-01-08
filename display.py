import random
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

target_size=7
cursor_size=3
display_width=25
display_height=25
scale=30

iterations = 1000
x_train = []
pct = 0
for i in range(0,iterations):
    target_pos = (random.randint(target_size,display_height-target_size-1),random.randint(target_size,display_width-target_size-1))
    cursor_pos = target_pos
    while target_pos == cursor_pos:
        cursor_pos = (random.randint(cursor_size,display_height-cursor_size-1),random.randint(cursor_size,display_width-cursor_size-1))

    img = Image.new('RGB', (display_height,display_width))
    draw = ImageDraw.Draw(img)

    target_y1 = target_pos[0]
    target_x1 = target_pos[1]

    cursor_y1 = cursor_pos[0]
    cursor_x1 = cursor_pos[1]

    draw.rectangle((target_y1,target_x1,target_y1+target_size,target_x1+target_size), fill=(0,0,0), outline=(255,255,255))
    draw.rectangle((cursor_y1,cursor_x1,cursor_y1+cursor_size,cursor_x1+cursor_size), fill=(255,255,255), outline=(255,255,255))

    img = img.convert('1')
    img = (np.asarray(img).flatten())
    img = np.array([int(k) for k in img]).reshape(25,25)
    x_train.append(img)
    if i % int(iterations/100) == 0:
        pct +=1
        print(f' ~~~~ {pct}% completed ~~~~ ', end="\r")
        


np.save('train.npy', x_train)

test = np.load('train.npy')

print()




