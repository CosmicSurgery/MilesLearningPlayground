from PIL import Image
import numpy as np
im_size = 8
scale = 1

class display():
    def display():
        data = np.zeros((im_size,im_size,3),dtype=np.uint8)
        (x,y) = np.random.randint(0,data.shape[0]),np.random.randint(0,data.shape[1])
        data[x,y] = [0,255,0]
        while np.sum(data[x,y]) != 0:
            (x,y) = np.random.randint(0,data.shape[0]) , np.random.randint(0,data.shape[1])
        data[x,y] = [255, 0,0] 
        
        img = Image.fromarray(np.array(data), 'RGB')
        disp_img = img.resize((800,800), Image.NEAREST)
        return data, disp_img, img
    
