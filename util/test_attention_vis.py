import torch
from PIL import Image
import numpy as np
import cv2

def tensor2att1(attention):
    """
    Attention is numpy array
    """
    top_k = 3
    attention_shape = 64

    max_indices = np.argmax(attention, axis=1)
    max_scores= np.amax(attention, axis=1)

    k_largest_indices = np.argpartition(max_scores, -top_k)[-top_k:]

    out_img = np.ones(shape=(64,64,3), dtype=np.uint8)
    out_img = out_img * 127    #Empty gray image
    
    color = (0, 0, 255) #Red in OpenCV
    thickness = 1 #Thickness of the arrow
    shift = 0
    
    #Draw Arrows
    for i in range(top_k):

        end_idx = k_largest_indices[i]
        start_idx = max_indices[end_idx]

        start_x = start_idx // attention_shape
        start_y = start_idx % attention_shape
        
        end_x = end_idx // attention_shape
        end_y = end_idx % attention_shape

        start_point = (start_x, start_y)
        end_point = (end_x, end_y)

        arrow_length =((end_x - start_x)**2 +(end_y - start_y)**2 )**(0.5)

        tipLength = 5 / arrow_length

        out_img = cv2.arrowedLine(out_img, start_point, end_point, color, thickness, line_type=cv2.LINE_AA, shift=shift, tipLength=tipLength)

    #out_img = cv2.resize(out_img, None, fx=4, fy=4, interpolation= cv2.INTER_CUBIC)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(out_img)

    return pil_image

def tensor2att2(attention):

    attention_shape = 64

    index = np.random.randint(low=attention_shape*attention_shape, dtype=int)
    attention = attention[index]

    #Convert probabilities into 0-255
    attention = (attention *255).astype(np.uint8)
    attention = np.reshape(attention, (attention_shape, attention_shape))
    heat_map = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(heat_map)
    return pil_image, attention

if __name__=='__main__':
    attention= torch.rand( size=(4096,4096), dtype=torch.float32,requires_grad=False)
    attention = attention.detach().cpu().numpy()
    vis, attention = tensor2att2(attention)
    vis.save("heat.png")


    #attention = (attention * 255).astype(np.uint8)
    attention = Image.fromarray(attention)
    attention.save("original_att.png")

    

