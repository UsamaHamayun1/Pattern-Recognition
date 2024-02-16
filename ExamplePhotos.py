# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:53:00 2023

@author: joao_
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% 
# This code serves simply to vizualize some of the images (examples)


imgVizualize2 = mpimg.imread('faces/an2i/an2i_left_angry_open_2.pgm')

plt.imshow(imgVizualize2, cmap='gray')
plt.axis('off')  # Turn off the axis labels
plt.show()

imgVizualize = mpimg.imread('faces/an2i/at33_left_neutral_sunglasses_2.pgm')

plt.imshow(imgVizualize, cmap='gray')
plt.axis('off')  # Turn off the axis labels
plt.show()

imgVizualize4 = mpimg.imread('faces/an2i/an2i_left_angry_open_4.pgm')

plt.imshow(imgVizualize4, cmap='gray')
plt.axis('off')  # Turn off the axis labels
plt.show()


imgVizualize5 = mpimg.imread('faces/an2i/an2i_up_angry_open_4.pgm')

plt.imshow(imgVizualize5, cmap='gray')
plt.axis('off')  # Turn off the axis labels
plt.show()

imgVizualize6 = mpimg.imread('faces/an2i/an2i_straight_sad_open_4.pgm')

plt.imshow(imgVizualize6, cmap='gray')
plt.axis('off')  # Turn off the axis labels
plt.show()

  






