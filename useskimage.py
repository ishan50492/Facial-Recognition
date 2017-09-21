import skimage.io as io
from skimage.color import rgb2gray
import scipy.misc

from PIL import Image
img = Image.open('input.png').convert('LA')
img.save('greyscale.png')

img = io.imread('input.png')

print("Original image shape:",img.shape)
img_grayscale = rgb2gray(img)


print("Grey scale image shape: ",img_grayscale.shape)

print("Image in matrix form:\n\n",img_grayscale)
