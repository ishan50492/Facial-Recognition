import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import ndimage
import scipy.stats.stats as st
import scipy.stats as sgtono

img = cv2.imread('input.png')
print("Printing image")
print(img.shape)

ratio=sgtono.signaltonoise(img,axis=None)
print("Signal to noise ratio:",ratio)


b,g,r=cv2.split(img)
print("Printing b,g,r")
print("mean of b=",b.mean())
#print("variance of b=",b.variance())
hist=cv2.calcHist([img],[0],None,[256],[0,256])
print("Printing histomean:",hist.mean())
print("Printing histovar:",hist.var())

hog = cv2.HOGDescriptor()
h = hog.compute(img)

ent=0
for j in range(0,len(h)):
    if (h[j][0] > 0):
        ent=ent-1*h[j]*math.log(h[j])

print("Entropy of HOG for the image is ",ent[0])

var_b=np.array(b)
print("Variance of b:",var_b.var())
print("Mean of b:",var_b.mean())
sum_b=sum(map(sum,b))
print("mean_b=")
print(sum_b/(len(b)+len(b[0])))

heights=np.array(b)
print("Skewness:",len(st.skew(hist)))
print("Kurtosis:",st.kurtosis(hist))


sum_g=sum(map(sum,g))
print("mean_g=")
print(sum_g)

sum_r=sum(map(sum,r))
print("mean_r=")
print(sum_r)


average_color_per_row=np.average(img,axis=0)
print("Average color per row")
print(average_color_per_row)
average_color = np.average(average_color_per_row, axis=0)
print("Average color")
print(np.average(average_color))

print(img.shape)





edges = cv2.Canny(img,100,200)
print(edges.shape)
edge_count = np.count_nonzero(edges)

compression_ratio=len(img)*len(img[0])/62500
aspect_ratio=len(img[0]/len(img))


print("Edge count")
print(edge_count)

print("Sum of edges")
sum_of_edges=sum(map(sum,edges))
print(sum_of_edges)

print("Average length of edges:")
print(sum_of_edges/edge_count)


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
