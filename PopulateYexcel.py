import numpy as np
import cv2
import glob
import math
import scipy.stats.stats as st
import scipy.stats as sgtono
from openpyxl import Workbook

#Number of features taken into consideration
Number_of_features=20

#Loading the image dataset for testing
cv_img_test = []
for img in glob.glob("/Users/ishan/PycharmProjects/FacialRecog/images1/testing/*.jpg"):
    n= cv2.imread(img)
    cv_img_test.append(n)

# Y is the testing dataset
print("Total number of images in the testing dataset:",len(cv_img_test))

print("Extracting features of Testing Dataset:")
Y=[[0 for j in range(Number_of_features)] for i in range(len(cv_img_test))]



for i in range(0,len(cv_img_test)):
    print("Working on image : ", i)
    # Feature 1: number of edges in an image
    edges=cv2.Canny(cv_img_test[i],100,200)
    edge_count = np.count_nonzero(edges)
    #print("Edge count for",i," image is ",edge_count)
    Y[i][0]=edge_count

    # Feature 2:Sum of edges in an image
    sum_of_edges = sum(map(sum, edges))
    #print("Sum of edges for ",i," image is ",sum_of_edges)
    Y[i][1] = sum_of_edges

    # Feature 3:Average length of edges in an image
    average_length=sum_of_edges/edge_count
    #print("Average length of edges for ", i, " image is ", average_length)
    Y[i][2] = average_length

    # Feature 4:Compression ratio of an image
    compression_ratio = len(cv_img_test[i]) * len(cv_img_test[i]) / 62500
    #print("Compression ratio for",i,"image is",compression_ratio)
    Y[i][3] = compression_ratio

    # Feature 5:Aspect ratio of an image
    aspect_ratio = len(cv_img_test[i]/len(cv_img_test[i]))
    #print("Aspect ratio for", i, "image is", aspect_ratio)
    Y[i][4] = aspect_ratio

    # Feature 6:Entropy of HOG descriptor of an image
    hog = cv2.HOGDescriptor()
    h = hog.compute(cv_img_test[i])
    ent=0
    for j in range(0,len(h)):
        if (h[j][0] > 0):
            ent=ent-1*h[j]*math.log(h[j])

    #print("Entropy of HOG for the image is ",ent[0])
    Y[i][5] = ent[0]


    hist_b = cv2.calcHist([cv_img_test[i]], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([cv_img_test[i]], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([cv_img_test[i]], [2], None, [256], [0, 256])

    b, g, r = cv2.split(cv_img_test[i])
    # Feature 7: mean of blue band of an image
    b_mean=b.mean()
    #print("Value of mean of blue band",b_mean)
    Y[i][6] = b_mean

    # Feature 8: mean of green band of an image
    g_mean = g.mean()
    #print("Value of mean of green band", g_mean)
    Y[i][7] = g_mean

    # Feature 9: mean of red band of an image
    r_mean = r.mean()
    #print("Value of mean of red band", r_mean)
    Y[i][8] = r_mean

    # Feature 10: variance of blue band of an image
    var_b = hist_b.var()
    #print("Variance of b:", var_b)
    Y[i][9] = var_b

    # Feature 11: variance of green band of an image
    var_g = hist_g.var()
    #print("Variance of g:", var_g)
    Y[i][10] = var_g

    # Feature 12: variance of red band of an image
    var_r = hist_r.var()
    #print("Variance of r:", var_r)
    Y[i][11] = var_r

    # Feature 13: skewness of blue band of an image
    skew_b = st.skew(hist_b)
    #print("Skew b:",skew_b[0])
    Y[i][12]=skew_b[0]

    # Feature 14: skewness of green band of an image
    skew_g = st.skew(hist_g)
    #print("Skew g:", skew_g[0])
    Y[i][13] = skew_g[0]

    # Feature 15: skewness of red band of an image
    skew_r = st.skew(hist_r)
    #print("Skew r:", skew_r[0])
    Y[i][14] = skew_r[0]

    # Feature 16: Kurtosis of blue band of an image
    kurt_b = st.kurtosis(hist_b)
    #print("Kurt b:", kurt_b[0])
    Y[i][15] = kurt_b[0]

    # Feature 17: Kurtosis of green band of an image
    kurt_g = st.kurtosis(hist_g)
    #print("Kurt g:", kurt_g[0])
    Y[i][16] = kurt_g[0]

    # Feature 18: Kurtosis of red band of an image
    kurt_r = st.kurtosis(hist_r)
    #print("Kurt r:", kurt_r[0])
    Y[i][17] = kurt_r[0]

    # Feature 19: Signal to noise ratio of an image
    noiseratio=sgtono.signaltonoise(cv_img_test,axis=None)
    #print("SgtoN:",float("{0:.2f}".format(noiseratio)))
    Y[i][18]=noiseratio.tolist()

    # Feature 20: Average color of an image
    average_color_per_row = np.average(cv_img_test[i], axis=0)
    average_color = np.average(np.average(average_color_per_row, axis=0))
    #print("average_color of the image :", average_color)
    Y[i][19] = average_color

# Writing Y to an excel file : Y.xlsx
wb = Workbook()
ws = wb.active

for i in range(0,len(Y)):
    for j in range(0,len(Y[0])):
        ws[chr(ord('A')+j)+str(i+1)] = Y[i][j]

wb.save("Y.xlsx")
