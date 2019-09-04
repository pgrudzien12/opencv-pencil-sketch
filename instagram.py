
import cv2
import numpy as np
from dataPath import DATA_PATH

image = cv2.imread(DATA_PATH+"images/nature-640.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(DATA_PATH+"images/nature-640.jpg", 1)
targetImage__ = cv2.imread(DATA_PATH+"images/target.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("targetImage__", targetImage__)

def first_der(image):
    # Apply sobel filter along x direction
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    # Apply sobel filter along y direction
    sobely = cv2.Sobel(image,cv2.CV_32F,0,1)
    gradient = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
    cv2.normalize(gradient, 
                    dst = gradient, 
                    alpha = 0, 
                    beta = 1, 
                    norm_type = cv2.NORM_MINMAX, 
                    dtype = cv2.CV_32F)
                    
    _, gradient = cv2.threshold(gradient, 0.01,1,cv2.THRESH_TOZERO)
    gradient = 1-gradient
    return gradient
gradient = first_der(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(np.uint8(gradient*255))
dst_gray, dst_color = cv2.pencilSketch(image2, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
cv2.imshow("image ", dst_gray)
cv2.imshow("Laplacian 10 ", cl1)
cv2.waitKey(0)