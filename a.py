import numpy as np
import cv2

def first_der(image):
    # Apply sobel filter along x direction
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    # Apply sobel filter along y direction
    sobely = cv2.Sobel(image,cv2.CV_32F,0,1)
    gradient = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
    cv2.normalize(gradient, dst = gradient, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                    
    return gradient

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

targetImg = cv2.imread("data/images/pencilSketch.jpg")
cv2.imshow('targetImg',targetImg)

img_rgb = cv2.imread("data/images/trump.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_rgb',img_rgb)
cv2.imshow('img_gray',img_gray)


blurred = cv2.GaussianBlur(img_gray, (3,3), 5)
result = np.uint8(255*( first_der(blurred)))

cv2.imshow('sobel',255- result)



kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]], np.float32)
dst=cv2.filter2D(result,-1,1/2*kernel)
cv2.imshow('sobel+sharpened',255- dst)


blurred = unsharp_mask(img_gray, (5,5), 5, 1, 127)
result = first_der(blurred)
dst=cv2.filter2D(result,-1,1/3*kernel)
dst = cv2.normalize(dst, dst = dst, alpha = 0.5, beta = 1, norm_type=cv2.NORM_L1)
cv2.imshow('unsharp_mask+sobel', 1-result)



blurred = unsharp_mask(img_gray, (5,5), 5, 1, 66)
lapl = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3, scale=3, delta = 1)
cv2.imshow('Laplacian', 255-lapl)


blurred = unsharp_mask(img_gray, (5,5), 7, 1, 66)
lapl = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3, scale=3, delta = 1)
kernel = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]], np.float32)
kernel=1/9*kernel
laplSharp=cv2.filter2D(lapl,-1,kernel)
cv2.imshow('Laplacian+sharpened',255- laplSharp)

img_norm = cv2.equalizeHist(img_gray)
blurred = unsharp_mask(img_gray, (5,5), 5, 1, 66)
lapl = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3, scale=3, delta = 1)
cv2.imshow('normalize+Laplacian',255-lapl)

#cv2.imshow('win', dst)
cv2.waitKey(0)