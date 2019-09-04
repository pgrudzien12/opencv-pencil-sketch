import numpy as np
import cv2

img_rgb = cv2.imread("data/images/nature-1920.jpg")
img_gray = cv2.imread("data/images/nature-1920.jpg", 0)
targetImage__ = cv2.imread("data/images/target.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("oryginal", img_rgb)
cv2.imshow("targetImage__", targetImage__)
image = img_rgb

#check if image exists
if image is None:
    print("can not find image")
    sys.exit()

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=5.0, threshold=0):
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
                    
    return gradient

hsvImage = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

result = hsvImage[:,:,2]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#result = cv2.dilate(img_gray, kernel)
blurred = cv2.bilateralFilter(img_gray, d = 3, sigmaColor = 40, sigmaSpace = 5)

kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32)
kernel=1/2*kernel
dst=cv2.filter2D(blurred,-1,kernel)


result = first_der(dst ) 
result = np.uint8(255*result)

#Apply bilateralFilter

lapl = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=3, scale=5, delta = 1)

#result = cv2.equalizeHist(lapl) 
resultA = np.uint8(np.clip(1.5 * result + 0, 0, 255))
resultB = np.uint8((np.float32(resultA) + lapl)/2.0)
#result = 255- cv2.dilate(lapl, kernel)

cv2.imshow("pencilsketch",255-lapl)
cv2.imshow("pencilsketch2",255-resultA)
cv2.imshow("pencilsketch3",255-resultB)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()