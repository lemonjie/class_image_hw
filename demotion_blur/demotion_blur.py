import cv2
import numpy as np

img = cv2.imread('images/grey_trees.jpg')

size = 15

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
print(kernel_motion_blur)

# applying the kernel to the input image
blurred = cv2.filter2D(img, -1, kernel_motion_blur)

cv2.imwrite('images/output_img/blur_grey_trees.jpg', blurred)