import cv2
import numpy as np
from pathlib import Path

COLOR = 3

# set sobel_x, sobel_y, laplacian
sobel_x = np.array([ [-1,-2,-1],
                     [ 0, 0, 0],
                     [ 1, 2, 1] ]) # get horizontal edge

sobel_y = np.array([ [-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1] ]) # get vertical edge

laplacian = np.array([ [-1,-1,-1],
                       [-1, 8,-1],
                       [-1,-1,-1] ]) # get point and line

# read image
origin_img = cv2.imread( 'images/cupcake.jpg' )

# get sobel_img and laplacian_img
height = origin_img.shape[0]
width = origin_img.shape[1]

sobel_img = np.zeros( [ height+4, width+4, COLOR ])
laplacian_img = np.zeros( [ height+4, width+4, COLOR ])
padding_origin = np.zeros( [ height+4, width+4, COLOR ])
padding_origin[2:-2, 2:-2,:] = origin_img

cov_range = int((sobel_x.shape[0]-1)/2)
for y in range(cov_range, height+4-cov_range):
    print(y)
    for x in range(cov_range, width+4-cov_range):
        part_img = padding_origin[ y-1:y+2, x-1:x+2, : ]
        
        # convolution with sobel
        for color in range(COLOR):
            value = abs(np.sum(part_img[:,:,color] * sobel_x)) + abs(np.sum(part_img[:,:,color] * sobel_y))
            sobel_img[y,x,color] = value

        # convolution with laplacian
        for color in range(COLOR):
            value = np.sum(part_img[:,:,color] * laplacian)
            laplacian_img[y,x,color] = value

sobel_img = sobel_img[2:-2, 2:-2,:]
laplacian_img = laplacian_img[2:-2, 2:-2,:]

# write sobel_img and laplacian_img
output_dir = Path.cwd() / 'images/output_img'
Path.mkdir(output_dir, exist_ok=True)

cv2.imwrite( 'images/output_img/sobel.jpg', sobel_img )
cv2.imwrite( 'images/output_img/laplacian.jpg', laplacian_img )

laplacian_sharp = origin_img + laplacian_img
cv2.imwrite( 'images/output_img/laplacian_sharp.jpg', laplacian_sharp )

# blur sobel_img
blured3_sobel_img = cv2.blur(sobel_img, (3, 3))
blured5_sobel_img = cv2.blur(sobel_img, (5, 5))
blured7_sobel_img = cv2.blur(sobel_img, (7, 7))
blured9_sobel_img = cv2.blur(sobel_img, (9, 9))
cv2.imwrite( 'images/output_img/blured3_sobel.jpg', blured3_sobel_img )
cv2.imwrite( 'images/output_img/blured5_sobel.jpg', blured5_sobel_img )
cv2.imwrite( 'images/output_img/blured7_sobel.jpg', blured7_sobel_img )
cv2.imwrite( 'images/output_img/blured9_sobel.jpg', blured9_sobel_img )

# normalize blured_sobel_img
blured_sobel_max = np.amax(blured3_sobel_img)
blured_sobel_min = np.amin(blured3_sobel_img)
blured3_sobel_img = (blured3_sobel_img-blured_sobel_min) / (blured_sobel_max-blured_sobel_min)

blured_sobel_max = np.amax(blured5_sobel_img)
blured_sobel_min = np.amin(blured5_sobel_img)
blured5_sobel_img = (blured5_sobel_img-blured_sobel_min) / (blured_sobel_max-blured_sobel_min)

blured_sobel_max = np.amax(blured7_sobel_img)
blured_sobel_min = np.amin(blured7_sobel_img)
blured7_sobel_img = (blured7_sobel_img-blured_sobel_min) / (blured_sobel_max-blured_sobel_min)

blured_sobel_max = np.amax(blured9_sobel_img)
blured_sobel_min = np.amin(blured9_sobel_img)
blured9_sobel_img = (blured9_sobel_img-blured_sobel_min) / (blured_sobel_max-blured_sobel_min)

# get masked laplacian_img
masked3_laplacian_img = blured3_sobel_img * laplacian_img
cv2.imwrite( 'images/output_img/masked3_laplacian_img.jpg', masked3_laplacian_img )

masked5_laplacian_img = blured5_sobel_img * laplacian_img
cv2.imwrite( 'images/output_img/masked5_laplacian_img.jpg', masked5_laplacian_img )

masked7_laplacian_img = blured7_sobel_img * laplacian_img
cv2.imwrite( 'images/output_img/masked7_laplacian_img.jpg', masked7_laplacian_img )

masked9_laplacian_img = blured9_sobel_img * laplacian_img
cv2.imwrite( 'images/output_img/masked9_laplacian_img.jpg', masked9_laplacian_img )

# get result_img
result3_img = origin_img + masked3_laplacian_img
cv2.imwrite( 'images/output_img/sharpend3_img.jpg', result3_img )

result5_img = origin_img + masked5_laplacian_img
cv2.imwrite( 'images/output_img/sharpend5_img.jpg', result5_img )

result7_img = origin_img + masked9_laplacian_img
cv2.imwrite( 'images/output_img/sharpend7_img.jpg', result7_img )

result9_img = origin_img + masked9_laplacian_img
cv2.imwrite( 'images/output_img/sharpend9_img.jpg', result9_img )
