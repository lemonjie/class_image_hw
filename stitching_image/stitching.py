import cv2
import numpy as np
import math

def Set_mark_points():
    img1_point1_x = 1298; img1_point1_y = 2227
    img1_point2_x = 2302; img1_point2_y = 3175
    img1_point3_x = 2512; img1_point3_y = 3225

    img2_point1_x = 900; img2_point1_y = 2230
    img2_point2_x = 1927; img2_point2_y = 3152
    img2_point3_x = 2126; img2_point3_y = 3191

    img1_xy = [ img1_point1_x, img1_point1_y, img1_point2_x, img1_point2_y, img1_point3_x, img1_point3_y ]
    img2_xy = [ img2_point1_x, img2_point1_y, img2_point2_x, img2_point2_y, img2_point3_x, img2_point3_y ]

    return img1_xy, img2_xy

def Get_affine_coef( xy_origin , xy_trans):
    matrixA = np.array([
        [xy_origin[0], xy_origin[1], 0, 0, 1, 0],
        [0, 0, xy_origin[0], xy_origin[1], 0, 1],
        [xy_origin[2], xy_origin[3], 0, 0, 1, 0],
        [0, 0, xy_origin[2], xy_origin[3], 0, 1],
        [xy_origin[4], xy_origin[5], 0, 0, 1, 0],
        [0, 0, xy_origin[4], xy_origin[5], 0, 1]
    ])

    matrixB = np.array( xy_trans ).reshape(6, -1)

    matrixA_inv = np.linalg.inv(matrixA)
    affine_coef = matrixA_inv.dot(matrixB)
    return affine_coef

def Affine( x_origin, y_origin, coef_array ):
    a = coef_array[0].item()
    b = coef_array[1].item()
    c = coef_array[2].item()
    d = coef_array[3].item()
    e = coef_array[4].item()
    f = coef_array[5].item()

    x_trans = a*x_origin + b*y_origin + e
    y_trans = c*x_origin + d*y_origin + f
    return x_trans, y_trans

# read image
img1 = cv2.imread( 'images/first_two/DSC_2231.jpg' ) # left
img2 = cv2.imread( 'images/first_two/DSC_2230.jpg' ) # right

# set three point
img1_xy, img2_xy = Set_mark_points()

# get trans matrix from 1 to 2 and from 2 to 1
affine_1to2_coef = Get_affine_coef(img1_xy, img2_xy)
affine_2to1_coef = Get_affine_coef(img2_xy, img1_xy)

# get stitch1 size from 2 to 1
img1_height, img1_width = img1.shape[:2]
img2_height, img2_width = img2.shape[:2]

border_x = [0, img1_width]
border_y = [0, img1_height]

for i in [0, img2_width]:
    for j in [0, img2_height]:
        temp_x , temp_y = Affine(i, j, affine_2to1_coef)
        border_x.append(temp_x)
        border_y.append(temp_y)

stitch1_top = math.floor(min(border_y))
stitch1_bottom = math.ceil(max(border_y))
stitch1_left = math.floor(min(border_x))
stitch1_right = math.ceil(max(border_x))

stitch1_height = stitch1_bottom - stitch1_top + 1
stitch1_width = stitch1_right - stitch1_left + 1

# set stitch1 zeros
stitch1 = np.zeros( [ stitch1_height, stitch1_width, 3 ])

# put img1 and img2 on stitch1
img1_x_min = 0 - stitch1_left; img1_x_max = img1_width -1 - stitch1_left
img1_y_min = 0 - stitch1_top; img1_y_max = img1_height -1 - stitch1_top

for y in range(0, stitch1_height):
    print(y)
    for x in range(0, stitch1_width):
        if (img1_x_min <= x <= img1_x_max) and (img1_y_min <= y <= img1_y_max):
            stitch1[y,x,:] = img1[y+stitch1_top, x+stitch1_left, :]
        else:
            x_in_img2, y_in_img2 = Affine(x, y, affine_1to2_coef)
            if not ((0 <= x_in_img2 <= (img2_width-1)) and (0 <= y_in_img2 <= (img2_height-1))):
                continue
            dist_a = x_in_img2 - math.floor(x_in_img2)
            dist_b = y_in_img2 - math.floor(y_in_img2)
            for k in range(0,3):
                point_a = img2[math.floor(y_in_img2), math.floor(x_in_img2), k]
                point_b = img2[math.floor(y_in_img2), math.ceil(x_in_img2), k]
                point_c = img2[math.ceil(y_in_img2), math.floor(x_in_img2), k]
                point_d = img2[math.ceil(y_in_img2), math.ceil(x_in_img2), k]
                greyscale = point_a*(1-dist_a)*(1-dist_b) + point_b*(dist_a)*(1-dist_b) + \
                            point_c*(1-dist_a)*(dist_b) + point_d*(dist_a)*(dist_b)
                stitch1[y,x,k] = greyscale

# write stitched image
cv2.imwrite( 'stitch1.jpg', stitch1 )
