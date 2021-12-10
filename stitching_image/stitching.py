import cv2
import numpy as np
import math

X = 0; Y = 1
HEIGHT = 0; WIDTH = 1
P1 = 0; P2 = 1; P3 = 2 # those three points we choose
IMG_L = 0; IMG_R = 1 # left image or right image at the overlap between two images
L_TO_R = 0; R_TO_L = 1 # L_TO_R means left affine to right
STITCH_12 = 0; STITCH_23 = 1; STITCH_34 = 2 # STITCH_12 means stitching image 1 & 2
STITCH_LIST = [STITCH_12, STITCH_23, STITCH_34]

def Set_3_mark_points():
    # img 1 2 mapping
    img12_point1_x = 2841; img12_point1_y = 2028
    img12_point2_x = 2642; img12_point2_y = 2336
    img12_point3_x = 2871; img12_point3_y = 3207

    img21_point1_x = 2223; img21_point1_y = 2085
    img21_point2_x = 2046; img21_point2_y = 2376
    img21_point3_x = 2264; img21_point3_y = 3204

    # img 2 3 mapping
    img23_point1_x = 3019; img23_point1_y = 2150
    img23_point2_x = 2731; img23_point2_y = 2453
    img23_point3_x = 3000; img23_point3_y = 3491

    img32_point1_x = 1993; img32_point1_y = 2324
    img32_point2_x = 1755; img32_point2_y = 2599
    img32_point3_x = 2012; img32_point3_y = 3567

    # img 3 4 mapping
    img34_point1_x = 2896; img34_point1_y = 2126
    img34_point2_x = 2538; img34_point2_y = 2632
    img34_point3_x = 3021; img34_point3_y = 3246

    img43_point1_x = 2201; img43_point1_y = 2331
    img43_point2_x = 1882; img43_point2_y = 2798
    img43_point3_x = 2322; img43_point3_y = 3386

    img12_mapping = [ [[img12_point1_x, img12_point1_y], [img12_point2_x, img12_point2_y], [img12_point3_x, img12_point3_y]],
                     [[img21_point1_x, img21_point1_y], [img21_point2_x, img21_point2_y], [img21_point3_x, img21_point3_y]] ]
    img23_mapping = [ [[img23_point1_x, img23_point1_y], [img23_point2_x, img23_point2_y], [img23_point3_x, img23_point3_y]],
                      [[img32_point1_x, img32_point1_y], [img32_point2_x, img32_point2_y], [img32_point3_x, img32_point3_y]] ]
    img34_mapping = [ [[img34_point1_x, img34_point1_y], [img34_point2_x, img34_point2_y], [img34_point3_x, img34_point3_y]],
                      [[img43_point1_x, img43_point1_y], [img43_point2_x, img43_point2_y], [img43_point3_x, img43_point3_y]] ]

    return [ img12_mapping, img23_mapping, img34_mapping] # stitch edge -> edge left/right -> point 1/2/3 -> x/y

def Get_affine_coef( xy_origin, xy_trans ):
    matrixA = np.array([
        [xy_origin[P1][X], xy_origin[P1][Y], 0, 0, 1, 0],
        [0, 0, xy_origin[P1][X], xy_origin[P1][Y], 0, 1],
        [xy_origin[P2][X], xy_origin[P2][Y], 0, 0, 1, 0],
        [0, 0, xy_origin[P2][X], xy_origin[P2][Y], 0, 1],
        [xy_origin[P3][X], xy_origin[P3][Y], 0, 0, 1, 0],
        [0, 0, xy_origin[P3][X], xy_origin[P3][Y], 0, 1]
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

# read image from left to right
img1 = cv2.imread( 'images/series4/DSC_2214.jpg' )
img2 = cv2.imread( 'images/series4/DSC_2213.jpg' )
img3 = cv2.imread( 'images/series4/DSC_2212.jpg' )
img4 = cv2.imread( 'images/series4/DSC_2211.jpg' )
imgs = [ img1, img2, img3, img4 ]

# set three point
xy_mapping = Set_3_mark_points()

# get trans matrix from 1 to 2 and from 2 to 1
affine_coef = [] # stitch edge -> affine left_to_right / right_to_left
for i in STITCH_LIST:
    coef_LtoR = Get_affine_coef(xy_mapping[i][IMG_L], xy_mapping[i][IMG_R])
    coef_RtoL = Get_affine_coef(xy_mapping[i][IMG_R], xy_mapping[i][IMG_L])
    current_stitch_coef = [coef_LtoR, coef_RtoL]
    affine_coef.append(current_stitch_coef)

# get stitched size
imgs_size = []
for img in imgs:
    current_size = img.shape[:2]
    imgs_size.append(list(current_size))

border_x = [0, imgs_size[0][WIDTH]]
border_y = [0, imgs_size[0][HEIGHT]]

stitch_part = 0
for img_size in imgs_size[1:]:
    for x in [0, img_size[WIDTH]]:
        for y in [0, img_size[HEIGHT]]:
            current_affine = stitch_part
            affined_x = x; affined_y = y
            while (current_affine >= 0):
                affined_x , affined_y = Affine(affined_x, affined_y, affine_coef[current_affine][R_TO_L]) # eg. img4 trans to 3, then trans to 2, then tans to 1
                current_affine = current_affine - 1
            border_x.append(affined_x)
            border_y.append(affined_y)
    stitch_part = stitch_part + 1

stitched_top = math.floor(min(border_y))
stitched_bottom = math.ceil(max(border_y))
stitched_left = math.floor(min(border_x))
stitched_right = math.ceil(max(border_x))

stitched_height = stitched_bottom - stitched_top + 1
stitched_width = stitched_right - stitched_left + 1

# set stitched zeros
stitched = np.zeros( [ stitched_height, stitched_width, 3 ])

# draw on stitched ndarray
for y in range(0, stitched_height):
    print(y)
    for x in range(0, stitched_width):
        if (0 <= (x + stitched_left) <= (img1_width - 1)) and (0 <= (y + stitched_top) <= (img1_height - 1)):
            stitched[y,x,:] = img1[y+stitched_top, x+stitched_left, :]
        else:
            x_in_img2, y_in_img2 = Affine(x+stitched_left, y+stitched_top, affine_coef[STITCH_12][L_TO_R])
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
                stitched[y,x,k] = greyscale

# write stitched image
cv2.imwrite( 'images/series4/output_img/stitch4_3points.jpg', stitched )
