import cv2
import numpy as np
import math

X = 0; Y = 1
P1 = 0; P2 = 1; P3 = 2 # those three points we choose
IMG_L = 0; IMG_R = 1 # left image or right image at the overlap between two images
STITCH_12 = 0; STITCH_23 = 1 # STITCH_12 means stitching image 1 & 2

def Set_3_mark_points():
    img12_point1_x = 2413; img12_point1_y = 2268
    img12_point2_x = 3082; img12_point2_y = 2117
    img12_point3_x = 2980; img12_point3_y = 3304

    img21_point1_x = 2007; img21_point1_y = 2262
    img21_point2_x = 2626; img21_point2_y = 2110
    img21_point3_x = 2564; img21_point3_y = 3249

    img23_point1_x = 2007; img23_point1_y = 2262 # 還沒設
    img23_point2_x = 2626; img23_point2_y = 2110 # 還沒設
    img23_point3_x = 2564; img23_point3_y = 3249 # 還沒設

    img32_point1_x = 2413; img32_point1_y = 2268 # 還沒設
    img32_point2_x = 3082; img32_point2_y = 2117 # 還沒設
    img32_point3_x = 2980; img32_point3_y = 3304 # 還沒設

    img12_xy = [ [img12_point1_x, img12_point1_y], [img12_point2_x, img12_point2_y], [img12_point3_x, img12_point3_y] ]
    img21_xy = [ [img21_point1_x, img21_point1_y], [img21_point2_x, img21_point2_y], [img21_point3_x, img21_point3_y] ]
    img23_xy = [ [img23_point1_x, img23_point1_y], [img23_point2_x, img23_point2_y], [img23_point3_x, img23_point3_y] ]
    img32_xy = [ [img32_point1_x, img32_point1_y], [img32_point2_x, img32_point2_y], [img32_point3_x, img32_point3_y] ]

    return [[img12_xy, img21_xy], [img23_xy, img32_xy]] # stitch edge -> edge left/right -> point 1/2/3 -> x/y

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

# read image
img1 = cv2.imread( 'images/first_two/DSC_2231.jpg' ) # left
img2 = cv2.imread( 'images/first_two/DSC_2230.jpg' ) # right

# set three point
xy_mapping = Set_3_mark_points()

# get trans matrix from 1 to 2 and from 2 to 1
affine_12_LtoR_coef = Get_affine_coef(xy_mapping[STITCH_12][IMG_L], xy_mapping[STITCH_12][IMG_R])
affine_12_RtoL_coef = Get_affine_coef(xy_mapping[STITCH_12][IMG_R], xy_mapping[STITCH_12][IMG_L])

# get stitch1 size from 2 to 1
img1_height, img1_width = img1.shape[:2]
img2_height, img2_width = img2.shape[:2]

border_x = [0, img1_width]
border_y = [0, img1_height]

for i in [0, img2_width]:
    for j in [0, img2_height]:
        temp_x , temp_y = Affine(i, j, affine_12_RtoL_coef)
        border_x.append(temp_x)
        border_y.append(temp_y)
        #print(temp_x, temp_y)

stitch1_top = math.floor(min(border_y))
stitch1_bottom = math.ceil(max(border_y))
stitch1_left = math.floor(min(border_x))
stitch1_right = math.ceil(max(border_x))

stitch1_height = stitch1_bottom - stitch1_top + 1
stitch1_width = stitch1_right - stitch1_left + 1

# set stitch1 zeros
stitch1 = np.zeros( [ stitch1_height, stitch1_width, 3 ])

# put img1 and img2 on stitch1
for y in range(0, stitch1_height):
    print(y)
    for x in range(0, stitch1_width):
        if (0 <= (x + stitch1_left) <= (img1_width - 1)) and (0 <= (y + stitch1_top) <= (img1_height - 1)):
            stitch1[y,x,:] = img1[y+stitch1_top, x+stitch1_left, :]
        else:
            x_in_img2, y_in_img2 = Affine(x+stitch1_left, y+stitch1_top, affine_12_LtoR_coef)
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
cv2.imwrite( 'images/first_two/output_img/test.jpg', stitch1 )
