import cv2
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift

img_name = 'grey_trees'
img = cv2.imread('images/'+img_name+'.jpg', cv2.IMREAD_GRAYSCALE)

def get_blur_filter( size=15 ):
    motion_blur_filter = np.zeros((size, size))
    for y in range(size):
        #motion_blur_filter[ int((size-1)/2), y ] = 1 # left to right
        #motion_blur_filter[ y, int((size-1)/2) ] = 1 # top to down
        motion_blur_filter[ y, size-y-1 ] = 1 # top right to bottom left
    motion_blur_filter = motion_blur_filter / size
    return motion_blur_filter

def get_padded_filter( img, filter ):
    full_pad_y = img.shape[0] - filter.shape[0]
    full_pad_x = img.shape[1] - filter.shape[1]
    low_pad_y = full_pad_y // 2
    low_pad_x = full_pad_x // 2
    high_pad_y = low_pad_y + ( full_pad_y % 2 )
    high_pad_x = low_pad_x + ( full_pad_x % 2 )
    padded_filter = np.pad(filter, ((low_pad_y,high_pad_y), (low_pad_x,high_pad_x)), 'constant', constant_values=(0,0))
    return padded_filter

motion_filter = get_blur_filter()

padded_motion_filter = get_padded_filter( img, motion_filter )
temp_padded_filter = (padded_motion_filter-np.amin(padded_motion_filter))/(np.amax(padded_motion_filter)-np.amin(padded_motion_filter))*255
cv2.imwrite('images/output_img/'+img_name+'/padded_motion_filter.png', temp_padded_filter)

F = fftn(img)
H = fftn(padded_motion_filter)

F_shift = fftshift(np.log(np.abs(F)+1))
H_shift = fftshift(np.log(np.abs(H)+1))
temp_F_shift = (F_shift-np.amin(F_shift))/(np.amax(F_shift)-np.amin(F_shift))*255
temp_H_shift = (H_shift-np.amin(H_shift))/(np.amax(H_shift)-np.amin(H_shift))*255
cv2.imwrite('images/output_img/'+img_name+'/F.png', temp_F_shift)
cv2.imwrite('images/output_img/'+img_name+'/H.png', temp_H_shift)

G_from_freq = np.multiply(F,H)
g_from_freq = fftshift(ifftn(G_from_freq).real)
G_from_freq_shift = fftshift(np.log(np.abs(G_from_freq)+1))
temp_shift = (G_from_freq_shift-np.amin(G_from_freq_shift))/(np.amax(G_from_freq_shift)-np.amin(G_from_freq_shift))*255
cv2.imwrite('images/output_img/'+img_name+'/G_from_freq.png', g_from_freq)
cv2.imwrite('images/output_img/'+img_name+'/G_from_freq_at_freq.png', temp_shift)

g_from_spatial = cv2.filter2D(img, -1, motion_filter)
cv2.imwrite('images/output_img/'+img_name+'/G_from_spatial.png', g_from_spatial)

F_hat_from_freq = np.divide(G_from_freq,H)
f_hat_from_freq = ifftn(F_hat_from_freq).real
F_hat_from_freq_shift = fftshift(np.log(np.abs(F_hat_from_freq)+1))
temp_shift = (F_hat_from_freq_shift-np.amin(F_hat_from_freq_shift))/(np.amax(F_hat_from_freq_shift)-np.amin(F_hat_from_freq_shift))*255
cv2.imwrite('images/output_img/'+img_name+'/F_hat_from_freq_at_freq.png', temp_shift)
cv2.imwrite('images/output_img/'+img_name+'/F_hat_from_freq.png', f_hat_from_freq)

G_from_spatial = fftn(g_from_spatial)
G_from_spatial_shift = fftshift(np.log(np.abs(G_from_spatial)+1))
temp_shift = (G_from_freq_shift-np.amin(G_from_spatial_shift))/(np.amax(G_from_spatial_shift)-np.amin(G_from_spatial_shift))*255
cv2.imwrite('images/output_img/'+img_name+'/G_from_spatial_at_freq.png', temp_shift)

F_hat_from_spatial = np.divide(G_from_spatial,H)
f_hat_from_spatial = ifftn(F_hat_from_spatial).real
F_hat_from_spatial_shift = fftshift(np.log(np.abs(F_hat_from_spatial)+1))
temp_shift = (F_hat_from_spatial_shift-np.amin(F_hat_from_spatial_shift))/(np.amax(F_hat_from_spatial_shift)-np.amin(F_hat_from_spatial_shift))*255
cv2.imwrite('images/output_img/'+img_name+'/F_hat_from_freq_at_spatial.png', temp_shift)
cv2.imwrite('images/output_img/'+img_name+'/F_hat_from_spatial.png', f_hat_from_spatial)
