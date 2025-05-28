from math import floor
import numpy as np
import cv2
def pad_matrix(mat, pad, value):

    def pad_integers(mat, pad, value):
        return cv2.copyMakeBorder(mat, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)

    if isinstance(value, bool):
        mat = mat.astype(np.uint8)
        padded = pad_integers(mat.astype(np.uint8), pad, 0)
        return padded.astype(bool)

    return pad_integers(mat, pad, value)
def convolve(image, k_size, function, msk, features): 
    import cv2
    pad = floor(k_size / 2)
    (image_height, image_width) = image.shape
    result = np.full((image_height, image_width, features), None, dtype=float)
    im_pded = pad_matrix(image, pad, 0)
    msk_pded = pad_matrix(msk, pad, False) 

    def get_neighborhood(image, x_index, y_index, pad):
        return image[y_index - pad: y_index + 1 + pad, x_index - pad: x_index + 1 + pad]

    for y_index in np.arange(pad, image_height + pad):
        for x_index in np.arange(pad, image_width + pad):
            nbhood = get_neighborhood(im_pded, x_index, y_index, pad)
            msk_nbhood = get_neighborhood(msk_pded, x_index, y_index, pad)
            result[y_index - pad, x_index - pad] = function(nbhood, msk_nbhood)

    return result
