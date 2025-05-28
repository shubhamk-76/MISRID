import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path, greyscale=False):
    mode = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, mode)

    if image is None:
        raise ValueError('Invalid image path {}'.format(image_path))

    return cv2.resize(image,(960,999))


def get_inverse_of_green_channel(image):
    return cv2.bitwise_not(image[:, :, 1])

def save_image(image, image_path):
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    return cv2.imwrite(image_path, image)

