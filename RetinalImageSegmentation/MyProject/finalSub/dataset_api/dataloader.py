import logging
import numpy as np
import sys
sys.path.append('..')
import Preprocesser_and_Trainer.operations as image_utils

class DataLoader: 
    def __init__(self, image_number, database):
        if isinstance(image_number, int):
            image_number = '{:02d}'.format(image_number)

        logging.debug('Reading image, mask, truth %s from database', image_number)

        self.image = image_utils.read_image('{}/image/{}.jpg'.format(database, image_number))
        self.truth = image_utils.read_image('{}/truth/{}.png'.format(database, image_number),
                                            greyscale=True).astype(bool)
        self.fov_mask = image_utils.read_image('{}/mask/{}.jpg'.format(database, image_number),
                                               greyscale=True).astype(bool)
