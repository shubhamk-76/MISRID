from datetime import datetime
import logging
import argparse as ap
from dataset_api.line_mask import generate_line_mask_list
from dataset_api.dataloader import DataLoader
import Preprocesser_and_Trainer.operations as operations
import Preprocesser_and_Trainer.svm as svm
import numpy as np
from Preprocesser_and_Trainer.convolve import convolve
from Preprocesser_and_Trainer.operations import get_inverse_of_green_channel
from Preprocesser_and_Trainer.line_score import line_score
from time import time
def log_execution(func):
    def wrapped(*args, **kwargs):
        logging.debug('Executing %s...', func.__name__)
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.debug('Completed %s (%.3fs)', func.__name__, end - start)
        return result
    return wrapped
def normalize_features(vectors):
    means = np.mean(vectors, axis=(0, 1))
    deviations = np.std(vectors, axis=(0, 1))

    return (vectors - means) / deviations

def calculate_features(image, fov_mask, mask_list, k_size):
    inverse_green = get_inverse_of_green_channel(image) 
    function = lambda x, y: line_score(x, y, mask_list) 
    result = convolve(inverse_green, k_size, function, fov_mask, 2)
    vectors = np.dstack((result, inverse_green)) 
    vectors = normalize_features(vectors)
    return vectors

@log_execution
def train_model(images, mask_list, k_size, probability):
    logging.info('Calculating and  normalizing feature vectors for %d image(s)', len(images))
    vectors_list = [calculate_features(x.image, x.fov_mask, mask_list, k_size) for x in images]
    truth_list = [x.truth for x in images]
    logging.info('Model training being done with %d image(s)', len(images))
    svm.train(vectors_list, truth_list, probability) 

@log_execution
def classify_image(images, mask_list, k_size, save,serial):
    logging.info('reading feature vectors for image')
    image = images[0]
    vectors = calculate_features(image.image, image.fov_mask, mask_list, k_size)
    logging.info('predicting image pixels label values for vessel detecion')
    probabilities, prediction = svm.classify(vectors)
    svm.assess(image.truth, prediction)
    if save:
        name="prediction"+str(serial)+".png"
        operations.save_image(prediction,name)

def main():
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--images', help='Image from database', nargs='+',
                        type=int, required=True)
    parser.add_argument('-k', '--kernel', help='Neighborhood size', type=int, default=15)
    parser.add_argument('-r', '--rotation', help='Rotational resolution', type=int, default=15)
    parser.add_argument('-t', '--train', help='Train SVM model', action='store_true')
    parser.add_argument('-p', '--probability', help='Probabilistic SVM model', action='store_true')
    parser.add_argument('-s', '--save', help='Save image', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG 
    logging.basicConfig(format='%(message)s', level=log_level)
    database = 'DRIVE'
    logging.info('get %d img from %s data', len(args.images), database)
    img_inputs = [DataLoader(x, database) for x in args.images]
    mask_list = generate_line_mask_list(args.kernel, args.rotation)

    if args.train:
        train_model(img_inputs, mask_list, args.kernel, args.proba)
    else:
        print(int(args.images[0])) 
        classify_image(img_inputs, mask_list, args.kernel, args.save,int(args.images[0]))

if __name__ == '__main__':
    main()
