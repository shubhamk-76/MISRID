import logging
from warnings import warn
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from matplotlib import pyplot
import logging
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
@log_execution
def train(input_image, truth_images, probability):
    input_image = np.array(input_image) 
    f_image = input_image.reshape(-1, input_image.shape[-1])
    f_truth = np.ravel(truth_images)
    base_estimator = SVC(gamma='auto', probability=probability)
    num_estimators = input_image.shape[0]
    num_samples = np.prod(input_image.shape[1:3])
    model = BaggingClassifier(base_estimator, n_estimators=num_estimators, max_samples=num_samples)
    model.fit(f_image, f_truth) 
    pickle.dump(model, open('models/model10.p', 'wb'))

@log_execution
def classify(feature_image):
    model = pickle.load(open('models/model.p', 'rb'))
    shape = feature_image.shape[:2]
    f_image = feature_image.reshape(-1, feature_image.shape[-1])
    probabilities = model.predict_proba(f_image) 
    probabilities = probabilities[:, 1].reshape(shape) 
    prediction = np.where(probabilities >= 0.5, True, False) 

    return probabilities, prediction

@log_execution
def assess(gt, prediction):
    true_positive = np.count_nonzero(np.logical_and(gt, prediction))
    true_negative = np.count_nonzero(np.logical_and(~gt, ~prediction))
    false_positive = np.count_nonzero(np.logical_and(~gt, prediction))
    false_negative = np.count_nonzero(np.logical_and(gt, ~prediction))
    precision = true_positive / np.count_nonzero(prediction)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive +
                                                  false_negative)
    logging.info('The Models Precision: %f', precision)
    logging.info('The Models Recall/Sensitivity: %f', sensitivity)
    logging.info('The Models Specificity: %f', specificity)
    logging.info('The Models Accuracy: %f', accuracy)

