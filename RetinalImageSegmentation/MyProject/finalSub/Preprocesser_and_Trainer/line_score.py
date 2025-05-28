from math import floor
import numpy as np

def line_score(nbhd, fov_mask, mask_list):
    center_of_image = floor(fov_mask.shape[0] / 2)
    if not fov_mask[center_of_image][center_of_image]:
        return np.array([0.0, 0.0]) # Center pixel outside of mask

    scores = list()
    nbhd_avg = np.mean(nbhd[fov_mask])
    nbhd[~fov_mask] = nbhd_avg

    def score_array(line_avg, orth_avg):
        return np.array([
            max(line_avg - nbhd_avg, 0.0),
            max(orth_avg - nbhd_avg, 0.0)
        ])

    for line_mask in mask_list:
        line_avg = np.mean(nbhd[line_mask.mask])
        orth_avg = np.mean(nbhd[line_mask.orthogonal_mask])
        scores.append(score_array(line_avg, orth_avg))

    return max(scores, key=lambda x: x[0])
