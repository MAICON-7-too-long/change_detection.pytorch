"""
Predict
"""
import ssl
import numpy as np
import os, sys, cv2, warnings
from glob import glob

ssl._create_default_https_context = ssl._create_unverified_context
prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
warnings.filterwarnings('ignore')


def read_images(img_path):
    files = glob(os.path.join(img_path, 'mask', '*.png'))
    prob = dict()
    for file in files:
        y = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        prob[os.path.split(file)[1]] = y.astype(np.short)
    return prob


if __name__ == '__main__':
    pred_dirs = 'results/preds/'
    prob = []
    for pred_dir in os.listdir(pred_dirs):
        prob.append(read_images(os.path.join(pred_dirs, pred_dir)))
    pred_result_dir_mask = os.path.join(prj_dir, 'results', 'pred', 'final_output')
    os.makedirs(pred_result_dir_mask, exist_ok=True)
    for i in prob[0].keys():
        p = np.zeros((prob[0][i].shape), dtype=np.uint8)
        for x0 in range(prob[0][i].shape[0]):
            for x1 in range(prob[0][i].shape[1]):
                data = [prob[0][i][x0, x1], prob[1][i][x0, x1], prob[2][i][x0, x1],
                        prob[3][i][x0, x1], prob[4][i][x0, x1], prob[5][i][x0, x1]]
                p[x0, x1] = max(data, key=data.count)
        cv2.imwrite(os.path.join(pred_result_dir_mask, i), p)
