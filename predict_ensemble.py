"""
Predict
"""
import ssl
import numpy as np
import os, sys, cv2, warnings
from glob import glob

import click

ssl._create_default_https_context = ssl._create_unverified_context
prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
warnings.filterwarnings('ignore')


def read_images(img_path):
    files = glob(os.path.join(img_path, '*.png'))
    prob = dict()
    for file in files:
        y = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        prob[os.path.split(file)[1]] = y.astype(np.short)
    return prob


@click.command()
@click.argument("pred_dirs")
@click.argument("result_dirs")
def main(pred_dirs, result_dirs):
    prob = []
    for pred_dir in os.listdir(pred_dirs):
        prob.append(read_images(os.path.join(pred_dirs, pred_dir)))
    os.makedirs(result_dirs, exist_ok=True)
    for i in prob[0].keys():
        p = np.zeros((prob[0][i].shape), dtype=np.uint8)
        for x0 in range(prob[0][i].shape[0]):
            for x1 in range(prob[0][i].shape[1]):
                data = [prob[0][i][x0, x1], prob[1][i][x0, x1], prob[2][i][x0, x1],
                        prob[3][i][x0, x1], prob[4][i][x0, x1], prob[5][i][x0, x1]]
                p[x0, x1] = max(data, key=data.count)
        cv2.imwrite(os.path.join(result_dirs, i), p)

if __name__ == "__main__":
    main()