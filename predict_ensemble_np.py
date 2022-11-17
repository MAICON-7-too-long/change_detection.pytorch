"""
Predict
"""
import ssl
import numpy as np
import os, sys, cv2, warnings
from glob import glob
from tqdm import tqdm
import click

ssl._create_default_https_context = ssl._create_unverified_context
prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
warnings.filterwarnings('ignore')

@click.command()
@click.argument("pred_dirs")
@click.argument("result_dirs")
def main(pred_dirs, result_dirs):
    os.makedirs(result_dirs, exist_ok=True)
    files = glob(os.path.join('infer_res/submitted_mask/model1_split', '*.png'))
    for file in tqdm(files):
        filename = os.path.split(file)[1]

        masks = [ cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.short) ]
        masks = np.append(masks, [ cv2.imread(os.path.join('infer_res/submitted_mask/model2_split', os.path.basename(file)), cv2.IMREAD_GRAYSCALE).astype(np.short) ], axis = 0)
        masks = np.append(masks, [ cv2.imread(os.path.join('infer_res/submitted_mask/model3_split', os.path.basename(file)), cv2.IMREAD_GRAYSCALE).astype(np.short) ], axis = 0)
        masks = np.append(masks, [ cv2.imread(os.path.join('infer_res/submitted_mask/model4_split', os.path.basename(file)), cv2.IMREAD_GRAYSCALE).astype(np.short) ], axis = 0)
        masks = np.append(masks, [ cv2.imread(os.path.join('infer_res/submitted_mask/model5_split', os.path.basename(file)), cv2.IMREAD_GRAYSCALE).astype(np.short) ], axis = 0)
        masks = np.append(masks, [ cv2.imread(os.path.join('infer_res/submitted_mask/model6_split', os.path.basename(file)), cv2.IMREAD_GRAYSCALE).astype(np.short) ], axis = 0)

        p = np.apply_along_axis(lambda x: np.bincount(x, weights=[1.001, 1.0001, 1.00001, 1.01, 1.000001, 1.1]).argmax(), axis=0, arr=masks)

        cv2.imwrite(os.path.join(result_dirs, filename), p)

if __name__ == "__main__":
    main()