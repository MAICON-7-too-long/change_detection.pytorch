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
        prob = []
        filename = os.path.split(file)[1]
        prob.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.short))
        prob.append(
            cv2.imread(os.path.join('infer_res/submitted_mask/model2_split', os.path.basename(file)),
                       cv2.IMREAD_GRAYSCALE).astype(np.short))
        prob.append(
            cv2.imread(os.path.join('infer_res/submitted_mask/model3_split', os.path.basename(file)),
                       cv2.IMREAD_GRAYSCALE).astype(np.short))
        prob.append(
            cv2.imread(os.path.join('infer_res/submitted_mask/model4_split', os.path.basename(file)),
                       cv2.IMREAD_GRAYSCALE).astype(np.short))
        prob.append(
            cv2.imread(os.path.join('infer_res/submitted_mask/model5_split', os.path.basename(file)),
                       cv2.IMREAD_GRAYSCALE).astype(np.short))
        prob.append(
            cv2.imread(os.path.join('infer_res/submitted_mask/model6_split', os.path.basename(file)),
                       cv2.IMREAD_GRAYSCALE).astype(np.short))

        p = np.zeros((prob[0].shape), dtype=np.uint8)
        for x0 in range(prob[0].shape[0]):
            for x1 in range(prob[0].shape[1]):
                data = [prob[5][x0, x1], prob[1][x0, x1], prob[2][x0, x1],
                        prob[3][x0, x1], prob[4][x0, x1], prob[0][x0, x1]]
                p[x0, x1] = max(data, key=data.count)
        cv2.imwrite(os.path.join(result_dirs, filename), p)

if __name__ == "__main__":
    main()