import click
import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

import cv2

from change_detection_pytorch.datasets import MAICON_Dataset


@click.command()
@click.argument("run_name")
def main(run_name):
    best_model = torch.load(f'./checkpoints/{run_name}.pth')
    
    valid_dataset = MAICON_Dataset('/workspace/data/01_data/test',
                                        sub_dir_1='input1',
                                        sub_dir_2='input2',
                                        img_suffix='.png',
                                        debug=False,
                                        test_mode=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    for (x1, x2, filename) in tqdm(valid_loader):
        for (fname, xx1, xx2) in zip(filename, x1, x2):
            y_pred = best_model.predict(xx1, xx2)
            y_pred = y_pred.argmax(1).squeeze().cpu().numpy().round()
            
            cv2.imwrite(os.path.join("infer_res", fname), y_pred)

if __name__ == "__main__":
    main()