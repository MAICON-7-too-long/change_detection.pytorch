import click
import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

import cv2

from change_detection_pytorch.datasets import MAICON_Dataset
import change_detection_pytorch as cdp

import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.argument("run_name")
def main(run_name):
    wandb.login()
    wandb.init(
        project="maicon-change-detection-submissions",
        name=run_name,
        config={
            "name" : run_name
        }
    )

    model = torch.load(f'./checkpoints/{run_name}.pth')
    model.eval()
    
    valid_dataset = MAICON_Dataset('/etc/maicon/data/maicon/test',
                                        sub_dir_1='input1',
                                        sub_dir_2='input2',
                                        img_suffix='.png',
                                        debug=False,
                                        test_mode=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

    valid_epoch = cdp.utils.train.ValidEpoch(
        model,
        loss = cdp.losses.DiceLoss(mode=cdp.losses.MULTICLASS_MODE, from_logits = True),
        metrics = [
            # cdp.utils.metrics.IoU(activation='softmax2d'),
            cdp.utils.my_metrics.Iou(class_num=0),
            cdp.utils.my_metrics.Iou(class_num=1),
            cdp.utils.my_metrics.Iou(class_num=2),
            cdp.utils.my_metrics.Iou(class_num=3),
        ],
        device=DEVICE,
        classes=4,
        wandb=wandb,
        verbose=True,
    )

    infer_dir = f'./infer_res/{run_name}'
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    valid_epoch.predict(model, valid_loader, save_dir=infer_dir)
    
    # for (x1, x2, filename) in tqdm(valid_loader):
    #     for (fname, xx1, xx2) in zip(filename, x1, x2):
    #         y_pred = model.predict(xx1, xx2)
    #         y_pred = y_pred.argmax(1).squeeze().cpu().numpy().round()
            
    #         cv2.imwrite(os.path.join("infer_res", fname), y_pred)

    wandb.finish()

if __name__ == "__main__":
    main()