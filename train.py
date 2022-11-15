from datetime import datetime
import os
import json
import click

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset

import wandb
from wandb import AlertLevel

import cv2
import os.path as osp

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import MAICON_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.argument("config_name")
@click.option('-o', '--output-file', 'output_file', required=False)
@click.option('-l', '--load-model', 'load_model', required=False)
def main(config_name, output_file, load_model):
    # load config
    with open(config_name) as config_file:
        config = json.load(config_file)

    # remove randomness
    torch.cuda.manual_seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # wandb init
    if config["id"] > 0:
        run_name = f'{config["dataset_name"]}_{config["model_name"]}_{config["id"]}'
    else:
        run_name = f'{config["dataset_name"]}_{config["model_name"]}_{datetime.now().strftime("%m_%d-%H_%M_%S")}'

    wandb.login()
    run = wandb.init(
        project="maicon-change-detection",
        name=run_name,
        config=config
    )

    # Model configure
    if load_model:
        model = torch.load(f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/checkpoints/{load_model}.pth')
    if config["model_name"] == "Unet":
        model = cdp.Unet(
            **config["model_config"]
        )
    elif config["model_name"] == "UnetPlusPlus":
        model = cdp.UnetPlusPlus(
            **config["model_config"]
        )
    elif config["model_name"] == "DeepLabV3":
        model = cdp.DeepLabV3(
            **config["model_config"]
        )
    elif config["model_name"] == "DeepLabV3Plus":
        model = cdp.DeepLabV3Plus(
            **config["model_config"]
        )
    elif config["model_name"] == "PSPNet":
        model = cdp.PSPNet(
            **config["model_config"]
        )
    elif config["model_name"] == "STANet":
        model = cdp.STANet(
            **config["model_config"]
        )
    else:
        raise Exception("Wrong model type")
    
    # Dataset configure
    if config["dataset_name"] == "MAICON":
        train_dataset = MAICON_Dataset(f'{os.environ.get("DATA_DIR", "/workspace/data/01_data")}/train',
                                    sub_dir_1='input1',
                                    sub_dir_2='input2',
                                    img_suffix='.png',
                                    ann_dir=f'{os.environ.get("DATA_DIR", "/workspace/data/01_data")}/train/mask',
                                    size=config["dataset_config"]["image_size"],
                                    augmentation=config["dataset_config"]["augmentation"],
                                    debug=False)

        test_dataset = MAICON_Dataset(f'{os.environ.get("DATA_DIR", "/workspace/data/01_data")}/test',
                                        sub_dir_1='input1',
                                        sub_dir_2='input2',
                                        img_suffix='.png',
                                        debug=False,
                                        test_mode=True)
    else:
        raise Exception("Wrong dataset type")

    # Split train dataset to train/test
    train_size = int(config["train_config"]["split_ratio"] * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    
    # DataLoader config
    train_loader = DataLoader(train_dataset, batch_size=config["train_config"]["train_batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config["train_config"]["train_batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["train_config"]["test_batch_size"], shuffle=False, num_workers=4)

    # Loss config
    if config["train_config"]["loss_name"] == "CrossEntropyLoss":
        loss = cdp.utils.losses.CrossEntropyLoss(**config["train_config"]["loss_config"])
    elif config["train_config"]["loss_name"] == "DiceLoss":
        loss = cdp.losses.DiceLoss(mode=cdp.losses.MULTICLASS_MODE, from_logits = True, **config["train_config"]["loss_config"])
    else:
        raise Exception("Wrong loss type")

    # Metrics config
    metrics = [
        cdp.utils.my_metrics.Iou(class_num=0),
        cdp.utils.my_metrics.Iou(class_num=1),
        cdp.utils.my_metrics.Iou(class_num=2),
        cdp.utils.my_metrics.Iou(class_num=3),
    ]

    # Optimizer config
    if config["train_config"]["optimizer_name"] == "Adam":
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), **config["train_config"]["loss_config"]),
        ])
    elif config["train_config"]["optimizer_name"] == "NAdam":
        optimizer = torch.optim.NAdam([
            dict(params=model.parameters(), **config["train_config"]["loss_config"]),
        ])
    elif config["train_config"]["optimizer_name"] == "AdamW":
        optimizer = torch.optim.AdamW([
            dict(params=model.parameters(), **config["train_config"]["loss_config"]),
        ])
    else:
        raise Exception("Wrong optimizer type")

    # Scheduler config
    if config["train_config"]["scheduler_name"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["train_config"]["scheduler_config"])
    elif config["train_config"]["scheduler_name"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["train_config"]["scheduler_config"])
    else:
        raise Exception("Wrong scheduler type")
    

    # Create epoch runners
    train_epoch = cdp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        classes=config["model_config"]["classes"],
        wandb=wandb,
        verbose=True,
    )

    valid_epoch = cdp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        classes=config["model_config"]["classes"],
        wandb=wandb,
        verbose=True,
    )
    
    # Early stopper
    early_stopping = cdp.utils.early_stopper.EarlyStopping(patience = config["train_config"]["earlystopping_patience"], path = f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/checkpoints/{run_name}_best.pth', verbose = True)
    
    # Inference for test images
    infer_dir = f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/infer_res/{run_name}'
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    if config["train_config"]["debug_predict"]:
        debug_dir = f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/debug_predict/{run_name}'
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    # train model for epochs
    MAX_EPOCH = config["train_config"]["epochs"]

    for i in range(MAX_EPOCH):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader, i, infer_dir)
        valid_logs = valid_epoch.run(valid_loader, i, infer_dir)
        
        # Generate mask for debug
        if config["train_config"]["debug_predict"]:
            for (x1, x2, filename) in test_loader:
                x1, x2 = valid_epoch.check_tensor(x1, False), valid_epoch.check_tensor(x2, False)
                x1, x2 = x1.float(), x2.float()
                x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
                y_pred = model.forward(x1, x2)
                
                y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
                y_pred = y_pred * 85

                for (fname, xx1, xx2, yy_pred) in zip(filename, x1, x2, y_pred):
                    fname = fname.split('.')[0] + f'_batch{i}.png'

                    cv2.imwrite(osp.join(debug_dir, fname), yy_pred)
                    
                break
        
        scheduler.step()

        if config["train_config"]["earlystopping_target"].find("IoU") == -1:
            early_stopping(valid_logs[config["train_config"]["earlystopping_target"]], model)
        else:
            early_stopping(1-valid_logs[config["train_config"]["earlystopping_target"]], model)
        
        
        if early_stopping.early_stop:
            print('\nEarly Stopping...')
            wandb.alert(
                title = "Early Stopping",
                text = run_name,
                level = AlertLevel.INFO
            )

            break

        torch.save(model, f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/checkpoints/{run_name}_epoch_{i}.pth')

    if output_file:
        torch.save(model, f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/checkpoints/{output_file}_last.pth')

    # Save model as wandb artifact
    model_artifact = wandb.Artifact(name=run_name, type='model')
    model_artifact.add_file(local_path=f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/checkpoints/{run_name}.pth', name='model_weights')
    run.log_artifact(model_artifact)

    # Generate mask image
    # best_model = torch.load(f'{os.environ.get("CDP_DIR", "/workspace/Final_Submission")}/checkpoints/{run_name}_best.pth')
    # valid_epoch.predict(best_model, valid_loader, save_dir=infer_dir)

    # Send finish alert to wandb
    wandb.alert(
        title = "Train finished",
        text = run_name,
        level = AlertLevel.INFO
    )

    wandb.finish()

if __name__ == "__main__":
    main()
