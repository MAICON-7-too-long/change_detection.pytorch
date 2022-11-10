from datetime import datetime
import os
import json
import click

import torch
from torch.utils.data import DataLoader, Dataset

import wandb
from wandb import AlertLevel

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import MAICON_Dataset, LEVIR_CD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.argument("config_name")
def main(config_name):
    # load config
    with open(config_name) as config_file:
        config = json.load(config_file)

    # wandb init
    run_name = f'{config["dataset_name"]}_{config["model_name"]}_{datetime.now().strftime("%m_%d-%H_%M_%S")}'

    wandb.login()
    run = wandb.init(
        project="maicon-change-detection",
        name=run_name,
        config=config
    )

    # Model configure
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
        train_dataset = MAICON_Dataset('/workspace/data/01_data/train',
                                    sub_dir_1='input1',
                                    sub_dir_2='input2',
                                    img_suffix='.png',
                                    ann_dir='/workspace/data/01_data/train/mask',
                                    debug=False)

        valid_dataset = MAICON_Dataset('/workspace/data/01_data/test',
                                        sub_dir_1='input1',
                                        sub_dir_2='input2',
                                        img_suffix='.png',
                                        debug=False,
                                        test_mode=True)
                                        
    elif config["dataset_name"] == "LEVIR_CD":
        train_dataset = LEVIR_CD_Dataset('/etc/maicon/data/LEVIR-CD/train',
                                        sub_dir_1='A',
                                        sub_dir_2='B',
                                        img_suffix='.png',
                                        ann_dir='/etc/maicon/data/LEVIR-CD/train/label',
                                        debug=False)

        valid_dataset = LEVIR_CD_Dataset('/etc/maicon/data/LEVIR-CD/test',
                                        sub_dir_1='A',
                                        sub_dir_2='B',
                                        img_suffix='.png',
                                        ann_dir='/etc/maicon/data/LEVIR-CD/test/label',
                                        debug=False,
                                        test_mode=True)
    else:
        raise Exception("Wrong dataset type")

    # Split train dataset to train/test
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # DataLoader config
    train_loader = DataLoader(train_dataset, batch_size=config["dataset_config"]["train_batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["dataset_config"]["train_batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config["dataset_config"]["test_batch_size"], shuffle=False, num_workers=4)

    # Loss config
    if config["train_config"]["loss"] == "CrossEntropyLoss":
        loss = cdp.utils.losses.CrossEntropyLoss()
    if config["train_config"]["loss"] == "DiceLoss":
        loss = cdp.losses.DiceLoss(mode=cdp.losses.MULTICLASS_MODE, from_logits = True)
    else:
        raise Exception("Wrong loss type")

    # Metrics config
    metrics = [
        # cdp.utils.metrics.IoU(activation='softmax2d'),
        cdp.utils.my_metrics.Iou(class_num=0),
        cdp.utils.my_metrics.Iou(class_num=1),
        cdp.utils.my_metrics.Iou(class_num=2),
        cdp.utils.my_metrics.Iou(class_num=3),
    ]

    # Optimizer config
    if config["train_config"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=config["train_config"]["lr"]),
        ])
    elif config["train_config"]["optimizer"] == "NAdam":
        optimizer = torch.optim.NAdam([
            dict(params=model.parameters(), lr=config["lr"]),
        ])
    else:
        raise Exception("Wrong optimizer")

    # Default scheduler
    scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=config["train_config"]["gamma"])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
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
    early_stopping = cdp.utils.early_stopper.EarlyStopping(patience = 7, path = f'./checkpoints/{run_name}.pth', verbose = True)

    # train model for epochs
    max_score = 0
    MAX_EPOCH = config["train_config"]["epochs"]

    for i in range(MAX_EPOCH):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        # ! we don't have ground truth in MAICON
        test_logs = valid_epoch.run(test_loader)
        # valid_logs = valid_epoch.run(valid_loader)
        scheduler_steplr.step()

        # do something (save model, change lr, etc.)
        # if max_score < test_logs['valid_mIoU']:
        #     max_score = test_logs['valid_mIoU']
        #     print('max_score', max_score)
        #     torch.save(model, f'./checkpoints/{run_name}.pth')
        #     print('Model saved!')
            
        early_stopping(test_logs['valid_mIoU'], model)
        
        if early_stopping.early_stop:
            print('\nEarly Stopping')
            wandb.alert(
                title = "Early Stopping",
                text = run_name,
                level = AlertLevel.INFO
            )
            break

    # Inference for test images
    infer_dir = f'./infer_res/{run_name}'
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    best_model = torch.load(f'./checkpoints/{run_name}.pth')
    valid_epoch.predict(best_model, valid_loader, save_dir=infer_dir)
    # valid_epoch.infer_vis(valid_loader, save=True, slide=False, save_dir=infer_dir)

    # Save model as wandb artifact
    model_artifact = wandb.Artifact(name=run_name, type='model')
    model_artifact.add_file(local_path=f'./checkpoints/{run_name}.pth', name='model_weights')
    run.log_artifact(model_artifact)

    # Send wandb alert
    wandb.alert(
        title = "Train finished",
        text = run_name,
        level = AlertLevel.INFO
    )

    wandb.finish()

if __name__ == "__main__":
    main()
