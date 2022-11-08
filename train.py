from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader, Dataset

import wandb
from wandb import AlertLevel

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import MAICON_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# config = {
#     "model_name" : "Unet",
#     "model_config" : {
#         "encoder_name" : "resnet34",
#         "encoder_weights" : "imagenet",
#         "siam_encoder" : True,
#         "fusion_form" : "concat",
#     },
#     "dataset_name" : "LEVIR_CD",
#     "lr" : 0.0001,
#     "gamma" : 0.1,
#     "epochs" : 60,
# }

config = {
    "model_name" : "UNet",
    "model_config" : {
        "encoder_name" : "resnet34",
        "encoder_weights" : "imagenet",
        "siam_encoder" : True,
        "fusion_form" : "concat",
    },
    "dataset_name" : "MAICON",
    "dataset_config" : {
        "train_batch_size" : 64,
        "test_batch_size" : 16
    },
    "lr" : 0.0001,
    "gamma" : 0.1,
    "epochs" : 60,
}

run_name = f'{config["dataset_name"]}_{config["model_name"]}_{datetime.now().strftime("%m/%d-%H:%M:%S")}'

wandb.login()
wandb.init(
    project="maicon-change-detection",
    name=run_name,
    config=config
)

model = cdp.Unet(
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your datasets)
    **config["model_config"]
)

# model = cdp.DeepLabV3(
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=2,  # model output channels (number of classes in your datasets)
#     **config["model_config"]
# )


train_dataset = MAICON_Dataset('/etc/maicon/data/maicon/train',
                                 sub_dir_1='input1',
                                 sub_dir_2='input2',
                                 img_suffix='.png',
                                 ann_dir='/etc/maicon/data/aihub/train/mask',
                                 debug=False)

valid_dataset = MAICON_Dataset('/etc/maicon/data/maicon/val',
                                 sub_dir_1='input1',
                                 sub_dir_2='input2',
                                 img_suffix='.png',
                                 debug=False,
                                 test_mode=True)

train_loader = DataLoader(train_dataset, batch_size=config["dataset_config"]["train_batch_size"], shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=config["dataset_config"]["test_batch_size"], shuffle=False, num_workers=0)

loss = cdp.utils.losses.CrossEntropyLoss()
metrics = [
    cdp.utils.metrics.Fscore(activation='argmax2d'),
    cdp.utils.metrics.Precision(activation='argmax2d'),
    cdp.utils.metrics.Recall(activation='argmax2d'),
    cdp.utils.metrics.IoU(activation='argmax2d'),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=config["lr"]),
])

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=config["gamma"])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = cdp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    wandb=wandb,
    verbose=True,
)

valid_epoch = cdp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    wandb=wandb,
    verbose=True,
)

# train model for 60 epochs

max_score = 0
MAX_EPOCH = config["epochs"]

for i in range(MAX_EPOCH):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    # !
    # valid_logs = valid_epoch.run(valid_loader)
    scheduler_steplr.step()

    # do something (save model, change lr, etc.)
    # !
    if max_score < train_logs['iou']:
        max_score = train_logs['iou']
        print('max_score', max_score)
        torch.save(model, f'./checkpoints/{run_name}.pth')
        print('Model saved!')

# save results (change maps)
"""
Note: if you use sliding window inference, set: 
    from change_detection_pytorch.datasets.transforms.albu import (
        ChunkImage, ToTensorTest)
    
    test_transform = A.Compose([
        A.Normalize(),
        ChunkImage({window_size}}),
        ToTensorTest(),
    ], additional_targets={'image_2': 'image'})

"""
# !
infer_dir = f'./infer_res/{run_name}'
os.makedirs(infer_dir)
    
valid_epoch.infer_vis(valid_loader, save=True, slide=False, save_dir=infer_dir)

wandb.alert(
    title = "Train finished",
    text = run_name,
    level = AlertLevel.INFO
)

wandb.finish()
