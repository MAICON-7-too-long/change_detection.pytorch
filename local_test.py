import torch
from torch.utils.data import DataLoader, Dataset

import wandb

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    "dataset" : "LEVIR_CD",
    "encoder_name" : "resnet34",
    "encoder_weights" : "imagenet",
    "siam_encoder" : True,
    "fusion_form" : "concat",
    "lr" : 0.0001,
    "gamma" : 0.1,
    "epochs" : 60,
}

wandb.login()
wandb.init(
    project="maicon-change-detection",
    name="local_test_1",
    config=config
)

model = cdp.Unet(
    encoder_name=config["encoder_name"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=config["encoder_weights"],  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your datasets)
    siam_encoder=config["siam_encoder"],  # whether to use a siamese encoder
    fusion_form=config["fusion_form"],  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
)

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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

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
    valid_logs = valid_epoch.run(valid_loader)
    scheduler_steplr.step()

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        print('max_score', max_score)
        torch.save(model, './best_model.pth')
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
valid_epoch.infer_vis(valid_loader, save=True, slide=False, save_dir='./infer_res')

wandb.finish()
