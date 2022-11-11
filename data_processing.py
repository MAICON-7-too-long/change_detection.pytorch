import click
import os, json, glob, shutil
import random

import torch
import numpy as np
from PIL import Image
import imghdr

from tqdm import tqdm

import wandb

from joblib import Parallel, delayed


@click.group()
def cli():
    pass

@cli.command()
@click.argument("dir")
def split_image(dir):
    if not os.path.exists(os.path.join(dir, "../input1")):
        os.makedirs(os.path.join(dir, "../input1"))
    if not os.path.exists(os.path.join(dir, "../input2")):
        os.makedirs(os.path.join(dir, "../input2"))

    for file in os.listdir(dir):
        img_file = os.path.join(dir, file)
        if os.path.isfile(img_file):
            img = Image.open(img_file)
            w, h = image.size
            image = np.array(image)
            img1 = image[:,:h,]
            img2 = image[:,h:,]
            
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)

            img1.save(os.path.join(dir, "../input1", file))
            img2.save(os.path.join(dir, "../input2", file))

@cli.command()
@click.argument("dir")
@click.argument("mask_dir")
def merge_mask(dir, mask_dir):
    if not os.path.exists(os.path.join(dir, "../", mask_dir)):
        os.makedirs(os.path.join(dir, "../", mask_dir))

    def merge(file):
        mask_file = os.path.join(dir, file)
        if os.path.isfile(mask_file):
            mask = Image.open(mask_file)
            w, h = mask.size
            mask = np.array(mask)
            mask_l = mask[:,:h,]
            mask_r = mask[:,h:,]

            mask_r[mask_r == 1] = 4
            mask = np.maximum(mask_l, mask_r)
            mask[mask == 4] = 1
            
            mask_img = Image.fromarray(mask)

            mask_img.save(os.path.join(dir, "../", mask_dir, file))

    res = Parallel(n_jobs=10)(delayed(merge)(file) for file in tqdm(os.listdir(dir)))

@cli.command()
# @click.argument("mask_file")
@click.argument("dir")
def vis_mask(dir):
    if not os.path.exists(os.path.join("./vis")):
        os.makedirs(os.path.join("./vis"))

    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            mask = Image.open(os.path.join(dir, file))
            mask = np.array(mask)
            mask = mask * 85
            mask_img = Image.fromarray(mask)

            mask_img.save(os.path.join("./vis", file.split(".")[0]+"_vis.png"))
            print(os.path.join("./vis", file.split(".")[0]+"_vis.png"))

@cli.command()
@click.argument("dir")
@click.argument("format")
def valid_images(dir, format):
    def valid(file):
        img_file = os.path.join(dir, file)
        if imghdr.what(img_file) == format:
            return True
        else:
            return False

    res = Parallel(n_jobs=10)(delayed(valid)(file) for file in tqdm(os.listdir(dir)))
    
    if np.all(res):
        print("Valid")
    else:
        print("Not valid")

@cli.command()
@click.argument("dir")
@click.argument("mask_dir")
def split_mask(dir, mask_dir):
    if not os.path.exists(os.path.join(dir, "../", mask_dir)):
        os.makedirs(os.path.join(dir, "../", mask_dir))

    def merge(file):
        mask_file = os.path.join(dir, file)
        if os.path.isfile(mask_file):
            mask = Image.open(mask_file)
            w, h = mask.size
            mask = np.array(mask)
            mask_l = np.where(mask == 2, mask, 0)
            mask_r = np.where(mask % 2 == 1, mask, 0)

            mask = np.concatenate((mask_l, mask_r), axis = 1)
            
            mask_img = Image.fromarray(mask)

            mask_img.save(os.path.join(dir, "../", mask_dir, file))

    res = Parallel(n_jobs=10)(delayed(merge)(file) for file in tqdm(os.listdir(dir)))

@cli.command()
@click.argument("x_dir")
@click.argument("y_dir")
def vis_results(x_dir, y_dir):
    submission_name = y_dir.split("/")[-1]

    wandb.login()
    wandb.init(
        project="maicon-change-detection-submissions",
        name=submission_name,
        config={
            "name" : submission_name
        }
    )

    infer_table = wandb.Table(columns=['Name', 'Image'])

    def upload(file):
        img_path = os.path.join(x_dir, file)
        mask_path = os.path.join(y_dir, file)

        img = Image.open(img_path)
        img = np.array(img)
        # img = torch.tensor(img)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        # mask = torch.tensor(mask)
        
        wandb_img = wandb.Image(img, masks = {
            "prediction" : {
                "mask_data" : mask,
                "class_labels" : { 
                    1 : "New",
                    2 : "Destory",
                    3 : "Change"
                } 
            },
        })

        infer_table.add_data(file, wandb_img)

    # res = Parallel(n_jobs=10)(delayed(upload)(file) for file in tqdm(os.listdir(x_dir)))
    for file in tqdm(os.listdir(x_dir)):
        upload(file)

    wandb.log({"Table" : infer_table})

    wandb.finish()

@cli.command()
@click.argument("dir")
def input_size_check(dir):
    def check(file):
        img1_file = os.path.join(dir, "input1", file)
        img2_file = os.path.join(dir, "input2", file)
        
        img1 = Image.open(img1_file)
        img2 = Image.open(img2_file)
        
        if img1.size == img2.size:
            return True
        else:
            return False

    res = Parallel(n_jobs=10)(delayed(check)(file) for file in tqdm(os.listdir(os.path.join(dir, "input1"))))
    
    if np.all(res):
        print("Valid")
    else:
        print("Not valid")

@cli.command()
@click.argument("dir")
def print_size(dir):
    for file in os.listdir(os.path.join(dir)):
        img = Image.open(os.path.join(dir, file))
        w, h = img.size
        
        if w == 750:
            print(file)
        

cli.add_command(split_image)
cli.add_command(merge_mask)
cli.add_command(vis_mask)
cli.add_command(valid_images)
cli.add_command(split_mask)
cli.add_command(vis_results)
cli.add_command(input_size_check)
cli.add_command(print_size)

if __name__ == "__main__":
    cli()