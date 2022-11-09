import click
import os, json, glob, shutil
import random

import torch
import numpy as np
from PIL import Image
import imghdr

from tqdm import tqdm

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
@click.argument("mask_file")
def vis_mask(mask_file):
    if not os.path.exists(os.path.join("./vis")):
        os.makedirs(os.path.join("./vis"))

    if os.path.isfile(mask_file):
        mask = Image.open(mask_file)
        mask = np.array(mask)
        mask = mask * 85
        mask_img = Image.fromarray(mask)

        mask_img.save(os.path.join("./vis", mask_file.split(".")[0]+"_vis.png"))
        print(os.path.join("./vis", mask_file.split(".")[0]+"_vis.png"))

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


cli.add_command(split_image)
cli.add_command(merge_mask)
cli.add_command(vis_mask)
cli.add_command(valid_images)

if __name__ == "__main__":
    cli()