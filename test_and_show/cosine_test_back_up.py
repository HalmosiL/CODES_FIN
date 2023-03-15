import torch
import json
import sys
import numpy as np
import cv2
import os

sys.path.append('../')

from modules.model import load_model, get_model_dummy
from attacks.cosine import model_immer_attack_auto_loss, Cosine_PDG_Adam
from dataset.dataset import SemDataSplit
import dataset.transform as transform

CONFIG_PATH_MAIN = sys.argv[1]
CONFIG_MAIN = json.load(open(CONFIG_PATH_MAIN ))

try:
    print(CONFIG_MAIN['SAVE_FOLDER'] + "cosine/")
    os.mkdir(CONFIG_MAIN['SAVE_FOLDER'] + "cosine/")
except OSError as error:
    print("Folder is alredy exist...")

if(CONFIG_MAIN["MODE"] == "DUMMY"):
    model = get_model_dummy(CONFIG_MAIN["DEVICE"]).eval()
    model_slice = model.getSliceModel().eval()
elif(CONFIG_MAIN["MODE"] == "NORMAL"):
    model = load_model(
        CONFIG_MAIN["MODEL_PATH"], 
        CONFIG_MAIN["DEVICE"]
    ).eval()
    model_slice = model.getSliceModel().eval()

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

val_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)]
)

val_loader = torch.utils.data.DataLoader(   
    dataset=SemDataSplit(
        split='val',
        data_root=CONFIG_MAIN['DATA_PATH'],
        data_list=CONFIG_MAIN['IMAGE_LIST'],
        transform=val_transform
    ),
    batch_size=1,
    num_workers=CONFIG_MAIN['NUMBER_OF_WORKERS'],
    pin_memory=CONFIG_MAIN['PIN_MEMORY']
)

attack = Cosine_PDG_Adam(
    step_size=0.005,
    clip_size=CONFIG_MAIN['EPS']
)

for e, (images, labels, label) in enumerate(val_loader):
    predictions = []

    if(-1 < e):
        for i in range(len(images)):
            image = images[i].to(CONFIG_MAIN["DEVICE"])
            target = labels[i].to(CONFIG_MAIN["DEVICE"])

            image = model_immer_attack_auto_loss(
                image,
                model_slice,
                attack,
                CONFIG_MAIN["NUMBER_OF_ITERS"],
                CONFIG_MAIN["DEVICE"]
            )

            if(CONFIG_MAIN["MODE"] == "DUMMY"):
                _, pred = model(image)
                pred = pred.max(1)[1]
            elif(CONFIG_MAIN["MODE"] == "NORMAL"):
                pred, _ = model(image)

            predictions.append(pred)

        pred_sum_mask = torch.zeros(898, 1796)

        i = 0

        for x in range(2):
            for y in range(4):
                pred_sum_mask[x*449:(x+1)*449, y*449:(y+1)*449] = predictions[i]
                i += 1

        torch.save(pred_sum_mask, CONFIG_MAIN['SAVE_FOLDER'] + "prediction_" + str(e) + ".pth")
        torch.save(label[0], CONFIG_MAIN['SAVE_FOLDER'] + "label_" + str(e) + ".pth")
