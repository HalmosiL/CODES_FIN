import torch
import json
import sys
import numpy as np
import cv2
import os

sys.path.append('../')

from modules.model import load_model, get_model_dummy
from attacks.bim import BIM
from dataset.dataset import SemDataSplit
import dataset.transform as transform

CONFIG_PATH_MAIN = "../configs/config_main.json"
CONFIG_MAIN = json.load(open(CONFIG_PATH_MAIN))

try:
    print(CONFIG_MAIN['SAVE_FOLDER'] + "bim_full_sat/")
    os.mkdir(CONFIG_MAIN['SAVE_FOLDER'] + "bim_full_sat/")
except OSError as error:
    print(error)

if(CONFIG_MAIN["MODE"] == "DUMMY"):
    model = get_model_dummy(CONFIG_MAIN["DEVICE"]).eval()
elif(CONFIG_MAIN["MODE"] == "NORMAL"):
    model = load_model(
        CONFIG_MAIN["MODEL_PATH"], 
        CONFIG_MAIN["DEVICE"]
    ).eval()

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

for e, (images, labels, label) in enumerate(val_loader):
    predictions = []

    if(35 < e):
        for i in range(len(images)):
            image = images[i].to(CONFIG_MAIN["DEVICE"])
            target = labels[i].to(CONFIG_MAIN["DEVICE"])

            image = BIM(
                image,
                target,
                model,
                eps=CONFIG_MAIN["EPS"],
                k_number=CONFIG_MAIN["NUMBER_OF_ITERS"],
                alpha=CONFIG_MAIN["ALPHA"],
                device=CONFIG_MAIN["DEVICE"]
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

        torch.save(pred_sum_mask, CONFIG_MAIN['SAVE_FOLDER'] + "bim_full_sat/prediction_" + str(e) + ".pth")
        torch.save(label[0], CONFIG_MAIN['SAVE_FOLDER'] + "bim_full_sat/label_" + str(e) + ".pth")
