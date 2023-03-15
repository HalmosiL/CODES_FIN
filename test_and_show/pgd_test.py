import torch
import json
import sys
import numpy as np
import cv2
import os
import time

sys.path.append('../')

from modules.model import load_model, get_model_dummy
from attacks.pgd import BIM
from dataset.dataset import SemDataSplit
import dataset.transform as transform

CONFIG_PATH_MAIN = sys.argv[1]
CONFIG_MAIN = json.load(open(CONFIG_PATH_MAIN))

try:
    os.mkdir(CONFIG_MAIN['SAVE_FOLDER'])
except OSError as error:
    print("Folder is alredy exist...")

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
    image_list = []
    adv_image_list = []

    if(-1 < e):
        for i in range(len(images)):
            image_original = images[i].to(CONFIG_MAIN["DEVICE"])
            target = labels[i].to(CONFIG_MAIN["DEVICE"])

            print(image_original.shape)

            start_time = time.time()

            image = BIM(
                image_original.clone(),
                target,
                model,
                eps=CONFIG_MAIN["EPS"],
                k_number=CONFIG_MAIN["NUMBER_OF_ITERS"],
                alpha=CONFIG_MAIN["ALPHA"],
                device=CONFIG_MAIN["DEVICE"]
            )

            print("Finished", i)
            print("--- %s seconds ---" % (time.time() - start_time))

            if(CONFIG_MAIN["MODE"] == "DUMMY"):
                _, pred = model(image)
                pred = pred.max(1)[1]
            elif(CONFIG_MAIN["MODE"] == "NORMAL"):
                pred, _ = model(image)

            adv_image_list.append(image[0])
            image_list.append(image_original[0])
            predictions.append(pred)

        pred_sum_mask = torch.zeros(898, 1796)
        adv_image_ = torch.zeros(3, 898, 1796)
        image_ = torch.zeros(3, 898, 1796)

        i = 0

        for x in range(2):
            for y in range(4):
                pred_sum_mask[x*449:(x+1)*449, y*449:(y+1)*449] = predictions[i]
                i += 1

        i = 0

        for x in range(2):
            for y in range(4):
                adv_image_[:, x*449:(x+1)*449, y*449:(y+1)*449] = adv_image_list[i]
                i += 1

        i = 0

        for x in range(2):
            for y in range(4):
                image_[:, x*449:(x+1)*449, y*449:(y+1)*449] = image_list[i]
                i += 1

        torch.save(pred_sum_mask, CONFIG_MAIN['SAVE_FOLDER'] + "prediction_" + str(e) + ".pth")
        torch.save(label[0], CONFIG_MAIN['SAVE_FOLDER'] + "label_" + str(e) + ".pth")
        torch.save(adv_image_, CONFIG_MAIN['SAVE_FOLDER'] + "adv_image_" + str(e) + ".pth")
        torch.save(image_, CONFIG_MAIN['SAVE_FOLDER'] + "image_" + str(e) + ".pth")
