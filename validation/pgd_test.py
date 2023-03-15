import torch
import json
import sys
import numpy as np

sys.path.append('../')

from modules.model import load_model, get_model_dummy
from attacks.pgd import BIM
from dataset.dataset import SemData
from dataset.meatrics import AverageMeter, intersectionAndUnionGPU
import dataset.transform as transform

CONFIG_PATH_MAIN = "../configs/config_main.json"
CONFIG_MAIN = json.load(open(CONFIG_PATH_MAIN ))

if(CONFIG_MAIN["MODE"] == "DUMMY"):
    model = get_model_dummy(CONFIG_MAIN["DEVICE"]).eval()
elif(CONFIG_MAIN["MODE"] == "NORMAL"):
    model = load_model(
        CONFIG_MAIN["MODEL_PATH"], 
        CONFIG_MAIN["DEVICE"]
    ).eval()

intersection_meter = AverageMeter()
union_meter = AverageMeter()
target_meter = AverageMeter()

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

val_transform = transform.Compose([
    transform.Crop(
        [CONFIG_MAIN["HEIGHT"], CONFIG_MAIN["WIGHT"]],
        crop_type='center',
        padding=mean,
        ignore_label=CONFIG_MAIN["IGNOR_LABEL"]
    ),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])

val_loader = torch.utils.data.DataLoader(
    dataset=SemData(
        split='val',
        data_root=CONFIG_MAIN['DATA_PATH'],
        data_list=CONFIG_MAIN['IMAGE_LIST'],
        transform=val_transform
    ),
    batch_size=1,
    num_workers=CONFIG_MAIN['NUMBER_OF_WORKERS'],
    pin_memory=CONFIG_MAIN['PIN_MEMORY']
)

while(True):
    for i, (image, target) in enumerate(val_loader):
        image = image.to(CONFIG_MAIN["DEVICE"])
        target = target.to(CONFIG_MAIN["DEVICE"])

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
            pred = model(image)
            pred = pred.max(1)[1]
        elif(CONFIG_MAIN["MODE"] == "NORMAL"):
            pred, _ = model(image)

        intersection_normal, union_normal, target_normal = intersectionAndUnionGPU(
            pred,
            target,
            CONFIG_MAIN['CLASSES'],
            CONFIG_MAIN["IGNOR_LABEL"]
        )

        intersection_normal, union_normal, target_normal = intersection_normal.cpu().numpy(), union_normal.cpu().numpy(), target_normal.cpu().numpy()
        intersection_meter.update(intersection_normal), union_meter.update(union_normal), target_meter.update(target_normal)

        if(i % 10 == 0):
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

            print("mIoU", mIoU)
            print("mAcc", mAcc)
            print("allAcc", allAcc)

print("-------------------------------Final------------------------------------")

iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
mIoU = np.mean(iou_class)
mAcc = np.mean(accuracy_class)
allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

print("mIoU", mIoU)
print("mAcc", mAcc)
print("allAcc", allAcc)
