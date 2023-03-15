import glob
import sys
import torch
import json
import numpy as np

sys.path.append('./')

from dataset.meatrics import AverageMeter, intersectionAndUnionGPU

def test(path):
    CONFIG_PATH_MAIN = "./configs/config_main.json"
    CONFIG_MAIN = json.load(open(CONFIG_PATH_MAIN))

    masks = glob.glob(path + "/prediction_*.pth")
    labels = glob.glob(path + "/label_*.pth")
    
    masks.sort()
    labels.sort()

    print("Number of masks:", len(masks))
    print("Number of labels:", len(labels))

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i in range(len(labels)):
        if(i % 20 == 0):
            print("Finished:", i*100/len(labels), "%")
        
        m = torch.load(masks[i])
        l = torch.load(labels[i])
        
        m[l == 255] = 255

        intersection_normal, union_normal, target_normal = intersectionAndUnionGPU(
            m,
            l,
            CONFIG_MAIN['CLASSES'],
            CONFIG_MAIN["IGNOR_LABEL"]
        )

        intersection_normal = intersection_normal.cpu().numpy()
        union_normal = union_normal.cpu().numpy()
        target_normal = target_normal.cpu().numpy()

        intersection_meter.update(intersection_normal)
        union_meter.update(union_normal)
        target_meter.update(target_normal)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print("mIoU", mIoU)
    print("mAcc", mAcc)
    print("allAcc", allAcc)


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise ValueError("Wrong command line argument...")
    else:
        results = glob.glob(sys.argv[1] + "/*")

        for r in results:
            print("\n" + r)
            test(r)
