import torch.nn as nn
import torch

mean_origin = [0.485, 0.456, 0.406]
std_origin = [0.229, 0.224, 0.225]

def FGSM(input, target, model, clip_min, clip_max, eps=0.2, device="cpu"):
    input_variable = input.detach().clone()
    input_variable.requires_grad = True

    model.zero_grad()
    _, result = model(input_variable)

    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).to(device)
    loss = criterion(result, target.detach())
    loss.backward()

    print("loss", loss.item())
    res = input_variable.grad

    adversarial_example = input.detach().clone()

    adversarial_example[:, 0, :, :] = adversarial_example[:, 0, :, :] * std_origin[0] + mean_origin[0]
    adversarial_example[:, 1, :, :] = adversarial_example[:, 1, :, :] * std_origin[1] + mean_origin[1]
    adversarial_example[:, 2, :, :] = adversarial_example[:, 2, :, :] * std_origin[2] + mean_origin[2]

    adversarial_example = adversarial_example + eps * torch.sign(res)
    adversarial_example = torch.max(adversarial_example, clip_min)
    adversarial_example = torch.min(adversarial_example, clip_max)
    adversarial_example = torch.clamp(adversarial_example, min=0.0, max=1.0)

    adversarial_example[:, 0, :, :] = (adversarial_example[:, 0, :, :] - mean_origin[0]) / std_origin[0]
    adversarial_example[:, 1, :, :] = (adversarial_example[:, 1, :, :] - mean_origin[1]) / std_origin[1]
    adversarial_example[:, 2, :, :] = (adversarial_example[:, 2, :, :] - mean_origin[2]) / std_origin[2]

    return adversarial_example

def BIM(input, target, model, eps=0.015, k_number=2, alpha=0.01, device="cpu"):
    input_unnorm = input.clone().detach()
    input_unnorm[:, 0, :, :] = input_unnorm[:, 0, :, :] * std_origin[0] + mean_origin[0]
    input_unnorm[:, 1, :, :] = input_unnorm[:, 1, :, :] * std_origin[1] + mean_origin[1]
    input_unnorm[:, 2, :, :] = input_unnorm[:, 2, :, :] * std_origin[2] + mean_origin[2]
    clip_min = input_unnorm - eps
    clip_max = input_unnorm + eps

    adversarial_example = input.detach().clone()
    adversarial_example.requires_grad = True
    for mm in range(k_number):
        adversarial_example = FGSM(adversarial_example, target, model, clip_min, clip_max, eps=alpha, device=device)
        adversarial_example = adversarial_example.detach()
        adversarial_example.requires_grad = True
        model.zero_grad()
    return adversarial_example
