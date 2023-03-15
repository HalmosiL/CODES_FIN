from torch.autograd import Variable
import torch
import torch.nn as nn

mean_origin = [0.485, 0.456, 0.406]
std_origin = [0.229, 0.224, 0.225]

class Adam_optimizer:
    def __init__(self, B1, B2, lr):
        self.B1 = B1
        self.B2 = B2
        self.lr = lr

        self.m_t = 0
        self.v_t = 0

        self.t = 1
        self.e = 1e-08

    def step_grad(self, grad, image):
        self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
        self.v_t = self.B2 * self.v_t + (1 - self.B2) * (grad ** 2)

        m_l = self.m_t / (1 - self.B1 ** self.t)
        v_l = self.v_t / (1 - self.B2 ** self.t)

        self.t += 1

        return (self.lr * m_l) / (torch.sqrt(self.v_t) + self.e)

    def step(self, grad, image):
        self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
        self.v_t = self.B2 * self.v_t + (1 - self.B2) * (grad ** 2)

        m_l = self.m_t / (1 - self.B1 ** self.t)
        v_l = self.v_t / (1 - self.B2 ** self.t)

        self.t += 1

        image = image - (self.lr * m_l) / (torch.sqrt(self.v_t) + self.e)

        return image

def PGD(input, target, model, clip_min, clip_max, optimizer=None, device="cpu"):
    input_variable = input.detach().clone()
    input_variable.requires_grad = True
    model.zero_grad()
    _, result = model(input_variable)

    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).to(device)
    loss = criterion(result, target.detach())
    loss.backward()
    
    ################################################################################
    adversarial_example = input.detach().clone()
    adversarial_example[:, 0, :, :] = adversarial_example[:, 0, :, :] * std_origin[0] + mean_origin[0]
    adversarial_example[:, 1, :, :] = adversarial_example[:, 1, :, :] * std_origin[1] + mean_origin[1]
    adversarial_example[:, 2, :, :] = adversarial_example[:, 2, :, :] * std_origin[2] + mean_origin[2]
    adversarial_example = optimizer.step(-1*input_variable.grad, adversarial_example)
    adversarial_example = torch.max(adversarial_example, clip_min)
    adversarial_example = torch.min(adversarial_example, clip_max)
    adversarial_example = torch.clamp(adversarial_example, min=0.0, max=1.0)

    adversarial_example[:, 0, :, :] = (adversarial_example[:, 0, :, :] - mean_origin[0]) / std_origin[0]
    adversarial_example[:, 1, :, :] = (adversarial_example[:, 1, :, :] - mean_origin[1]) / std_origin[1]
    adversarial_example[:, 2, :, :] = (adversarial_example[:, 2, :, :] - mean_origin[2]) / std_origin[2]
    ################################################################################
    return adversarial_example



def BIM(input, target, model, eps=0.01, k_number=2, alpha=0.01, device="cpu"):
    optimizer = Adam_optimizer(lr=alpha, B1=0.9, B2=0.99)
    
    input_unnorm = input.clone().detach()
    input_unnorm[:, 0, :, :] = input_unnorm[:, 0, :, :] * std_origin[0] + mean_origin[0]
    input_unnorm[:, 1, :, :] = input_unnorm[:, 1, :, :] * std_origin[1] + mean_origin[1]
    input_unnorm[:, 2, :, :] = input_unnorm[:, 2, :, :] * std_origin[2] + mean_origin[2]
    clip_min = input_unnorm - eps
    clip_max = input_unnorm + eps

    adversarial_example = input.detach().clone()
    adversarial_example.requires_grad = True
    
    for mm in range(k_number):
        adversarial_example = PGD(adversarial_example, target, model, clip_min, clip_max, optimizer, device)
        adversarial_example = adversarial_example.detach()
        adversarial_example.requires_grad = True
        model.zero_grad()
    return adversarial_example
