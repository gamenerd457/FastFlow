import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

import constants as const
import dataset
import fastflow2
import utils


import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

config_s="/content/FastFlow/configs/resnet18.yaml"
config_t="/content/FastFlow/configs/wide_resnet50_2.yaml"
data="/content/data"
category="bottle"


device = 'cpu'
def build_train_data_loader():
    train_dataset = dataset.MVTecDataset(
        root=data,
        category=category,
        input_size=(256,256),
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )


def build_test_data_loader():
    test_dataset = dataset.MVTecDataset(
        root=data,
        category=category,
        input_size=(256,256),
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )


def build_model(config):
    model = fastflow2.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )



class LSH(nn.Module):
    def __init__(self, input_dim, output_dim, std=1.0, with_l2=True, LSH_loss='BCE'):
        super(LSH, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std = std
        self.LSH_loss_type = LSH_loss

        self.LSH_weight = nn.Linear(self.input_dim, self.output_dim, bias=True)
        if with_l2:
            self.mse_loss = torch.nn.MSELoss(reduction='mean')
        else:
            self.mse_loss = None
        if LSH_loss == 'BCE':
            self.LSH_loss = nn.BCEWithLogitsLoss()
        elif LSH_loss == 'L2':
            self.LSH_loss = torch.nn.MSELoss(reduction='mean')
        elif LSH_loss == 'L1':
            self.LSH_loss = torch.nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError(LSH_loss)

        self._initialize()
        

    def _initialize(self):
        nn.init.normal_(self.LSH_weight.weight, mean=0.0, std=self.std)
        nn.init.constant_(self.LSH_weight.bias, 0)
        self.LSH_weight.weight.requires_grad_(False)
        self.LSH_weight.bias.requires_grad_(False)


    def init_bias(self, model_t, train_loader, print_freq=None, use_median=True):
        if use_median:
            print("=> Init LSH bias by median")
        else:
            print("=> Init LSH bias by mean")
        dataset_size = len(train_loader.dataset)
        if use_median:
            all_hash_value = torch.zeros(dataset_size, self.output_dim)
        else:
            mean = torch.zeros(self.output_dim)

        model_t.eval()

        for idx, data in enumerate(train_loader):
            input = data[0]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()

            # ============= forward ==============
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=False)
                feat_t = [f.detach() for f in feat_t]
                hash_t = self.LSH_weight(feat_t[-1])

            if use_median:
                index = data[-1]
                all_hash_value[index] = hash_t.cpu()
            else:
                mean += hash_t.sum(0).cpu() / dataset_size
            if print_freq is not None:
                if idx % print_freq == 0:
                    print("Init Bias: [{}/{}]".format(idx, len(train_loader)))

        if use_median:
            self.LSH_weight.bias.data[:] = - all_hash_value.median(0)[0]
        else:
            self.LSH_weight.bias.data[:] = - mean


    def forward(self, f_s, f_t):
        if self.mse_loss:
            l2_loss = self.mse_loss(f_s, f_t)
        else:
            l2_loss = 0
        hash_s = self.LSH_weight(f_s)
        hash_t = self.LSH_weight(f_t)
        if self.LSH_loss_type == 'BCE':
            pseudo_label = (hash_t > 0).float()
            loss = self.LSH_loss(hash_s, pseudo_label) 
        else:
            loss = self.LSH_loss(hash_s, hash_t)
        return l2_loss + loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train():
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_student = yaml.safe_load(open(config_s, "r"))
    config_teacher = yaml.safe_load(open(config_t, "r"))
    model_s = build_model(config_student)
    model_t= build_model(config_teacher)
    optimizer = build_optimizer(model_t)

    train_dataloader = build_train_data_loader()
    test_dataloader = build_test_data_loader()
    model_s.to(device)
    model_t.to(device)
    criterion_1=LSH(64*64*64,2)
    criterion_2=LSH(128*32*32,2)
    criterion_3=LSH(256*16*16,2)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_1)    # classification loss
    criterion_list.append(criterion_2)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_3)
    losses = AverageMeter('Loss', ':6.3f')
    model_t.eval()
    model_s.train()
    for i ,images in enumerate(train_dataloader):
      images=images.to(device)
      feat_s=model_s(images)
      with torch.no_grad():
        feat_t=model_t(images)
      #print(feat_s[0].shape)
      #print(feat_t[0].shape)

      l1=criterion_1(feat_s[0].reshape((32,64*64*64)),feat_t[0].reshape((32,64*64*64)))
      l2=criterion_2(feat_s[1].reshape((32,128*32*32)),feat_t[1].reshape((32,128*32*32)))
      l3=criterion_3(feat_s[2].reshape((32,256*16*16)),feat_t[2].reshape((32,256*16*16)))
      loss=l1+l2+l3
      losses.update(loss.item(), images.size(0))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    return losses.avg
    

if __name__ == "__main__":
  epochs=40
  for i in range(epochs):
    train_loss=train()
    print("Epoch {} /  train loss : {} ".format(i,train_loss))
