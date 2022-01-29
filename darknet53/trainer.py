import os

import numpy as np

import torch.nn
from mobilenet_v2_module import *
import dataset
from darbnet53_module import *
from torch import nn

def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)#N,45,13,13==>N,13,13,45
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)#N,13,13,3,15
    # print("output:",output.shape)
    mask_obj = target[..., 0] > 0#N,13,13,3
    # print("mask_obj:",mask_obj.shape)
    mask_noobj = target[..., 0] == 0
    # print("mask_noobj:",mask_noobj.shape)
    # print("output[mask_obj]:",output[mask_obj].shape)
    # print("output[mask_noobj]:", output[mask_noobj].shape)
    loss_p_fun=nn.BCELoss()
    loss_p=loss_p_fun(torch.sigmoid(output[...,0]),target[...,0])
    loss_box_fun=nn.MSELoss()
    loss_box=loss_box_fun(output[mask_obj][...,1:5],target[mask_obj][...,1:5])
    loss_cls_box_fun=nn.CrossEntropyLoss()
    loss_cls_box=loss_cls_box_fun(output[mask_obj][...,5:],torch.argmax(target[mask_obj][...,5:],dim=1,keepdim=True).squeeze(dim=1))
    loss = alpha * loss_p + (1-alpha)*0.5*loss_box+ (1-alpha)*0.5*loss_cls_box
    return loss


if __name__ == '__main__':
    weight_path= 'darknet_params/net597.pt'
    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=5, shuffle=True)

    net = Darknet53().cuda()
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    net.train()

    opt = torch.optim.Adam(net.parameters())
    epoch = 0
    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            target_13, target_26, target_52, img_data=target_13.cuda(), target_26.cuda(), target_52.cuda(), img_data.cuda()
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13.float(), target_13.float(), 0.6)
            loss_26 = loss_fn(output_26.float(), target_26.float(), 0.6)
            loss_52 = loss_fn(output_52.float(), target_52.float(), 0.6)
            #
            loss = loss_13 + loss_26 + loss_52
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(epoch,loss.item())
        torch.save(net.state_dict(), f'darknet_params/net{epoch}.pt')
        print(f'{epoch}保存成功')
        epoch+=1