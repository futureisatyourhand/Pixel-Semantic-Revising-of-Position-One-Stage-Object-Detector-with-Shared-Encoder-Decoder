import torch.nn as nn
import numpy as np
from revise import CenterModule
from resnet50 import ResNet50
from fpns import FPNs
import torch
from center_loss import FCOSLoss
from test_select import TestSelect
from utils import EncoderDecoder
class Detect(nn.Module):
    def __init__(self,in_channels,nums,classes,fpn_strides,encoder_decoder_levels=3,layers=[3,4,6,3],overshold=0.5,topk=100,train=True,min_size=32):
        super(Detect,self).__init__()
        self.backbone=ResNet50(layers)##resnet50基本网络
        self.fpn=FPNs()##根据基本网络构建fpns
        self.head=CenterModule(in_channels,nums,classes)##fcos的检测器头部
        self.fpn_strides=fpn_strides##构建中心先验的步长
        self.train=train
        self.loss=FCOSLoss(0.45)##训练时的损失
        self.encoder_decoder=EncoderDecoder(in_chnnels,out_channels,encoder_decoder_levels)
        self.select=TestSelect(overshold,topk,min_size,num_classes)
        ##测试时根据先验候选中心locations以及预测的四个边距得到预测结果


    def createPrior(self,features):
        ###为所有level的特征图构建中心先验
        locations=[]
        for i,feature in enumerate(features):
            _,_,h,w=feature.shape
            stride=self.fpn_strides[i]
            y=torch.arange(0,h*stride,steps=stride,dtype=troch.float32,device=device)
            x=torch.arange(0,w*stride,steps=stride,dtype=torch.float=32,device=device)
            shift_x,shift_y=torch.meshgrid(x,y)
            shift_x,shift_y=shift_x.reshape(-1),shift_y.reshape(-1)
            location=torch.stack([shift_x,shift_y],dim=1)+stride//2
            locations.append(location)
    def forward(self,x,targets=None):
        features=self.backbone(x)
        features=self.fpn(features)
        features=self.encoder_decoder(features)
        box,cls,center,revise=self.head(features)
        locations=self.createPrior(features)
        if self.train:
            cls_loss,reg_loss,center_loss=self.loss(features,cls,box,center,revise,targets)
            return {"cls_loss":cls_loss,"reg_loss":reg_loss,"center_loss":center_loss}
        else:
            return self.select(box,cls,center,locations,revise)

