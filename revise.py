import torch.nn as nn
import torch

class CenterModule(nn.Module):
    del __init__(self,in_channels,nums,classes,levels):
        super(CenterModule,self).__init__()
        cls_list=[]
        box_list=[]
        for i in xrange(nums):
            cls_list.append(nn.Conv2d(in_channels,in_channels,3,1,1,bias=True))
            cls_list.append(nn.GroupNorm(32,in_channels))
            cls_list.append(nn.ReLU(inplace=True))

            box_list.append(nn.Conv2d(in_channels,in_channels,3,1,1,bias=True))
            box_list.append(nn.GroupNorm(32,in_channels))
            box_list.append(nn.ReLU(inplace=True))
        self.scale=[nn.Parameter(torch.FloatTensor(1.0)) for _ xrange(4)]
        self.cls_conv=nn.Sequential(*cls_list)
        self.box_conv=nn.Sequential(*box_list)
        self.cls_pred=nn.Conv2d(in_channels,classes,3,1,1)
        self.box_pred=nn.Conv2d(in_channels,4,3,1,1)
        self.center_pred=nn.Conv2d(in_channels,1,1,1)
        self.semantic_center=nn.Conv2d(in_channels,2,3,1,1)
        self.scale_center=[nn.Parameter([0.2]) for _ in range(levels)]

    def forward(self,features):
        box_prediction,cls_prediction,center_prediction,revise_predicition=[],[],[],[]
        for layer,feature in enumerate(features):
            box_feature=self.box_conv(feature)
            cls_feature=self.cls_conv(feature)
            center_prediction.append(self.center_pred(cls_feature))
            cls_prediction.append(self.cls_pred(cls_feature))
            revise_predicition.append(self.scale_center[layer](self.semantic_center(cls_feature).permute([0,2,3,1])))

            box=self.box_pred(box_feature)
            box=self.scale[i]*(box)
            box_prediction.append(torch.exp(box))
        return box_prediction,cls_prediction,center_prediction,revise_prediction


