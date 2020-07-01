import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class Basic(torch.nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1,\
        dilation=0,bias=True,gp=True,ry=False,groups=64,separable=False):
        super(Basic,self).__init__()
        if separable==True:
            self.conv1=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,stride=stride,padding=padding,\
                      groups=in_channel,dilation=dilation,bias=True),
            nn.Conv2d(in_channel,out_channel,1,1,0,1,1,bias=bias),
            )
        else:
            self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.norm=nn.GroupNorm(32,out_channel) if gp==True else nn.BatchNorm2d(out_channel)
        self.relu=nn.LeakyReLU(inplace=False) if ry==True else nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.conv1(x)
        x=self.norm(x)
        x=self.relu(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self,nums,in_channels,out_channels):
        self.levels=nums
        for i in range(nums):
            settattr(self,'encode_level{}'.format(i+1),\
                Basic(in_channels,in_channels,stride=2))
             setattr(self,"airfacts_level{}".format(i+1),\
                 Basic(out_channels,out_channels,stride=2))
        self.flatten=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.GroupNorm(32,out_channels),
            nn.ReLU(inplace=True)
        )

        ##channel attention
        self.reduction=nn.Sequential(
            nn.Conv2d(in_channels+out_channels,out_channels,3,1,1),
            nn.GroupNorm(32,out_channels),
            nn.ReLU(inplace=True)
        )

        ##attention mechanism
        self.conv_spatial=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=1,padding=0,stride=1,bias=True),
            nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid())
        self.ex_avg=nn.Sequential(
            nn.Linear(out_channels,out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels//reduction,out_channels))

        self.ex_max=nn.Sequential(
            nn.Linear(out_channels,out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels//reduction,out_channels))
    def forward(self,features):
        rtn=[]
        for i,feature in enumerate(features):
            feature1=feature
            for level in range(self.levels):
                feature=getattr(self,"encode_level{}".format(level+1))(feature)
                setattr(self,"encode_feature{}".format(level+1),feature)

            ##middle
            fpn=self.flatten(getattr(self,"encode_feature{}".format(self.levels)))

            ##decoder
            level_features=[]
            for level in range(self.levels-1,-1,-1):
                fpn=F.interpolate(fpn,scale_factor=2,mode='bilinear')

                fpn=getattr(self,"airfacts_level{}".format(level+1))(fpn)
                b,c,h,w=fpn.size()
                atn_avg=F.avg_pool2d(fpn,(h,w),stride=(h,w))
                atn_max=F.max_pool2d(fpn,(h,w),stride=(h,w))
                ###experiment min-pooling 
                #atn_min=F.min_pool2d(fpn,(h,w),stride=(h,w))
                #channel_attention=nn.Sigmoid()(self.ex_min(atn_min.vew(b,c)))
                channel_attention=nn.Sigmoid()(torch.add(self.ex_avg(atn_avg.view(b,c)),self.ex_max(atn_max.view(b,c)/2.)))
                atn=channel_attention.view(b,c,1,1)
                fpn=fpn*atn

                ##experiment spatial attention
                #spatial_pool=torch.cat((torch.max(channel_refined,1)[0].unsqueeze(1),torch.mean(channel_refined,1).unsqueeze(1)),dim=1)
                #spatial=self.conv_spatial(spatial_pool)
                #del atn_avg,atn_max,channel_attention,atn,spatial_pool                
                #fpn=channel_refined*spatial
            #print(feature1.shape,ftn.shape)
            fpn=torch.cat((F.interpolate(fpn,size=(feature1.size()[-2],feature1.size()[-1]),mode='bilinear'),feature1),1)

            rtn.append(self.reduction(fpn))
            return rtn