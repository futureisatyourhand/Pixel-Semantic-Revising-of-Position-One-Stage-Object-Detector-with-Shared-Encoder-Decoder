from resnet50 import ResNet50
import torch.nn as nn
class FPNs(nn.Module):
    def __init__(self):
        super(FPNs,self).__init__()
        self.latelayer4=nn.Conv2d(2048,256,1,1,0)
        ###for upsample
        self.latelayer3=nn.Conv2d(1024,256,1,1,0)
        self.latelayer2=nn.Conv2d(512,256,1,1,0)
        self.latelayer1=nn.Conv2d(256,256,1,1,0)

        ##for smooth
        self.smooth3=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth1=nn.Conv2d(256,256,3,1,1)
    def upsample_add(self,x,y):
        return nn.functional.upsample(x,size=(y.shape[-2],y.shape[-1]),mode='bilinear')+y
    def forward(self,x):
        p4=self.latelayer4(x[-1])
        p3=self.smooth3(self.upsample_add(p4,self.latelayer3(x[2])))
        p2=self.smooth2(self.upsample_add(p3,self.latelayer2(x[1])))
        p1=self.smooth1(self.upsample_add(p2,self.latelayer1(x[0])))
        return [p1,p2,p3,p4]

def createFpns(layers,images):
    resnet50=ResNet50(layers)
    p=resnet50(x)
    fpns=FPNs()
    return fpns(p)