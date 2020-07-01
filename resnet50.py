import torch.nn as nn
class ResNet50(nn.Module):
    def __init__(self,layers):
        super(ResNet50,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPooling2d(kernel_size=3,stride=2,padding=1),
        )
        conv_list=[]
        channels=[64,128,256,512]
        self.layers=layers
        for i in xrange(len(layers)):
            conv_list.append(self.getModule(layers[i],channels[i],True))
        self.conv_list=conv_list
        #self.avg=nn.AdaptiveAvgPool2d((1,1))
        #self.fc=nn.Linear(2048,1000)
        #self.softmax=nn.SoftMax(-1)
        self.relu=nn.ReLU(inplace=True)
    #[64,128,256,512]
    def getModule(self,layers,in_channels,first=False)
        conv1=[]
        out_channels=in_channels*4
        mid=in_channels
        for i in xrange(layers):
             residual=None
            if i==0 and first==False:
                in_channels,s=in_channels*2,2
                residual=nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,3,s,1,bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            elif i==0 and first==True:
                in_channels,s=in_channels,2
                residual=nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,3,s,1,bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                in_channels,s=in_channels*4,1
            layer=nn.Sequential(
                nn.Conv2d(in_channels,mid,1,1,bias=True),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid,mid,3,s,1,bias=True),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid,out_channels,1,1,bias=True),
                nn.BatchNorm2d(out_channels),
            )
            conv1.append([layer,residual])
        return conv1
    def forward(self,x):
        fpn=[]
        for layers in self.conv_list:
            for layer in layers:
                residual=x
                if layer[1] is not None:
                    residual=layer[1](x)
                x=self.relu(layer[0](x)+residual)
            fpn.append(x)
        return fpn
