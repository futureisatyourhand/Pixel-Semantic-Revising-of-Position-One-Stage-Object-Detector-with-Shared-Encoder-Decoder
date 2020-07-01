import torch.nn as nn
import numpy as np

class IOUs(nn.Module):
    def __init__(self,iou_type='giou'):
        super(IOUs,self).__init__()
        self.iout_type=iou_type

    ##构建iou损失和giou损失
    def forward(self,prediction,targets,weights=None):
        min_w=torch.min(prediction[:,0],targets[:,0])+torch.min(prediction[:,2],targets[:,2])
        min_h=torch.min(prediction[:,1],targets[:,1])+torch.min(prediction[:,3],targets[:,3])
        insect=min_w*min_h
        pred_w=prediction[:,0]+prediction[:,2]
        pred_h=prediction[:,1]+predcition[:,3]
        pred_area=pred_w*pred_h
        target_w=targets[:,0]+targets[:,2]
        target_h=targets[:,1]+targets[:,3]
        target_area=target_w*target_h
        areas_union=target_area+pred_area-insect
        ious=(insect+1.0)/(areas_union+1.0)

        max_w=torch.max(prediction[:,0],targets[:,0])+torch.max(prediction[:,2],targets[:,2])
        max_h=torch.max(prediction[:,1],targets[:,1])+torch.max(prediction[:,3],targets[:,3])
        max_areas=max_w*max_h+1e-7

        gious=ious-(max_areas-areas_union)/max_areas
        if self.iou_type=='iou':
            loss=-torch.log(ious)
        elif self.iou_type=='giou':
            loss=1-gious
        elif self.iou_type=='linear_iou':
            loss=1-ious
        
        if weights is not None and weights.sum()>0:
            return (loss*weights).sum()
        else:
            assert loss.numel()!=0
            return loss.sum()