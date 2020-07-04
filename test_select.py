import torch.nn as nn
import torch
import numpy as np

class TestSelect(nn.Module):
    def __init__(self,overshold,topk,min_size,num_classes):
        super(TestSelect,self).__init__()
        self.overshold=overshold
        self.topk=topk
        self.num_classes=num_classes
        self.min_size=min_size
    def feature_for_forward(self,box,cls,center,point,revise):
        n,c,h,w=cls.shape
        cls=cls.view(n,c,h,w).permute(0,2,3,1).reshape(n,-1,c).sigmoid()
        box=box.view(n,4,h,w).permute(0,2,3,1).reshape(n,-1,4)
        center=center.view(n,1,h,w).permute(0,2,3,1).reshape(n,-1).sigmoid()
        revise=revise.reshape(n,-1,2)

        candidates_inds=cls>self.overshold
        pre_nms_topk=candidates_inds.view(n,-1).sum(1)
        pre_nms_topk=pre_nms_topk.clamp(max=self.topk)

        cls=cls*center[:,:,None]
        results=[]
        for i in xrange(n):          
            per_candidate=candidates_inds[i]
            per_cls=cls[i]
            per_cls=per_cls[per_candidate]

            per_candidate_nonzero=per_candidate.nonzero()
            per_candidate_loc=per_candidate_nonzero[:,0]
            per_box=box[i]
            per_box=per_box[per_candidate_loc]##获取得分大于阈值的定位预测
            point=point[per_candidate_loc]##获取得分大于阈值的中心点候选
            rev=revise[i][per_candidate_loc]
            point=point+rev

            per_candidate_class=per_candidate_nonzero[:,1]+1###分类得分大于阈值的类别
            per_pre_nms_topk=pre_nms_topk[i]

            ####当得分大于阈值的候选框数量超过了topk，则使用分类得分最高的topk个
            if per_candidate.sum().item()>per_pre_nms_topk.item():
                per_cls,top_k=per_cls.topk(per_pre_nms_topk,sorted=False)
                per_candidate_class=per_candidate_class[top_k]
                per_box=per_box[top_k]
                point=point[top_k]
            detections=torch.stack([
                point[:,0]-per_box[:,0],point[:,1]-per_box[:,1],
                point[:,0]+per_box[:,2],point[:,1]+per_box[:,3]
                ],dim=1)
            results.append([detections,per_candidate_class,per_cls])
        return results
    def forward(self,box,cls,center,points,revise):
        sample_boxes=[]
        for i,(b,l,c,p) in enumerate(zip(box,cls,center,points,revise)):
            sample_boxes.append(self.feature_for_forward(b,l,c,p,r))
        
        return sample_boxes


