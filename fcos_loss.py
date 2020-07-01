import torch.nn as nn
from sigmoid_ious import IOUs
INF=1000000
class FCOSLoss(nn.Module):
    def __init__(self,overthreshold):
        self.overthreshold=overthreshold
        self.iou_loss=IOUs()
        self.cls_loss=nn.BCEWithLogits()
        self.centerness_loss=nn.BCEWithLogitsLoss(reduction='sum')
    def computation_targets(self,points,targets):
        ###不同level级别的特征图检测不同尺寸的对象
        #[中心点与四个边界的最小距离，中心点与四个边界的最大距离]
        range_areas=[
            [-1,64],#浅层检测小对象
            [64,128],
            [128,256],
            [256,512],
            [512,INF]##高层检测大对象
        ]
        ###构建不同fpns级别学习不同尺寸的对象，1：-1-64；2：64-128.。。
        ##构建与每层具有相同候选点数目的边距范围候选
        expanded_size=[]
        for i,point in enumerate(points):
            ares=point.new_tensor(range_areas[i])
            expanded_size.append(ares[None].repeat(len(points[i]),-11))
        expanded_siz=torch.cat(expanded_size,dim=0)

        self.nums_of_points=[len(point) for point in points]
        points_all=torch.cat(points,dim=0)
        ##为所有的图片的所有对象构建目标label和target，使得每个中心点只作为一个对象的prior
        labels,reg_target=self.computate_point_for_location(points_all,targets,expanded_siz)

        ###根据构建的目标label与候选
        # 原始labels结构为[图片1的所有level的目标label，图片2的所有level的目标label,...]
        #原始reg_targets的结构为[图片1的所有level的目标边距，图片2的所有level的目标边距，...]
        ###将原始这种结构划分为
        #[[图片1的level1的目标label,图片1的level2的目标label,...],[图片2的level1的目标label,图片2的level2的目标label,...]]
        ##和[[图片1的level1的目标边距，图片1的level2的目标边距,..],[图片2的level1的目标边距，图片2的level2的目标边距,..]]
        N=len(targets)###图片的数量
        for i in range(N):
            labels[i]=torch.split(labels[i],self.nums_of_points,dim=0)
            reg_target[i]=torch.split(reg_target[i],self.nums_of_points,dim=0)

        ####将划分后的目标label和目标边距后修改结构为
        #[[图片1的level1的目标label,图片2的level1的目标label,...],[图片1的level2的目标label,图片2的level2的目标label,...]]
        #[[图片1的level1的目标边距,图片2的level1的目标边距,...],[图片1的level2的目标边距,图片2的level2的目标边距,...]]
        labels_of_level,targets_of_level=[],[]
        for level in xrange(len(points)):
            labels_of_level.append(
                torch.cat([labels_level[level] for labels_level in labels],dim=0)
                )
            targets_of_level.append(
                torch.cat([reg_target_level[level] for reg_target_level in reg_target],dim=0)
                )
        return labels_of_level,targets_of_level 


    def computate_point_for_location(self,locations,targets,target_of_areas):
        xs=locations[:,0]
        ys=locations[:,1]
        N=len(targets)
        labelss,regs=[],[]
        for i in range(N):
            box_target=targets[i][0]###xyxy
            labels_target=targets[i][1]##one-hot
            areas_target=targets[i][2]##areas
            left=xs[:,None]-box_target[:,0][None]
            right=box_target[:,2][None]-xs[:,None]
            top=ys[:,None]-box_target[:,1][None]
            bottom=box_target[:,3][None]-ys[:,None]
            reg_targets=torch.stack([left,top,right,bottom],dim=2)
            is_in_prior=reg_targets.min(dim=2)[0]>0

            areas_in_prior=(reg_targets.min(dim=2)[0]>=target_of_ares[:,0])&\
                (reg_targets.max(din=2)[0]<=target_of_ares[:,1])
            location_of_areas=areas_target.repeat(len(locations),1)
            location_of_areas[areas_in_prior==0]=INF
            location_of_areas[is_in_prior==0]=INF
            ##当一个prior中心点有多个对象时，只选择面积最小的那个对象
            target_of_prior_values,target_of_prior_ind=location_of_areas.min(dim=1)
            reg_targets=reg_targets[range(len(locations)),target_of_prior_ind]
            labels_target=label_target[target_of_prior_ind]

            ##每个level的特征图规定了检测使用规定面积的对象
            ###这里其实主要是为了让同一个图片中的不同尺寸的对象使用不同层的中心先验进行检测
            labels_target[target_of_prior_values==INF]=0
            
            labels.append(labels_target)
            regs.append(reg_targets)
        return labels,regs
    ##使用min(left,right)/max(left,right) * min(top,bottom)/max(top,bottom)
    def get_centerness_target(self,reg_target_flatten):
        left_right=reg_target_flatten[:,[0,2]]
        top_bottom=reg_target_flatten[:,[1,3]]
        centerness=(left_right.min(dim=-1)[0]/left_right.max(dim=-1)[0])*(top_bottom.min(dim=-1)[0]/top_bottom.max(dim=-1)[0])

    ###
    def forward(self,locations,cls,box,centerness,targets):
        batch,num_classes,h,w=cls[0].shape
        label,reg_target=self.computation_targets(locations,targets)
        box_label_flatten=[]
        box_regression_flatten=[]
        box_centerness_flatten=[]
        reg_targets_flatten=[]
        label_target_flatten=[]
        for i in range(len(locations)):
            box_label_flatten.append(cls[i].permute(0,2,3,1).reshape(-1,num_classes))
            box_regression_flatten.append(box[i].permute(0,2,3,1).reshape(-1,4))
            box_centerness_flatten.append(centerness[i].permute(0,2,3,1).reshape(-1))
            reg_targets_flatten.append(reg_target[i].reshape(-1,4))
            label_target_flatten.append(label[i].reshape(-1))
        ##进行拼接，比如[[][]]和[[][]]拼接为[[][][][]]
        box_label_flatten=torch.cat(box_label_flatten,dim=0)
        box_regression_flatten=torch.cat(box_regression_flatten,dim=0)
        box_centerness_flatten=torch.cat(box_centerness_flatten,dim=0)
        reg_targets_flatten=torch.cat(reg_targets_flatten,dim=0)
        label_target_flatten=torch.cat(label_target_flatten,dim=0)

        ##筛选正负样本，根据目标分类得分是否大于0来判断，获取正样本
        pos_inds=torch.nonzero(label_target_flatten>0).squeeze(1)
        box_regression_flatten=box_regression_flatten[pos_inds]
        box_centerness_flatten=box_centerness_flatten[pos_inds]
        reg_targets_flatten=reg_targets_flatten[pos_inds]

        cls_loss=self.cls_loss(box_label_flatten,label_target_flatten.int())
        if pos_inds.numel()>0:
            centerness_weights=self.get_centerness_target(reg_targets_flatten)##得到中心点到周围的比例权重
            ##进行加权giou损失，这里的权重就是目标中心点到四个边的比例权重
            reg_loss=self.iou_loss(box_regression_flatten,reg_targets_flatten,centerness_weights)
            ###根据预测的中心比例权重和真实的中心比例权重得到中心点损失
            center_loss=self.centerness_loss(box_centerness_flatten,centerness_weights)
        else:
            reg_loss=box_regression_flatten.sum()
            centerness_loss=box_centerness_flatten.sum()
        return cls_loss,reg_loss,center_loss