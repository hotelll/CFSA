import torch
import torch.nn as nn
import math
from utils.builder import *
from utils.dpm_decoder import seg_and_interpolate
from utils.loss import frame2learnedstep_dist

FIX_LEN = 3


class TemporalNet(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim=128, fix_len=5):
        super(TemporalNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1)
        self.fc = nn.Linear(fix_len * num_filters, output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


class Align_adaK_learnStep(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=2048,
                 pretrain=None,
                 dropout=0):

        super(Align_adaK_learnStep, self).__init__()

        self.num_clip = num_clip
        self.dim_size = dim_size
        
        module_builder = Builder(num_clip, pretrain, False, dim_size)
        
        self.backbone = module_builder.build_backbone()
        self.bottleneck = nn.Conv2d(2048, 128, 3, 1, 1)
        
        self.get_token = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten(),
                                       Reshape(-1, self.num_clip, dim_size))
        
        self.step_encoder = nn.Sequential(nn.Conv1d(dim_size, dim_size, kernel_size=3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv1d(dim_size, dim_size, kernel_size=1, padding=0))
        
        self.temporal_learner = TemporalNet(input_dim=128, num_filters=128, output_dim=128, fix_len=FIX_LEN)
        
        self.global_net = nn.Sequential(Reshape(-1, self.num_clip * dim_size),
                                        nn.Linear(self.num_clip * dim_size, dim_size))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_size, num_class)
    
    
    def forward(self, x1, x2, embed=False):
        x1 = self.backbone(x1)
        x1 = self.bottleneck(x1)
        seq_features1 = self.get_token(x1)
        seq_features1 = seq_features1.permute(0,2,1)
        seq_features1 = self.step_encoder(seq_features1)
        seq_features1 = seq_features1.permute(0,2,1)
        
        
        x2 = self.backbone(x2)
        x2 = self.bottleneck(x2)
        seq_features2 = self.get_token(x2)
        seq_features2 = seq_features2.permute(0,2,1)
        seq_features2 = self.step_encoder(seq_features2)
        seq_features2 = seq_features2.permute(0,2,1)
        
        (B, T, C), device = seq_features1.shape, seq_features1.device
        # the similarity matrix: 16 * 16
        pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
        pred = pred.cumsum(-2).cumsum(-1)
        
        D = torch.zeros((B, T, T, T), device=device)
        D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
        
        D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
        
        area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
        area = area[:, :, :, None, None] - area[:, :, None, None, :] - \
            area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :]
        block_mat = pred[:, :, :, None, None] - pred[:, :, None, None, :] - \
            pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :]
        i, j, a, b = torch.meshgrid(*[torch.arange(T, device=device)]*4)
        area = area.clamp_min(1).sqrt()

        block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
        
        for k in range(1, T):
            tmp = D[:, k-1, None, None, :, :] + block_mat
            D[:, k] = torch.max(tmp.flatten(3), -1).values
            D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
        
        final_result = D[:, :, T-1, T-1]
        loss_step = -(final_result.max(dim=-1).values).mean()
        step_num = final_result.max(dim=-1).indices
        
        batched_seg_list1 = []
        batched_seg_list2 = []
        batched_step_list1 = []
        batched_step_list2 = []
        
        for batch in range(B):
            seg1_list = []
            seg2_list = []
            i, j, a, b = T-1, T-1, T-1, T-1 
            # k = step_num[batch].item()
            k = 
            
            step_list1 = []
            step_list2 = []
            
            video1_end = T
            video2_end = T
            
            while k > 0:
                ind = D_ind[batch, k, i, j].item()
                a = ind // T
                b = ind % T
                
                video1_start = a + 1
                video2_start = b + 1
                step_feature1 = seg_and_interpolate(seq_features1[batch], video1_start, video1_end, FIX_LEN)
                step_feature2 = seg_and_interpolate(seq_features2[batch], video2_start, video2_end, FIX_LEN)
                step_feature1 = self.temporal_learner(step_feature1)
                step_feature2 = self.temporal_learner(step_feature2)
                
                step_list1.insert(0, step_feature1)
                step_list2.insert(0, step_feature2)
                video1_end = video1_start
                video2_end = video2_start
            
                seg1_list.insert(0, a)
                seg2_list.insert(0, b)
                i, j, k = a, b, k-1
            
            video1_start = 0
            video2_start = 0
            step_feature1 = seg_and_interpolate(seq_features1[batch], video1_start, video1_end, FIX_LEN)
            step_feature2 = seg_and_interpolate(seq_features2[batch], video2_start, video2_end, FIX_LEN)
            step_feature1 = self.temporal_learner(step_feature1)
            step_feature2 = self.temporal_learner(step_feature2)
            step_list1.insert(0, step_feature1)
            step_list2.insert(0, step_feature2)
            
            step_features1 = torch.stack(step_list1, dim=1)
            step_features2 = torch.stack(step_list2, dim=1)
            seg_tensor1 = torch.tensor(seg1_list, device=device)
            seg_tensor2 = torch.tensor(seg2_list, device=device)
            
            batched_step_list1.append(step_features1[0])
            batched_step_list2.append(step_features2[0])
            
            batched_seg_list1.append(seg_tensor1)
            batched_seg_list2.append(seg_tensor2)
        
        frame2step_dist = frame2learnedstep_dist(seq_features1, batched_step_list2) \
                            + frame2learnedstep_dist(seq_features2, batched_step_list1)
        
        
        # return batched_step_list1, batched_step_list2, batched_seg_list1, batched_seg_list2
        global_features1 = self.global_net(seq_features1)
        global_features2 = self.global_net(seq_features2)
        
        if embed:
            return global_features1, global_features2, seq_features1, seq_features2
        
        global_features1 = self.dropout(global_features1)
        pred1 = self.cls_fc(global_features1)
        
        global_features2 = self.dropout(global_features2)
        pred2 = self.cls_fc(global_features2)
        
        return pred1, pred2, frame2step_dist, loss_step
