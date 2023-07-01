from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  

class MAXPrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(MAXPrototypeContrastLoss, self).__init__()

        self.temperature = 1
        self.m = 10
        self.n = 20
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def _negative_construct(self, feas, labels_): 
        unique_labels = torch.unique(labels_)
        fea_dict = dict()
        for i in range(len(unique_labels)):
            index = torch.where(labels_ == unique_labels[i])
            fea_dict[unique_labels[i].item()]=rearrange(feas[index],'b c h w->b (h w) c')
        return fea_dict  

    def _contrastive(self, base_fea, pos_fea, neg_dict, labels_, Q_labels):

        loss = torch.zeros(1).cuda()

        for base, pos, cls, Q_gt in zip(base_fea, pos_fea, labels_, Q_labels):
            idx = torch.where(Q_gt.flatten(0)==1)[0]
            base = rearrange(base,'c h w->c (h w)').unsqueeze(0)
            pos  = rearrange(pos, 'c h w->(h w) c').unsqueeze(0)
            positive_dot_contrast = self.generate_maxsimliarity(base,pos)
            positive_dot_contrast = 0.5*(torch.index_select(positive_dot_contrast, dim = -1, index = idx).mean(-1)+1)
            # positive_dot_contrast = torch.einsum('cn,mc->nm',base,pos).max(-1)[0].sum(-1)
            # positive_dot_contrast = torch.div(F.cosine_similarity(base, pos,0),
            #                             self.temperature)
            negative_samples = neg_dict[cls.item()]
            if negative_samples.shape[0]>self.m:
                perm = torch.randperm(negative_samples.shape[0])
                negative_samples = negative_samples[perm[:self.m]]
            negative_dot_contrast = self.generate_maxsimliarity(base, negative_samples)
            negative_dot_contrast = 0.5*(torch.index_select(negative_dot_contrast, dim = -1, index = idx).mean(-1)+1)
                                        # self.temperature)

            # pos_logits  = torch.exp(positive_dot_contrast)
            # neg_logits = torch.exp(negative_dot_contrast).sum()
            # mean_log_prob_pos = torch.log((pos_logits/(pos_logits+neg_logits))+1e-8)
            mean_log_prob_pos = -torch.log(positive_dot_contrast+1e-8)#-torch.log(1-negative_dot_contrast+1e-8).sum()
            loss = loss + mean_log_prob_pos.mean()

        return loss/len(labels_)

    def forward(self, Q_feats, S_feats, Q_predit, Q_labels, S_labels, query_bg_out, supp_bg_out, classes, negative_dict):    
        classes = classes.clone()
        Q_labels = Q_labels.unsqueeze(1).float().clone()
        S_labels = S_labels.float().clone()

        Q_labels = F.interpolate(Q_labels,(Q_feats.shape[2], Q_feats.shape[3]), mode='nearest')
        S_labels = F.interpolate(S_labels,(S_feats.shape[2], S_feats.shape[3]), mode='nearest')

        # zeros = torch.zeros_like(Q_labels).cuda()
        # Q_labels = torch.where(Q_labels==255,zeros,Q_labels)
        # S_labels = torch.where(S_labels==255,zeros,S_labels)

        Q_disrupt_labels = F.relu(1-query_bg_out.max(1)[1].unsqueeze(1) - Q_labels)
        S_disrupt_labels = F.relu(1-supp_bg_out.max(1)[1].unsqueeze(1) - S_labels)

        Q_dsp_fea = Q_feats * Q_disrupt_labels
        S_dsp_fea = S_feats * S_disrupt_labels
        Q_predit_fea = Q_feats * Q_labels
        S_GT_fea = S_feats * S_labels

        # S_dsp_pro = self.generate_maxidx(Q_predit_fea,S_dsp_fea)
        # S_GT_pro = self.generate_maxidx(Q_predit_fea, S_GT_fea)
        # Q_dsp_pro = self.generate_maxidx(Q_predit_fea, Q_dsp_fea)

        # bsize, ch_sz, sp_sz, _ = Q_predit_fea.size()[:]
        # for fea,idx in zip(S_dsp_fea.view(bsize, ch_sz, -1),S_maxdsp_idx):
        #     pro_list.append(fea[:,idx])
        # S_dsp_fea.view(bsize, ch_sz, -1)[S_maxdsp_idx.unsqueeze(-1).unsqueeze(0).repeat(1,ch_sz,1)]
        # Q_dsp_pro = Weighted_GAP(Q_feats, Q_disrupt_labels).squeeze(-1) 
        # S_dsp_pro = Weighted_GAP(S_feats, S_disrupt_labels).squeeze(-1)
        # Q_predit_pro = Weighted_GAP(Q_feats, Q_predit.max(1)[1].unsqueeze(1)).squeeze(-1)
        # S_GT_pro = Weighted_GAP(S_feats, S_labels).squeeze(-1)
        # for fea,lab in zip(Q_feats,Q_labels):
        #     idx = torch.where(lab.flatten(0)==1)[0]
        #     fea = fea.flatten(1)
        #     fea_selec = torch.index_select(fea, dim = -1, index = idx)

        Q_dsp_dict = self._negative_construct(Q_dsp_fea, classes)
        S_dsp_dict = self._negative_construct(S_dsp_fea, classes)

        for key in Q_dsp_dict.keys():
            if key not in negative_dict:
                negative_dict[key] = torch.cat((Q_dsp_dict[key],S_dsp_dict[key]),0).detach()
            else:
                orignal_value = negative_dict[key]
                negative_dict[key] = torch.cat((Q_dsp_dict[key],S_dsp_dict[key],orignal_value),0).detach()
                if negative_dict[key].shape[0]>self.n:
                    negative_dict[key] = negative_dict[key][:self.n,:]

        loss = self._contrastive(Q_predit_fea, S_GT_fea, negative_dict, classes, Q_labels)

        return loss,negative_dict


    def generate_maxsimliarity(self, q, s):
        cosine_eps = 1e-7
        # pro_list = []
        # q = query_feat
        # s = support_feat
        bsize = s.size(0)

        tmp_query = q.repeat(bsize,1,1)
        # tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        # tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
        # tmp_supp = tmp_supp.contiguous().permute(0,2, 1) 
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        # similarity = similarity.mean(2).view(bsize, sp_sz*sp_sz)   
        # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        # # corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
        # idxes = similarity.argmax(-1)
        # for fea,idx in zip(tmp_supp.transpose(-1,-2),idxes):
        #     pro_list.append(fea[:,idx].unsqueeze(0))

        return similarity.max(1)[0]