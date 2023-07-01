#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import libs.models.Net.resnet as models
from einops import rearrange
from libs.models.prototypical_contrast_withQGT import PrototypeContrastLoss
from libs.models.maxprototypical_contrast_pixel2pixel_paddn import MAXPrototypeContrastLoss
class Self_Attention(nn.Module):
    # Not using location
    def __init__(self, indim, keydim):
        super(Self_Attention, self).__init__()
        self.query = nn.Conv2d(indim, keydim,kernel_size=1, stride=1)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=1, stride=1)
        self.Value = nn.Conv2d(indim, indim, kernel_size=1, stride=1)
        self.query2 = nn.Conv2d(indim, keydim,kernel_size=1,stride=1)
        self.Key2 = nn.Conv2d(indim, keydim, kernel_size=1, stride=1)
        self.Value2 = nn.Conv2d(indim, indim, kernel_size=1, stride=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(indim, indim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(indim, indim, kernel_size=1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(indim, indim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(indim, indim, kernel_size=1))
    def forward(self, x,batch,frame):

        _,c,h,w = x.size()
        middle_frame = frame//2 
        x_mframe = rearrange(x,'(b f) c h w -> b f c h w ',b = batch, f = frame)[:,middle_frame]
        q_mframe = rearrange(self.query(x_mframe),'b c h w -> b c (h w) ')
        k = rearrange(self.Key(x),'(b f) c h w -> b c (f h w)',b = batch, f = frame)
        v = rearrange(self.Value(x),'(b f) c h w -> b (f h w) c',b = batch, f = frame)
        attn1 = (torch.einsum('bck,bcn->bkn', q_mframe, k)/math.sqrt(c)).softmax(-1)
        out = rearrange(torch.einsum('bkn,bnc->bkc', attn1, v),'b (h w) c -> b c h w',h=h,w=w)
        out = x_mframe + out
        out = out + self.mlp(out)

        q2 = rearrange(self.query2(x),'(b f) c h w -> b (f h w) c',b = batch, f = frame)
        k2 = rearrange(self.Key2(out),'b c h w -> b c (h w)')
        v2 = rearrange(self.Value2(out),'b c h w -> b (h w) c')
        attn2 = (torch.einsum('bkc,bcn->bkn', q2, k2)/math.sqrt(c)).softmax(-1)
        out2 = torch.einsum('bkn,bnc->bkc', attn2, v2)
        out2 = rearrange(out2,'b (f h w) c -> (b f) c h w',b = batch, f = frame, h=h, w=w)
        out2 = x + out2
        out2 = out2 + self.mlp2(out2)
        return out2




def Weighted_GAP(supp_feat, mask):
    mask = F.interpolate(mask, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)

    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)

        self.layer1 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024 --> 1/8
        self.layer4 = resnet.layer4  # 1/16, 1024 --> 1/8

        
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, in_f, mask=None):

        f = in_f
        x = self.layer0(f)
        l1 = self.layer1(x)  # 1/4, 256
        l2 = self.layer2(l1)  # 1/8, 512
        l3 = self.layer3(l2)  # 1/8, 1024
        if mask is not None:
            mask = F.interpolate(mask, size=(l3.size(2), l3.size(3)), mode='bilinear', align_corners=True)
            l4 = self.layer4(l3*mask)            
        else:
            l4 = self.layer4(l3) 

        return l4, l3, l2, l1


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = Encoder() # output 2048
        encoder_dim = 1024

        self.pyramid_bins = [60, 30, 15, 8]
        fea_dim = 1024 + 512 
        reduce_dim = 256
        bg_dim = 512
        classes = 2
        mask_add_num = 1
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )  

        self.attn3 = Self_Attention(reduce_dim,reduce_dim//2)


        self.init_merge1 = []
        self.init_merge2 = []
        self.init_merge3 = []
        self.init_merge4 = []
        self.attn1 = []
        self.attn2 = []

        self.inital_beta_conv = []
        self.inital_inner_cls = []        

        self.second_beta_conv = []
        self.second_inner_cls = []

        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge1.append(nn.Sequential(
                nn.Conv2d(reduce_dim+bg_dim , reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))   
            self.init_merge2.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                     
            self.init_merge3.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 , reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.init_merge4.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                     
            self.inital_beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inital_inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
            self.inital_inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))  

            self.attn1.append(Self_Attention(reduce_dim,reduce_dim//2)
            )      
            self.attn2.append(Self_Attention(reduce_dim,reduce_dim//2)
            )    
        
            self.second_beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))   
            self.second_inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            

            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge1 = nn.ModuleList(self.init_merge1) 
        self.init_merge2 = nn.ModuleList(self.init_merge2) 
        self.init_merge3 = nn.ModuleList(self.init_merge3) 
        self.init_merge4 = nn.ModuleList(self.init_merge4) 
        self.attn1 = nn.ModuleList(self.attn1) 
        self.attn2 = nn.ModuleList(self.attn2) 
        # self.attn3 = nn.ModuleList(self.attn3) 

        self.inital_beta_conv = nn.ModuleList(self.inital_beta_conv)
        self.inital_inner_cls = nn.ModuleList(self.inital_inner_cls)
        self.second_beta_conv = nn.ModuleList(self.second_beta_conv)
        self.second_inner_cls = nn.ModuleList(self.second_inner_cls)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)     

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        self.bg_prototype = nn.Parameter(torch.zeros(1, bg_dim,1,1))
        self.bg_cirloss = nn.CrossEntropyLoss(reduction='none')
        self.bg_res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        ) 
        self.down_bg = nn.Sequential(
            nn.Conv2d(reduce_dim+bg_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        self.bg_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )   

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )  
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, 1, kernel_size=1)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.contrast_loss = PrototypeContrastLoss()
        self.maxcontrast_loss = MAXPrototypeContrastLoss()
    def get_optimizer(self, args):
        optimizer = torch.optim.SGD(
        [
            ##### background prototype ####
            {'params': self.bg_prototype},
            {'params': self.down_bg.parameters()},
            {'params': self.bg_res1.parameters()},
            {'params': self.bg_cls.parameters()},
            ###############################

            {'params': self.down_query.parameters()},
            {'params': self.down_supp.parameters()},

            {'params': self.init_merge1.parameters()},
            {'params': self.init_merge2.parameters()},
            {'params': self.init_merge3.parameters()},
            {'params': self.init_merge4.parameters()},

            # {'params': self.attn1.parameters()},

            {'params': self.inital_beta_conv.parameters()},
            {'params': self.inital_inner_cls.parameters()},
            {'params': self.second_beta_conv.parameters()},
            {'params': self.second_inner_cls.parameters()},

            {'params': self.alpha_conv.parameters()},
            {'params': self.beta_conv.parameters()},
            {'params': self.inner_cls.parameters()},

            {'params': self.res1.parameters()},
            {'params': self.res2.parameters()},
            {'params': self.cls.parameters()},
        ],
        lr=args.lr, momentum=0.9, weight_decay=0.0001)

        trans_optimizer = torch.optim.Adam(
        [
            {'params': self.attn1.parameters()},
            {'params': self.attn2.parameters()},
            {'params': self.attn3.parameters()}

        ],
        lr=1e-5, betas=(0.9, 0.999), weight_decay=5e-4)
        return optimizer,trans_optimizer

    def forward(self, img,query_mask, support_image, support_mask,classes=None,time=None,prototype_neg_dict=None, max_prototype_neg_dict=None):

        ## img,support_image: 4,5,3,241,425; support_mask:4,5,1,241,425

        batch, frame, in_channels, height, width = img.shape
        _, sframe, mask_channels, Sheight,Swidth = support_mask.shape
        assert  height == Sheight
        assert width == Swidth
        batch_frame = batch*frame
        img = img.view(-1, in_channels, height, width)
        support_image = support_image.view(-1, in_channels, height, width)
        support_mask = support_mask.view(-1, mask_channels, height, width)

        
        with torch.no_grad():
            query_feat_4, query_feat_3,query_feat_2,query_feat_1 = self.encoder(img)
            supp_feat_4, supp_feat_3, supp_feat_2,_ = self.encoder(support_image,support_mask)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature     

        # for i in range(self.shot):

        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat_nomask = self.down_supp(supp_feat)
        supp_feat = Weighted_GAP(supp_feat_nomask, support_mask)


        corr_query_mask = self.generate_prior(query_feat_4, supp_feat_4, support_mask, supp_feat_3.size(-1))

        bg = self.bg_prototype.expand(query_feat.size(0),-1,query_feat.size(2),query_feat.size(3))

    
        qrybg_feat = torch.cat((query_feat,bg),dim=1)
        qrybg_feat1 = self.down_bg(qrybg_feat)
        qrybg_feat1 = self.attn3(qrybg_feat1,batch,frame)
        qrybg_feat2 = self.bg_res1(qrybg_feat1) + qrybg_feat1         
        query_bg_out = self.bg_cls(qrybg_feat2)

        if self.training:        

            # for supp_feat_nomask in supp_nomask_feat_list:
            suppbg_feat = torch.cat((supp_feat_nomask,bg),dim=1)
            suppbg_feat = self.down_bg(suppbg_feat)
            suppbg_feat = self.attn3(suppbg_feat,batch,frame)
            suppbg_feat = self.bg_res1(suppbg_feat) + suppbg_feat          
            supp_bg_out = self.bg_cls(suppbg_feat)
                # supp_bg_out_list.append(supp_bg_out)        
        # img flow : [batch, frame, channels, height, width] -> [batch*frame, channels, height, width]
        # support_mask : [batch, sframe, mask_channels, height, width]
        inital_out_list = []
        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)

            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            bg_feat_bin = self.bg_prototype.expand(query_feat.size(0),-1,bin,bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_binbg = torch.cat([query_feat_bin, bg_feat_bin],1)
            merge_feat_binbg = self.init_merge1[idx](merge_feat_binbg)
            merge_feat_binbg = self.attn1[idx](merge_feat_binbg,batch,frame)
            merge_feat_binfg = torch.cat([merge_feat_binbg,supp_feat_bin, corr_mask_bin],1)
            merge_feat_binfg = self.init_merge2[idx](merge_feat_binfg)

            merge_feat_binfg = self.inital_beta_conv[idx](merge_feat_binfg) + merge_feat_binfg   
            inital_inner_out_bin = self.inital_inner_cls[idx](merge_feat_binfg)
            inital_out_list.append(inital_inner_out_bin)

            fg_pro = nn.AdaptiveAvgPool2d(1)(inital_inner_out_bin.max(1)[1].unsqueeze(1)*query_feat_bin)
            merge_feat_binfg2 = torch.cat([merge_feat_binfg,fg_pro.expand_as(merge_feat_binbg), corr_mask_bin],1)
            
            merge_feat_binfg2 = self.init_merge4[idx](merge_feat_binfg2)
            merge_feat_binfg2 = self.attn2[idx](merge_feat_binfg2,batch,frame)
            merge_feat_binfg2 = self.second_beta_conv[idx](merge_feat_binfg2) + merge_feat_binfg2   
            second_inner_out_bin = self.second_inner_cls[idx](merge_feat_binfg2)
            inital_out_list.append(second_inner_out_bin)

            query_bg_out_bin = F.interpolate(query_bg_out, size=(bin, bin), mode='bilinear', align_corners=True)
            confused_mask = F.relu(1- query_bg_out_bin.max(1)[1].unsqueeze(1) -  second_inner_out_bin.max(1)[1].unsqueeze(1)) 
            confused_prototype = nn.AdaptiveAvgPool2d(1)(confused_mask*query_feat_bin)
            confused_prototype_bin = confused_prototype.expand(-1,-1,bin,bin)
            merge_feat_bin = torch.cat([merge_feat_binfg2,confused_prototype_bin],1)
            merge_feat_bin = self.init_merge3[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)

            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat_out = torch.cat(pyramid_feat_list, 1)
        query_feat_out = self.res1(query_feat_out)
        query_feat_out = self.res2(query_feat_out) + query_feat_out           
        out = self.cls(query_feat_out)            

        # aspp
        # x = self.aspp(after_transform)
        if time is not None:
            time.t2()
        # x = self.Decoder(after_transform, query_feat_2, query_feat_1, img)
        out_t = F.interpolate(out, size=(height, width), mode='bilinear', align_corners=True).sigmoid()
        # out = out.softmax(1)[:,1,:,:]
            
        # pred_map = torch.nn.Sigmoid()(x)

        # batch, frame, outchannel, height, width
        out = out_t.view(batch, frame, 1, height, width)
        if self.training:

            aux_loss1 = torch.zeros(1).cuda()    
            aux_loss2 = torch.zeros(1).cuda()  
            query_mask = query_mask.view(-1, height, width)
            # classes = classes.contiguous().view(batch * frame)
            h,w = height, width

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss1 = aux_loss1 + self.criterion(inner_out, query_mask.long())   
            aux_loss1 = aux_loss1 / len(out_list)

            for idx in range(len(inital_out_list)):    
                inner_out = inital_out_list[idx]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss2 = aux_loss2 + self.criterion(inner_out, query_mask.long())   
            aux_loss2 = aux_loss2 / len(inital_out_list)


            prototype_contrast_loss, prototype_neg_dict = self.contrast_loss(query_feat, supp_feat_nomask, out_t, \
                query_mask, support_mask, query_bg_out, supp_bg_out, classes, prototype_neg_dict)
            
            if max_prototype_neg_dict is not None:
                c_s= support_mask.contiguous().view(batch * frame,-1).sum(-1)
                c_q = query_mask.contiguous().view(batch * frame,-1).sum(-1)
                cc = ((c_s>1000).float() + (c_q>1000).float())==2
                maxprototype_contrast_loss, max_prototype_neg_dict = self.maxcontrast_loss(query_feat[cc], supp_feat_nomask[cc], out_t[cc].detach(), \
                    query_mask[cc], support_mask[cc], query_bg_out[cc], supp_bg_out[cc], classes[cc], max_prototype_neg_dict)
    
                prototype_contrast_loss = (prototype_contrast_loss+maxprototype_contrast_loss)/2

            return out,query_bg_out,supp_bg_out,0.4*aux_loss1+0.6*aux_loss2
        else:
            return out

    def bg_loss(self,query_bg_out,supp_bg_out,y,s_y):
        batch, frame, height, width = y.shape
        y = y.contiguous().view(-1,height, width)
        s_y = s_y.contiguous().view(-1,height, width)
        query_bg_out = F.interpolate(query_bg_out, size=(height, width), mode='bilinear', align_corners=True)
        supp_bg_out = F.interpolate(supp_bg_out, size=(height, width), mode='bilinear', align_corners=True)

        mygt1 = torch.ones_like(y).cuda()
        mygt0 = torch.zeros_like(y).cuda()

        query_bg_loss = self.weighted_BCE(query_bg_out, mygt0, y)+0.5*self.criterion(query_bg_out,mygt1.long())
        supp_bg_loss = self.weighted_BCE(supp_bg_out, mygt0, s_y)+0.5*self.criterion(supp_bg_out,mygt1.long())
        bg_loss = (query_bg_loss + supp_bg_loss)/ 2
        return bg_loss

    def weighted_BCE(self,input, target,mask):
        loss_list =[]
        cmask = torch.where(mask.long() == 1,mask.long(),target.long())
        
        for x,y,z in zip(input,target,cmask):
            loss = self.bg_cirloss(x.unsqueeze(0),y.unsqueeze(0).long())
            area = torch.sum(z)+1e-5
            Loss = torch.sum(z.unsqueeze(0)*loss) /area
            loss_list.append(Loss.unsqueeze(0))
        LOSS = torch.cat(loss_list,dim=0)                     
        return torch.mean(LOSS)

    def generate_prior(self, query_feat_4, tmp_supp_feat, mask_list, fts_size):
        # corr_query_mask_list = []
        cosine_eps = 1e-7
        # for i, tmp_supp_feat in enumerate(final_supp_list):
        resize_size = tmp_supp_feat.size()[2:]
        tmp_mask = F.interpolate(mask_list, size=(resize_size[0], resize_size[1]), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
        q = query_feat_4
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, sh_sz = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

        tmp_supp = s               
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].view(bsize, sp_sz*sh_sz)   
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.view(bsize, 1, sp_sz, sh_sz)
        corr_query = F.interpolate(corr_query, size=(sp_sz, sh_sz), mode='bilinear', align_corners=True)


        return corr_query

