r""" Visualize model predictions """
import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
from libs.utils.davis_JF import db_eval_boundary, db_eval_iou
def to_cpu(tensor):
    return tensor.detach().clone().cpu()

class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255),'one':(195,74,52),'two':(0,143,122)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = './vis4/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, out_bg, pred_map_o, cls_id_b, batch_idx, iou_b=None):#  mix_pro1,mix_pro2,mix_pro3,mix_pro4,mix_pro5=None
        # spt_img_b = F.interpolate(spt_img_b, size=qry_mask_b.size()[1:], mode='bilinear', align_corners=True)
        # spt_mask_b = F.interpolate(spt_mask_b.squeeze(0).float(), size=qry_mask_b.size()[1:], mode='nearest')
        # qry_img_b = F.interpolate(qry_img_b, size=qry_mask_b.size()[1:], mode='bilinear')
        spt_img_b = to_cpu(spt_img_b)
        spt_mask_b = to_cpu(spt_mask_b)
        qry_img_b = to_cpu(qry_img_b)
        qry_mask_b = to_cpu(qry_mask_b)
        out_bg = to_cpu(out_bg)
        pred_map_o = to_cpu(pred_map_o)
        # out_do = utils.to_cpu(out_do.squeeze(0).squeeze(0))
        # out_ini = utils.to_cpu(out_ini.squeeze(0))
        cls_id_b = to_cpu(cls_id_b[0])

        # qry_img_pill = Image.fromarray(cls.to_numpy(qry_img_b,'img'))
        # out_bg_pill = cls.to_numpy(out_bg,'mask')
        # out_do_pill = cls.to_numpy(out_do,'mask')
        # out_ini_pill = cls.to_numpy(out_ini,'mask')

        # merged_pil = cls.merge_image_pair([qry_img_pill,out_bg_pill,out_do_pill,out_ini_pill])
        # merged_pil.save(cls.vis_path + '1_%d_class-%d_iou-%.2f' % (batch_idx, cls_id_b, iou_b) + '.jpg')

        for k, data in enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, out_bg, pred_map_o)):
            spt_img, spt_mask, qry_img, qry_mask, out, pred_map = data
            iou = db_eval_iou(qry_mask.numpy(),out.numpy())
            iou_o = db_eval_iou(qry_mask.numpy(),pred_map.numpy())
            if (iou-iou_o)>0.1:
                qry_pill = Image.fromarray(cls.apply_mask(cls.to_numpy(qry_img,'img').astype(np.uint8), cls.to_numpy(qry_mask,'mask').astype(np.uint8), cls.colors['red']))
                pred_map_pill = Image.fromarray(cls.apply_mask(cls.to_numpy(qry_img,'img').astype(np.uint8), cls.to_numpy(pred_map,'mask').astype(np.uint8), cls.colors['red']))
                out_pill = Image.fromarray(cls.apply_mask(cls.to_numpy(qry_img,'img').astype(np.uint8), cls.to_numpy(out,'mask').astype(np.uint8), cls.colors['red']))
            # out_do_pill = Image.fromarray(cls.apply_mask(cls.to_numpy(qry_img_b,'img').astype(np.uint8), cls.to_numpy(out_do,'mask').astype(np.uint8), cls.colors['red']))
                sup_pill = Image.fromarray(cls.apply_mask(cls.to_numpy(spt_img,'img').astype(np.uint8), cls.to_numpy(spt_mask,'mask').astype(np.uint8), cls.colors['blue']))
            # merged_pil = cls.merge_image_pair([out_bg_pill,out_do_pill,out_ini_pill])
                qry_pill.save(cls.vis_path + 'class-%d_k_%d_%d' % (cls_id_b,k, batch_idx) + 'qry.jpg')
                pred_map_pill.save(cls.vis_path + 'class-%d_k_%d_%d' % (cls_id_b,k, batch_idx) + 'pred_map.jpg')
                out_pill.save(cls.vis_path + 'class-%d_k_%d_%d' % (cls_id_b,k, batch_idx) + 'out.jpg')
                sup_pill.save(cls.vis_path + 'class-%d_k_%d_%d' % (cls_id_b,k, batch_idx) + 'sup.jpg')



        # qry_img_b = cls.to_numpy(qry_img_b, 'img')
        # Qfea = Qfea.squeeze(0).squeeze(0).detach().cpu().numpy()
        # Qbg = Qbg.squeeze(0).squeeze(0).detach().cpu().numpy()
        # Dofea = Dofea.squeeze(0).squeeze(0).detach().cpu().numpy()

        # weightmap = (Qfea-np.min(Qfea))/(np.max(Qfea)-np.min(Qfea))
        # weightmap = np.uint8(255 * weightmap) 
        # heatmap = cv2.applyColorMap(weightmap, cv2.COLORMAP_JET)
        # heat_img = cv2.addWeighted(qry_img_b, 0.3, heatmap, 0.6, 0)
        # cv2.imwrite(cls.vis_path + 'pic%d_class-%d_iou-%.2f_Qfea' % (batch_idx,cls_id_b, iou_b) + '_.jpg', heat_img) 

        # weightmap = (Qbg-np.min(Qbg))/(np.max(Qbg)-np.min(Qbg))
        # weightmap = np.uint8(255 * weightmap) 
        # heatmap = cv2.applyColorMap(weightmap, cv2.COLORMAP_JET)
        # heat_img = cv2.addWeighted(qry_img_b, 0.3, heatmap, 0.6, 0)
        # cv2.imwrite(cls.vis_path + 'pic%d_class-%d_iou-%.2f_Qbg' % (batch_idx,cls_id_b, iou_b) + '_.jpg', heat_img) 

        # weightmap = (Dofea-np.min(Dofea))/(np.max(Dofea)-np.min(Dofea))
        # weightmap = np.uint8(255 * weightmap) 
        # heatmap = cv2.applyColorMap(weightmap, cv2.COLORMAP_JET)
        # heat_img = cv2.addWeighted(qry_img_b, 0.3, heatmap, 0.6, 0)
        # cv2.imwrite(cls.vis_path + 'pic%d_class-%d_iou-%.2f_Dofea' % (batch_idx,cls_id_b, iou_b) + '_.jpg', heat_img) 


        # qry_img = cls.to_numpy(qry_img_b, 'img')
        # qry_mask = cls.to_numpy(qry_mask_b, 'mask')
        # diff_mask = F.relu(pred_mask_b - qry_mask_b)
        # diff_mask = cls.to_numpy(diff_mask, 'mask')

        # heat_img = cls.apply_mask2(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), cls.colors['two'],diff_mask.astype(np.uint8), cls.colors['red'])
        # heat_img_pill = Image.fromarray(heat_img)

        # # merged_pil = cls.merge_image_pair(heat_img_pill)
        # heat_img_pill.save(cls.vis_path + '%d_class-%d_iou-%.2f' % (batch_idx, cls_id_b, iou_b) + '.jpg')

        # mix_pro1_b = utils.to_cpu(mix_pro1.squeeze(0))
        # mix_pro2_b = utils.to_cpu(mix_pro2.squeeze(0).squeeze(0))
        # mix_pro3_b = utils.to_cpu(mix_pro3.squeeze(0).squeeze(0))
        # mix_pro4_b = utils.to_cpu(mix_pro4.squeeze(0).squeeze(0))
        # mix_pro5_b = utils.to_cpu(mix_pro5.squeeze(0).squeeze(0))
        # mix_pro6_b = None
        # mix_pro7_b = utils.to_cpu(mix_pro7.squeeze(0).squeeze(0))

        # for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
        #         enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
        #     iou = iou_b[sample_idx] if iou_b is not None else None
        # cls.visualize_prediction(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx,iou_b)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx,iou=None):

        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]

        qry_img = cls.to_numpy(qry_img, 'img')
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        # Sbg = cls.to_numpy(Sbg, 'mask')
        # Qbg = cls.to_numpy(Qbg, 'mask')
        # init_out = cls.to_numpy(init_out, 'mask')
        # mix_pro4 = cls.to_numpy(mix_pro4, 'mask')
        # mix_pro5 = cls.to_numpy(mix_pro5,'mask')
        # mix_pro6 = cls.to_numpy(mix_pro6.permute(1,2,0), 'mask')
        # mix_pro7 = cls.to_numpy(mix_pro7, 'mask')

        # mix_pro1_pill = Image.fromarray(cls.apply_heatmap(qry_img.astype(np.uint8), mix_pro1.astype(np.uint8), pred_color))
        # Sbg = Image.fromarray(cls.apply_mask(spt_imgs[0].astype(np.uint8), Sbg.astype(np.uint8), pred_color))
        # Qbg = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), Qbg.astype(np.uint8), pred_color))
        # init_out = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), init_out.astype(np.uint8), pred_color))
        # mix_pro5_pill = Image.fromarray(cls.apply_mask(spt_imgs[0].astype(np.uint8), mix_pro5.astype(np.uint8), pred_color))
        # mix_pro6_pill = Image.fromarray(cls.apply_heatmap(qry_img.astype(np.uint8), mix_pro6.astype(np.uint8), pred_color))
        # mix_pro7_pill = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), mix_pro7.astype(np.uint8), pred_color))

        merged_pil = cls.merge_image_pair(spt_masked_pils +[ qry_masked_pil,pred_masked_pil])
        # merged_pil = cls.merge_image_pair(spt_masked_pils + [pred_masked_pil, qry_masked_pil])
        iou = iou.item() if iou else 0.0
        merged_pil.save(cls.vis_path + '%d_class-%d_iou-%.2f' % (batch_idx, cls_id, iou) + '.jpg')

    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])

        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_heatmap(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        return heatmap

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply heatmap to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def apply_mask2(cls, image, mask1, color1,mask2, color2, alpha=0.5,beta=0.7):
        r""" Apply heatmap to the given image. """
        for c in range(3):
            image[:, :, c] = np.where((mask2 == 1)&(mask1 == 0),
                                      image[:, :, c] *
                                      (1 - beta) + beta * color1[c] * 255,
                                      image[:, :, c])
        for c in range(3):
            image[:, :, c] = np.where((mask1 == 1)&(mask2 == 0),
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color2[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
