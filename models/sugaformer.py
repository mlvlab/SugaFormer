from util.misc import (nested_tensor_from_tensor_list, get_world_size, is_dist_avail_and_initialized)
from .transformer import build_transformer
from .backbone import build_blip_backbone
import torch.nn.functional as F
from torch import nn
import torch
import json
import copy

class SugaFormer(nn.Module):
    def __init__(self, backbone, transformer, num_att_classes, q_former, mask=None, args=None):
        super().__init__()

        self.q_former = q_former
        self.mode = args.mode
        hidden_dim = args.hidden_dim
        image_hidden_dim = args.image_hidden_dim

        self.input_proj = nn.Conv2d(image_hidden_dim, hidden_dim, kernel_size=1)
        self.ctx_input_proj = nn.Conv2d(image_hidden_dim, hidden_dim, kernel_size=1)
        self.m_input_proj = nn.Conv2d(image_hidden_dim, hidden_dim, kernel_size=1)

        projection_dim = 3 
        self.proj_zs_text = nn.Linear(hidden_dim * projection_dim, hidden_dim)
        self.proj_m_zs_text = nn.Linear(hidden_dim * projection_dim, hidden_dim)
        self.proj_ctx_zs_text = nn.Linear(hidden_dim * projection_dim, hidden_dim)

        self.backbone = backbone
        self.transformer = transformer
        self.ctx_transformer = copy.deepcopy(transformer)
        self.m_transformer = copy.deepcopy(transformer)
        self.num_queries = num_att_classes
        self.att_mask = mask

        if args.dataset_file == "vaw":
            if self.mode == "zero_shot":
                base_novel_dict = json.load(open(args.base_novel_dict ,'r'))
                self.seen_index = [v for _,v in base_novel_dict['base'].items()]
                
            self.att_feats = self._load_and_normalize_features(args.att_feats, args.device)
            self.init_super_query_embed = self._load_and_normalize_features(args.sc_feats, args.device)
            
        super_att = json.load(open(args.hierarchy, 'r'))
        self._initialize_groups(super_att, args)

        self._freeze_module(self.backbone, "Visual Backbone")
        self._freeze_module(self.q_former, "Q-Former")

    def _load_and_normalize_features(self, path, device):
        feats = torch.load(path, map_location=torch.device('cpu')).to(device)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats.detach().float()

    def _initialize_groups(self, super_att, args):
        if self.mode == "zero_shot":
            base_novel_dict = json.load(open(args.base_novel_dict, 'r'))
            self.all = base_novel_dict['base'].copy()
            self.all.update(base_novel_dict['novel'])
            self.base_index = base_novel_dict['base'].copy()
            self.base_group = self._create_group(base_novel_dict['base'], super_att)
            self.all_group = self._create_group(self.all, super_att)
            self.att_index = [v for _, v in self.base_group.items()]

        elif self.mode == "supervised":
            base_novel_dict = json.load(open(args.base_novel_dict, 'r'))
            self.all = base_novel_dict['base'].copy()
            self.all.update(base_novel_dict['novel'])
            self.all_group = self._create_group(self.all, super_att)
            self.att_index = [v for _, v in self.all_group.items()]            

    def _create_group(self, attribute_dict, super_att):
        group = {k: [] for k in super_att.keys()}
        for _, v in attribute_dict.items():
            for super_class, att_list in super_att.items():
                if v in att_list:
                    group[super_class].append(v)
        return group

    def _freeze_module(self, module, module_name):
        print(f"Freeze {module_name}")
        for p in module.parameters():
            p.requires_grad = False

    def create_nested_tensor(self, images):
        return nested_tensor_from_tensor_list([img for sublist in images for img in sublist])

    def multi_context_decoding(self, sc_inputs, inputs, targets):
        samples = inputs[0]['samples']
        crop_samples = inputs[0]['crop_samples']
        crop_masks = torch.cat(inputs[0]['crop_masks'], dim=0)

        crop_imgs = self.create_nested_tensor(crop_samples)
        whole_imgs = nested_tensor_from_tensor_list(samples)

        crop_src, crop_pos = self.backbone(crop_imgs)[0][-1].tensors, self.backbone(crop_imgs)[1]
        whole_src, whole_pos = self.backbone(whole_imgs)[0][-1].tensors, self.backbone(whole_imgs)[1]

        interpolated_masks = F.interpolate(
            crop_masks.clone(),
            size=crop_src.shape[-2:],
            mode='bicubic',
            align_corners=False
        )

        masked_src = crop_src.clone() * interpolated_masks.repeat(1, crop_src.shape[1], 1, 1)

        indices = [i for i, target in enumerate(targets) for _ in target['boxes']]
        whole_src = torch.cat([whole_src[idx].unsqueeze(0) for idx in indices], dim=0)
        whole_pos = torch.cat([whole_pos[0][idx].unsqueeze(0) for idx in indices], dim=0)

        crop_hs = self.transformer(self.input_proj(crop_src), sc_inputs['c_sq'], crop_pos[-1])[0]
        mask_hs = self.m_transformer(self.m_input_proj(masked_src), sc_inputs['m_sq'], crop_pos[-1])[0]
        whole_hs = self.ctx_transformer(self.ctx_input_proj(whole_src), sc_inputs['w_sq'], whole_pos)[0]

        hs = {
            'c_hs': crop_hs,
            'm_hs': mask_hs,
            'w_hs': whole_hs
        }

        return hs

    def super_class_query_init(self, inputs, targets):
        att_init = self.init_super_query_embed.detach().float()
        crop_samples = torch.cat(inputs[0]['crop_samples'], dim=0)
        att_init = att_init.unsqueeze(0).repeat(crop_samples.shape[0], 1, 1)

        vis_queries = self.q_former.extract_features({"image": crop_samples}, mode="image").image_embeds_proj
        crop_mean_query = vis_queries.mean(dim=1).unsqueeze(1).repeat(1, att_init.shape[1], 1)

        obj_name_list = [obj for target in targets for obj in target['obj_names']]
        obj_max_queries = self._compute_obj_max_queries(vis_queries, obj_name_list, att_init.shape[1])

        concat_features = torch.cat([obj_max_queries, crop_mean_query, att_init], dim=-1)
        c_sq = self.proj_zs_text(concat_features)
        m_sq = self.proj_m_zs_text(concat_features)
        w_sq = self.proj_ctx_zs_text(concat_features)

        return {"c_sq": c_sq, "m_sq": m_sq, "w_sq": w_sq}


    def _compute_obj_max_queries(self, vis_queries, obj_name_list, num_queries):
        obj_max_queries = []
        for vis_query, obj_name in zip(vis_queries, obj_name_list):
            object_sample = {"text_input": obj_name}
            obj_text = self.q_former.extract_features(object_sample, mode="text").text_embeds_proj[:, 0, :]
            sim = vis_query @ obj_text.T
            max_query = vis_query[(sim == sim.max()).nonzero(as_tuple=True)[0][0]].unsqueeze(0)
            obj_max_queries.append(max_query)
        obj_max_queries = torch.cat(obj_max_queries, dim=0)
        return obj_max_queries.unsqueeze(1).repeat(1, num_queries, 1)

    def compute_zrse_scores(self, cropped_samples, att_feats, args):
        cropped_imgs = nested_tensor_from_tensor_list([cropped_img for cropped_imgs in cropped_samples for cropped_img in cropped_imgs])                    
        crop_input = {"image": cropped_imgs.tensors}
        features_image = self.q_former.extract_features(crop_input, mode="image").image_embeds_proj
        sim_scores = (features_image @ att_feats.t())
        zero_shot_topk = []
        for sim_score in sim_scores:
            score_init = torch.zeros_like(sim_score[0])
            max_query = sim_score[(sim_score.max() == sim_score).nonzero()[0][0]]
            score_init[max_query.topk(args.ztopk).indices] = max_query.topk(args.ztopk).values
            zero_shot_topk.append(score_init.unsqueeze(0))        
        zero_shot_topk = torch.cat(zero_shot_topk, dim=0)
        return zero_shot_topk

    def compute_class_outputs(self, hs, group, att_feats):
        c_hs, m_hs, w_hs = hs['c_hs'], hs['m_hs'], hs['w_hs']

        logits = {'crop': [], 'mask': [], 'whole': []}
        for i, (_, att_indices) in enumerate(group.items()):
            logits['crop'].append(c_hs[:, :, i] @ att_feats[att_indices].T)
            logits['mask'].append(m_hs[:, :, i] @ att_feats[att_indices].T)
            logits['whole'].append(w_hs[:, :, i] @ att_feats[att_indices].T)

        logits['crop'] = torch.cat(logits['crop'], dim=-1)
        logits['mask'] = torch.cat(logits['mask'], dim=-1)
        logits['whole'] = torch.cat(logits['whole'], dim=-1)

        att_index = [idx for _, indices in group.items() for idx in indices]
        full_size = logits['crop'].shape[:-1] + (att_feats.shape[0],)

        def map_logits(logits_tmp):
            logits_full = torch.zeros(full_size, device=logits_tmp.device, dtype=logits_tmp.dtype)
            logits_full[:, :, att_index] = logits_tmp
            return logits_full.unsqueeze(-1)

        outputs_class = map_logits(logits['crop'])[:,:,self.seen_index,:] if self.mode == 'zero_shot' and self.training else map_logits(logits['crop'])
        m_outputs_class = map_logits(logits['mask'])[:,:,self.seen_index,:] if self.mode == 'zero_shot' and self.training else map_logits(logits['mask'])
        w_outputs_class = map_logits(logits['whole'])[:,:,self.seen_index,:] if self.mode == 'zero_shot' and self.training else map_logits(logits['whole'])

        return outputs_class, m_outputs_class, w_outputs_class

    def predict(self, hs, targets):

        group = self.base_group if (self.mode == 'zero_shot' and self.training) else self.all_group
        outputs_class, m_outputs_class, w_outputs_class = self.compute_class_outputs(hs, group, self.att_feats)

        out = {
            'pred_logits': outputs_class[-1].squeeze(2),
            'pred_m_logits': m_outputs_class[-1].squeeze(2),
            'pred_w_logits': w_outputs_class[-1].squeeze(2),
        }

        if self.training:
            sc_token_outputs = torch.cat([target['sc_token_output'] for target in targets], dim=0)
            sc_pred_outputs = (hs['c_hs'][-1] + hs['m_hs'][-1] + hs['w_hs'][-1]) / 3
            out.update({
                'sc_pred_outputs': sc_pred_outputs,
                'sc_tokens': self.q_former.text_proj(sc_token_outputs),
            })
        else:
            out['pred_logits'] = (out['pred_logits'] + out['pred_m_logits'] + out['pred_w_logits']) / 3

        return out

    def forward(self, inputs, targets, args):
        sc_inputs = self.super_class_query_init(inputs, targets)
        hs = self.multi_context_decoding(sc_inputs, inputs, targets)
        outputs = self.predict(hs, targets)
        if args.zrse and not self.training:
            cropped_samples = inputs[0]['crop_samples']
            zero_shot_topk = self.compute_zrse_scores(cropped_samples, self.att_feats, args)
            outputs['pred_logits'] += args.zrse_scale * zero_shot_topk
        return outputs

class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, att_class_weight, args=None):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.mode = args.mode
        if self.mode == 'zero_shot':
            base_novel_dict = json.load(open(args.base_novel_dict ,'r'))
            self.seen_index = [v for _,v in base_novel_dict['base'].copy().items()]

        self.asymloss = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=0, clip=args.clip, disable_torch_grad_focal_loss=True, att_class_weight=att_class_weight, args=args)

    def loss_attributes(self, outputs, targets):
        src_logits = outputs['pred_logits']
        m_src_logits = outputs['pred_m_logits']
        ctx_src_logits = outputs['pred_w_logits']
        if self.mode == 'zero_shot':
            target_classes = torch.cat([t['pos_att_classes'] for t in targets])[:, self.seen_index]
            target_m_classes = target_classes.clone()
            target_ctx_classes = target_classes.clone()
        else:
            target_classes = torch.cat([t['pos_att_classes'] for t in targets])
            target_m_classes = target_classes.clone()
            target_ctx_classes = target_classes.clone()        
        loss_att_ce = self.asymloss(src_logits, target_classes)
        loss_m_att_ce = self.asymloss(m_src_logits, target_m_classes)
        loss_ctx_att_ce = self.asymloss(ctx_src_logits, target_ctx_classes)
        loss_all_ce = loss_ctx_att_ce + loss_m_att_ce + loss_att_ce
        losses = {'loss_att_ce': loss_all_ce} 
        return losses

    def loss_scr(self, outputs, targets=None, sc_num=7):
        sc_tokens = outputs['sc_tokens']
        sc_pred_outputs = outputs['sc_pred_outputs']
        loss_scr = F.l1_loss(sc_pred_outputs[:,:sc_num,:], sc_tokens[:,:sc_num,:])
        losses = {'loss_scr': loss_scr}   
        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'att_labels':self.loss_attributes,
            'scr':self.loss_scr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        return losses

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, att_class_weight=None,args=None):
        super(AsymmetricLoss, self).__init__()
        if args.mode == 'zero_shot':
            base_novel_dict = json.load(open(args.base_novel_dict ,'r'))
            self.seen_index = [v for _,v in base_novel_dict['base'].items()]

        self.att_class_weight = att_class_weight[self.seen_index] if args.mode == "zero_shot" else att_class_weight
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = (2-y) * y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = ((2-y)/2) * (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        loss *=  self.att_class_weight

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            pt0 = xs_pos * y * (2-y)
            pt1 = xs_neg * (1 - y) * ((2-y)/2)    
            pt = pt0 + pt1            
            one_sided_gamma = self.gamma_pos * y * (2 - y) + self.gamma_neg * (1 - y) * ((2 - y)/2)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

def build(args):
    if args.dataset_file == "vaw":
        att_class_weight = torch.load(args.att_class_weight).to(args.device)        
        num_att_classes = args.num_att_classes
    device = torch.device(args.device)
    backbone, q_former = build_blip_backbone(args)
    transformer = build_transformer(args)   
    model = SugaFormer(
            backbone,
            transformer,
            num_att_classes=num_att_classes,
            q_former = q_former,
            args=args
            )
    weight_dict = {}
    losses = ['att_labels','scr']
    weight_dict.update({'loss_att_ce':args.att_loss_coef})
    weight_dict.update({'loss_scr':args.scr_coef})
    criterion = SetCriterion(weight_dict=weight_dict, losses=losses, att_class_weight=att_class_weight, args=args)
    criterion.to(device)
    return model, criterion