from datasets.vaw_eval import Evaluator, preprocess_pos_neg
from torch.cuda.amp import autocast
from typing import Iterable
import util.misc as utils
import numpy as np
import torch
import math
import json
import sys

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, crop_samples, mask_samples, targets in metric_logger.log_every(data_loader, print_freq, header):    
        samples = samples.tensors.to(device)
        crop_samples = [crop_sample.to(device) for crop_sample in crop_samples]
        crop_masks = [crop_mask.to(device) for crop_mask in mask_samples]
        targets = [{k: v.to(device) if k != 'obj_names' and type(v) != int else v for k, v in t.items()} for t in targets]
        inputs = [
            {
                "samples": samples,
                "crop_samples": crop_samples,
                "crop_masks": crop_masks
            }
        ]
        with autocast():
            outputs = model(inputs, targets, args)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_att(model, data_loader, device, args=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 10
    model.eval()
    header = 'Test:'
    preds = []
    gts = []
    for samples, crop_samples, crop_masks, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.tensors.to(device)  
        crop_samples = [crop_sample.to(device) for crop_sample in crop_samples]
        crop_masks = [crop_mask.to(device) for crop_mask in crop_masks]        
        inputs = [{"samples": samples, "crop_samples": crop_samples, "crop_masks": crop_masks}]
        targets = [{k: v.to(device) if k != 'obj_names' and type(v) != int else v for k, v in t.items()} for t in targets]
        with autocast():
            outputs = model(inputs, targets, args)['pred_logits']
        outputs = outputs.sigmoid()
        preds.extend(outputs.detach().cpu().numpy())
        gt = preprocess_pos_neg(targets)
        gts.extend(gt)

    metric_logger.synchronize_between_processes()
    preds = torch.cat(utils.all_gather(torch.from_numpy(np.array(preds))))
    annos = torch.cat(utils.all_gather(torch.from_numpy(np.array(gts))))
    evaluator = Evaluator(args.fpath_attribute_index, args.fpath_head_tail)
    scores_per_class = evaluator.evaluate(preds, annos)
    CATEGORIES = ['all', 'head', 'medium', 'tail']
    stats = {f'mAP_{category}': scores_per_class[category]['ap'] for category in CATEGORIES}
    if args.mode == 'zero_shot':
        stats.update(compute_zero_shot_mAP(evaluator, args))

    return stats

def compute_zero_shot_mAP(evaluator, args):
    base_novel_dict = json.load(open(args.base_novel_dict, 'r'))
    base_class = [v for _, v in base_novel_dict['base'].items()]
    novel_class = [v for _, v in base_novel_dict['novel'].items()]    
    base_mAP = sum(evaluator.get_score_class(i_class).ap for i_class in base_class) / len(base_class)
    novel_mAP = sum(evaluator.get_score_class(i_class).ap for i_class in novel_class) / len(novel_class)
    return {'mAP_base': base_mAP, 'mAP_novel': novel_mAP}