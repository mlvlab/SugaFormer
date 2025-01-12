from torch.utils.data import DataLoader, DistributedSampler
from engine import evaluate_att, train_one_epoch
from datasets import build_dataset
from models import build_model
import util.misc as utils
from pathlib import Path
import numpy as np
import argparse
import datetime
import random
import json
import time
import torch
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Set SugaFormer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=8, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--gamma_neg', default=4, type=int, help="gamma_neg for Assymloss")
    parser.add_argument('--clip', default=0.05, type=float)
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--freeze_backbone', action='store_true', help="freezing backbone flag")

    parser.add_argument('--dataset_file', default='', type=str)
    parser.add_argument('--vaw_path', default='data/vaw', type=str)
    parser.add_argument('--output_dir', default='', help="path where to save, empty for no saving")

    parser.add_argument('--device', default='cuda', help="device to use for training / testing")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help="resume from checkpoint")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help="start epoch")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--world_size', default=1, type=int, help="number of distributed processes")
    parser.add_argument('--dist_url', default='env://', help="url used to set up distributed training")


    parser.add_argument('--enc_layers', default=0, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--image_hidden_dim', default=1408, type=int, help="Visual Backbone output dimension")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--num_obj_classes', default=2260, type=int, help="Number of object classes")
    parser.add_argument('--num_att_classes', default=620, type=int, help="Number of attribute classes")
    parser.add_argument('--att_loss_coef', default=1, type=float, help="attribute loss coefficient")
    parser.add_argument('--scr_coef', default=2, type=float, help="scr loss coefficient")

    parser.add_argument('--fpath_attribute_index', type=str,  default='data/vaw/annotations/attribute_index.json') 
    parser.add_argument('--fpath_head_tail', type=str, default='data/vaw/annotations/head_tail.json') 
    parser.add_argument('--sc_feats', default='data/vaw/annotations/sc_embedding.pt', type=str, help="super-class text embedding")
    parser.add_argument('--att_feats', default='data/vaw/annotations/att_embedding.pt', type=str, help="attribute text embedding")
    parser.add_argument('--hierarchy', default='data/vaw/annotations/hierarchy.json', type=str, help="super-class to attribute hierarchy")
    parser.add_argument('--att_class_weight', default='data/vaw/annotations/att_class_weight.pt', type=str, help="weight for attribute classes")
    parser.add_argument('--base_novel_dict', default="data/vaw/annotations/base_novel_dict.json", type=str, help="base, novel index in attribute classes")

    parser.add_argument('--mode', default='', type=str, help="", choices=('zero_shot', 'supervised'))
    parser.add_argument('--use_scr', default='data/vaw/annotations/scr_tokens/', type=str, help="[MASK] feature for scr loss")
    parser.add_argument('--zrse', action='store_true', help="use zero-shot retrieval-based score enhancement")
    parser.add_argument('--zrse_scale', default=2, type=float)
    parser.add_argument('--ztopk',default=2, type=int)

    return parser



def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=("cpu"))

        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.pretrained:
        if args.eval:
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        
            test_stats = evaluate_att(model, data_loader_test, device, args)
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args)
 
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 2) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SugaFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)