import torch.nn.functional as F
import datasets.transforms as T
from pathlib import Path
import torch.utils.data
from PIL import Image
import numpy as np
import torch
import json

class VAW_Dataloader(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, attribute_index, transforms, args=None):
        self.att_ids = list(self.load_json(attribute_index).values())
        self.annotations = self.load_json(anno_file)
        self.img_set = img_set
        self.img_folder = img_folder
        self.transforms = transforms
        self.sc_token_path = args.use_scr

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def num_attributes(self):
        return len(self.att_ids)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]
        img_id = str(img_anno['image_id'])
        object_names = img_anno['object_name']
        file_dir = self.get_file_path(img_anno['file_name'])
        img = Image.open(self.img_folder / file_dir).convert('RGB')
        boxes, orig_size, cropped_masks, keep = self.process_boxes_and_mask(img_anno, img)
        pos_att_classes, neg_att_classes = self.create_attribute_classes(img_anno)
        pos_att_classes = pos_att_classes[keep]
        neg_att_classes = neg_att_classes[keep]
        if self.img_set == 'train':
            sc_mask_output = self.get_sc_mask_output(img_id)
            sc_mask_output = sc_mask_output[keep]
            target = self.create_target_dict(img_anno, boxes, pos_att_classes, neg_att_classes, orig_size, object_names, sc_mask_output)
        elif self.img_set == 'test':
            target = self.create_target_dict(img_anno, boxes, pos_att_classes, neg_att_classes, orig_size, object_names)
        transformed_img, target = self.apply_transforms(img, target)
        crop_imgs = self.crop_and_normalize_boxes(img, boxes)
        return transformed_img, crop_imgs, cropped_masks, target

    def get_file_path(self, file_name):
        return file_name.split('/')[-2] + '/' + file_name.split('/')[-1]

    def process_boxes_and_mask(self, img_anno, img):
        boxes = torch.as_tensor(img_anno['boxes'], dtype=torch.float32).reshape(-1, 4)
        w, h = img.size
        orig_size = torch.as_tensor([h, w])
        boxes[:, 2:] += boxes[:, :2]  
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        mask_path = img_anno['mask']
        masks = torch.from_numpy(np.load(mask_path))
        masks = masks[keep]
        cropped_masks = self.crop_masks(masks, boxes, orig_size)
        return boxes, orig_size, cropped_masks, keep

    def create_attribute_classes(self, img_anno):
        pos_att_classes = torch.zeros((len(img_anno['boxes']), self.num_attributes()), dtype=torch.float32)
        neg_att_classes = torch.zeros((len(img_anno['boxes']), self.num_attributes()), dtype=torch.float32)
        for b, pos_id in zip(pos_att_classes, img_anno['pos_att_id']):
            b[pos_id] = 1
        for b, neg_id in zip(neg_att_classes, img_anno['neg_att_id']):
            b[neg_id] = 1
        return pos_att_classes, neg_att_classes

    def get_sc_mask_output(self, img_id):
        sc_mask_output = torch.load(self.sc_token_path + img_id + '.pt', map_location='cpu')
        return sc_mask_output

    def create_target_dict(self, img_anno, boxes, pos_att_classes, neg_att_classes, orig_size, object_names, sc_mask_output=None):
        target = {
            'boxes': boxes,
            'pos_att_classes': pos_att_classes,
            'neg_att_classes': neg_att_classes,
            'img_id': img_anno['image_id'],
            'orig_size': orig_size,
            'obj_names': object_names,
        }
        if sc_mask_output is not None:
            target['sc_token_output'] = sc_mask_output
        return target

    def apply_transforms(self, img, target):
        if self.transforms:
            transformed_img, _ = self.transforms(img, target)
        return transformed_img, target

    def crop_and_normalize_boxes(self, img, boxes):
        crop_imgs = []
        for box in boxes:
            box = box.int()
            cropped_img = img.crop((box[0].item(), box[1].item(), box[2].item(), box[3].item()))
            cropped_img = T.ToTensor()(cropped_img)  
            cropped_img, _ = T.resize(cropped_img, None, size=(224, 224))
            cropped_img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(cropped_img)
            crop_imgs.append(cropped_img)  
        return torch.stack(crop_imgs)

    def crop_masks(self, masks, boxes, orig_size):
        resized_masks = []
        for mask, box in zip(masks, boxes):
            crop_resized_mask = self.mask_crop_resize(mask.unsqueeze(0), orig_size, box)
            resized_masks.append(crop_resized_mask)
        return torch.cat(resized_masks)

    def mask_crop_resize(self, mask, orig_size, box, img_size=224):
        orig_h, orig_w = orig_size[0].item(), orig_size[1].item()
        box[0::2].clamp_(min=0, max=orig_w)
        box[1::2].clamp_(min=0, max=orig_h)
        cropped_mask = mask[:, :orig_h, :orig_w][:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]        
        if True in cropped_mask:
            resized_mask = F.interpolate(cropped_mask.unsqueeze(0).type(torch.float), size=(img_size, img_size), mode='bicubic', align_corners=False)
            resized_mask = (resized_mask > 0.5).float()        
        else: 
            resized_mask = torch.ones((1, 1, img_size, img_size))
        return resized_mask

def make_vaw_transforms(image_set, img_size=224):

    def transform(img, target):
        img = T.ToTensor()(img)
        img, _ = T.resize(img, None, size=(img_size, img_size))  
        img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img) 
        return img, target

    if image_set in ['train', 'test']:
        return transform

def build(image_set, args):
    root = Path(args.vaw_path)
    assert root.exists(), f'Provided VAW path {root} does not exist'
    PATHS = {
        'train': (root / 'images', root / 'annotations' / 'train.json'),
        'val': (root / 'images', root / 'annotations' / 'test.json'),
        'test': (root / 'images', root / 'annotations' / 'test.json')
    }

    attribute_index = root / 'annotations' / 'attribute_index.json'

    img_folder, anno_file = PATHS[image_set]
    assert img_folder.exists(), f"Image folder {img_folder} does not exist"
    assert anno_file.exists(), f"Annotation file {anno_file} does not exist"

    transforms = make_vaw_transforms(image_set)
    dataset = VAW_Dataloader(image_set, img_folder, anno_file, attribute_index, transforms, args)

    return dataset
