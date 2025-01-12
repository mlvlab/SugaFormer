from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .lavis.models import load_model_and_preprocess
from .position_encoding import build_position_encoding
from typing import List
from torch import nn
import torch
import math

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs_list = self[0](tensor_list.tensors)
        out: List[NestedTensor] = []
        pos = []
        for xs in xs_list:
            xs = xs[:,1:,:]
            B, D, C = xs.shape
            D_s = int(math.sqrt(D))
            x = xs.permute(0,2,1).view(B,C,D_s,D_s)
            x = nested_tensor_from_tensor_list([x_s for x_s in x]) 
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_blip_backbone(args):
    position_embedding = build_position_encoding(args)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = load_model_and_preprocess(name="blip2", model_type="pretrain", device=device)
    visual_encoder = model.visual_encoder
    visual_encoder = Joiner(visual_encoder, position_embedding)
    return visual_encoder, model
