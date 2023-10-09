import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from collections import OrderedDict
import timm
from models import swin_transformer as swin
from utils import TextCNN


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
      
class VisualEncoder(nn.Module):
    def __init__(self, args, config):
        super(VisualEncoder, self).__init__()  
        self.args = args
        self.vit = getattr(swin, 'swin_tiny_patch4_window7_224')(pretrained=True, config=config, )
        if args.freeze_imgencoder:
            freeze_layers(self.vit)
        
        if args.image_aggregation == 'fc':
            self.fc = nn.Linear(args.vit_hid, args.embed_size)
            self.init_weights()

        self.imgnorm = args.imgnorm

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, img):
        image_embeds = self.vit.patch_embed(img)
        if self.vit.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.vit.absolute_pos_embed
        image_embeds = self.vit.pos_drop(image_embeds)

        image_embeds_layers = []
        for layer in self.vit.layers:
            image_embeds = layer(image_embeds)
            image_embeds_layers.append(image_embeds)
        
        input = image_embeds_layers[-1]
        if self.args.image_aggregation == 'fc':        
            output = self.fc(input) 
        
        if self.imgnorm:
            output = F.normalize(output, p=2, dim=-1)
        
        return output

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in own_state.items():
            if name in state_dict:
                new_state[name] = state_dict[name]
            else:
                new_state[name] = param
        super(VisualEncoder, self).load_state_dict(new_state)
