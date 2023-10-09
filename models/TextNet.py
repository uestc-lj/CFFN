import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 

from pytorch_pretrained_bert.modeling import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import TextCNN

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    
class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_path)        
        if args.freeze_textencoder:
            freeze_layers(self.bert)
        
        if args.text_aggregation == 'fc':
            self.aggregation = nn.Linear(args.bert_hid, args.embed_size)
        elif args.text_aggregation == 'cnn':
            self.aggregation = TextCNN(args)

        self.txtnorm = args.txtnorm        

    def forward(self, input_ids, attention_mask, token_type_ids):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=True)
        input = all_encoder_layers[-1]
        
        if self.args.text_aggregation == 'fc':       
            output = self.aggregation(input) 
            
        elif self.args.text_aggregation == 'cnn':
            output, _ = self.aggregation(input)
        
        if self.txtnorm:
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
        super(TextEncoder, self).load_state_dict(new_state)