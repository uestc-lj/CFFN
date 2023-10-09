import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict 
from models.TextNet import TextEncoder
from models.ImageNet import VisualEncoder
from torch.autograd import Variable

class ExpertNet(nn.Module):
    def __init__(self, args):
        super(ExpertNet, self).__init__()   
        self.args = args
        
        self.pos_expert = nn.Sequential(
            nn.Linear(args.embed_size, args.expert_hid),
            nn.Tanh(),
            nn.Linear(args.expert_hid, 1),
            nn.Sigmoid(),
        )
        
        self.neg_expert = nn.Sequential(
            nn.Linear(args.embed_size, args.expert_hid),
            nn.Tanh(),
            nn.Linear(args.expert_hid, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, pos, neg, expert_neg_mask):  
        pos_score = self.pos_expert(pos).masked_fill(expert_neg_mask.unsqueeze(-1), 0).permute(1,0)        
        pos_aggregate = torch.mm(F.relu(pos_score), pos)
        
        neg_score = self.neg_expert(neg).permute(1,0)
        neg_aggregate = torch.mm(F.relu(neg_score), neg)

        return torch.cat([neg_aggregate, pos_aggregate], dim=0)
  
class DetectionModule(nn.Module):
    def __init__(self, args, config):
        super(DetectionModule, self).__init__()   
        self.args = args
        
        self.textEncoder = TextEncoder(args)
        self.visualEncoder = VisualEncoder(args, config)
        
        self.expert = ExpertNet(args)
        
        self.fc = nn.Linear(args.embed_size, 1)
        
        self.classifier_corre = nn.Sequential(
            nn.Linear(args.embed_size, args.h_dim),
            nn.ReLU(),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.h_dim, 2),
        ) 
        
    def forward_emb(self, input_ids, attention_mask, token_type_ids, img_emb):
        word_emb = self.textEncoder(input_ids, attention_mask, token_type_ids)        
        region_emb = self.visualEncoder(img_emb)
        
        return word_emb, region_emb
               
    def fusion_attention(self, word_emb, region_emb, attention_mask): 
        thres = self.args.thres
        
        attn = torch.bmm(word_emb, region_emb.permute(0,2,1))
        attn_thres = attn - torch.ones_like(attn) * thres        
        attn_neg = attn_thres.gt(0).float()
        
        attn_pos = F.softmax(attn.masked_fill((1 - attn_neg).bool(), -1e9) * self.args.lambda_softmax, dim=-1)
        pos = torch.bmm(attn_pos, region_emb) + word_emb
        
        all_word = word_emb.unsqueeze(-2).repeat(1, 1, region_emb.size(1), 1)
        all_region = region_emb.unsqueeze(1).repeat(1, word_emb.size(1), 1, 1)

        all_emb = (all_word + all_region)
        neg_mask = (1 - attn_neg) * attention_mask.unsqueeze(-1)
        
        #all the neg is 1
        expert_neg_mask = attn_neg.sum(-1) == 0 
        text_length = attention_mask.sum(-1)
                
        result = []
        for i in range(word_emb.size(0)):
            current_pos = pos[i, :, :][attention_mask[i, :].bool()]
            current_neg_mask = neg_mask[i, :, :]
            current_all_emb = all_emb[i, :, :, :]
            
            current_neg = current_all_emb[(current_neg_mask).bool()]

            result.append(self.expert(current_pos, current_neg, expert_neg_mask[i, :text_length[i]]))
        
        result = torch.stack(result, dim=0)
        
        choose = self.fc(result).permute(0, 2, 1)
        choose_mask = (attn_neg.sum(-1).sum(-1) == 0)
        choose[:, 0 , 0].masked_fill_(choose_mask, -1e9)
        
        return torch.bmm(F.softmax(choose, dim=-1), result).squeeze(1), F.softmax(choose, dim=-1).squeeze(1)

                
    def forward(self, input_ids, attention_mask, token_type_ids, img_emb):        
        word_emb, region_emb = self.forward_emb(input_ids, attention_mask, token_type_ids, img_emb)
        
        final_result, path_prob = self.fusion_attention(word_emb, region_emb, attention_mask)
        
        return self.classifier_corre(final_result), path_prob

    
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
        super(DetectionModule, self).load_state_dict(new_state)