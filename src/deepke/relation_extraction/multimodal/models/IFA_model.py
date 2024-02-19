import torch
from torch import nn

import torch.nn.functional as F
from .modeling_IFA import IFAModel
from deepke.relation_extraction.multimodal.hzk_models.modeling_hzk2 import HZK2
from transformers import BertConfig, BertModel, CLIPConfig, CLIPModel

# class IFAREModel(nn.Module):
#     def __init__(self, num_labels, tokenizer, args):
#         super(IFAREModel, self).__init__()

#         self.args = args
#         self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
#         self.text_config = BertConfig.from_pretrained(self.args.bert_name)

#         clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
#         bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

#         print(self.vision_config)
#         print(self.text_config)

#         # for re
#         self.vision_config.device = args.device
#         self.model = IFAModel(self.vision_config, self.text_config)

#         # load:
#         vision_names, text_names = [], []
#         model_dict = self.model.state_dict()
#         for name in model_dict:
#             if 'vision' in name:
#                 clip_name = name.replace('vision_', '').replace('model.', '')
#                 if clip_name in clip_model_dict:
#                     vision_names.append(clip_name)
#                     model_dict[name] = clip_model_dict[clip_name]
#             elif 'text' in name:
#                 text_name = name.replace('text_', '').replace('model.', '')
#                 if text_name in bert_model_dict:
#                     text_names.append(text_name)
#                     model_dict[name] = bert_model_dict[text_name]
#         assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
#                     (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
#         self.model.load_state_dict(model_dict)

#         self.model.resize_token_embeddings(len(tokenizer))

#         self.dropout = nn.Dropout(0.5)
#         self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
#         self.head_start = tokenizer.convert_tokens_to_ids("<s>")
#         self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
#         self.tokenizer = tokenizer

#     def forward(
#             self, 
#             input_ids=None, 
#             attention_mask=None, 
#             token_type_ids=None, 
#             labels=None, 
#             images=None, 
#             aux_imgs=None,
#             rcnn_imgs=None,
#     ):
#         bsz = input_ids.size(0)
#         output = self.model(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,

#                             pixel_values=images,
#                             aux_values=aux_imgs, 
#                             rcnn_values=rcnn_imgs,
#                             return_dict=True,)

#         last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
#         bsz, seq_len, hidden_size = last_hidden_state.shape
#         entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
#         for i in range(bsz):
#             head_idx = input_ids[i].eq(self.head_start).nonzero().item()
#             tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
#             head_hidden = last_hidden_state[i, head_idx, :].squeeze()
#             tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
#             entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
#         entity_hidden_state = entity_hidden_state.to(self.args.device)
#         logits = self.classifier(entity_hidden_state)
#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss()
#             return loss_fn(logits, labels.view(-1)), logits
#         return logits
    

class IFAREModel_recon(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel_recon, self).__init__()

        self.args = args

        # zs
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        # for re
        self.vision_config.device = args.device
        self.ifa_model = IFAModel(self.vision_config, self.text_config,self.args)


        # load:
        vision_names, text_names = [], []
        model_dict = self.ifa_model.state_dict()

        # zs
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.ifa_model.load_state_dict(model_dict)

        # zs
        self.ifa_model.resize_token_embeddings(len(tokenizer))
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        # self.head_end = tokenizer.convert_tokens_to_ids("</s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        # self.tail_end = tokenizer.convert_tokens_to_ids("</o>")
        self.tokenizer = tokenizer
        # self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)

        self.dropout = nn.Dropout(0.5)
        # HZK
        self.HZK2=HZK2(self.args)
        self.ScaledDotProductAttention= ScaledDotProductAttention(768)
        self.mean = torch.mean

    def forward(
            self,
            # [32,80]
            input_ids=None,
            # [32,80]
            attention_mask=None,
            # [32,80]
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        #批次32
        bsz = input_ids.size(0)
        # zs IFA
        # (32,61)(32,61)(32.61)(32,3,224,224)(32,3,3,224,224)(32,3,3,224,224)
        output_t,output_v= self.ifa_model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)

        # loss = self.HZK(vision_embedding_output=utterance_text, text_embedding_output=hzk_v_output)
        a,b,c,T_recon,V_recon = self.HZK2(text_embedding_output=output_t.last_hidden_state, vision_embedding_output=output_v.last_hidden_state)

        # [32,80,768],[32,768]
        last_hidden_state_t= T_recon
        pooler_output=output_t.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state_t.shape
        # v
        # last_hidden_state_v,pooler_output_v= output_v.last_hidden_state,output_v.pooler_output
        last_hidden_state_v=V_recon


        entity_hidden_state = torch.Tensor(bsz, hidden_size * 2)  # batch, 2*hidden

        last_hidden_state_t, _ = self.ScaledDotProductAttention(last_hidden_state_t,last_hidden_state_v,last_hidden_state_v)

        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state_t[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state_t[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        
        # last_hidden_state_new= torch.Tensor(bsz,80,hidden_size)
        # for i in range(bsz):

        #     last_hidden_state_new[i,:,:],none=self.ScaledDotProductAttention(last_hidden_state_t[i,:,:],last_hidden_state_v[i,:,:],last_hidden_state_v[i,:,:])

        #     # try:
        #     head_idx = input_ids[i].eq(self.head_start).nonzero().item()
        #     tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
        #     head_idx_end = input_ids[i].eq(self.head_end).nonzero().item()
        #     tail_idx_end = input_ids[i].eq(self.tail_end).nonzero().item()
        #     # except:
        #     #     print(head_idx,tail_idx,head_idx_end,tail_idx_end)
        #     # [768]
        #     head_hidden = last_hidden_state_new[i, head_idx, :]
        #     tail_hidden = last_hidden_state_new[i, tail_idx, :]
        #     # head_hidden_mean = torch.mean(head_hidden,dim=0)
        #     # tail_hidden_mean = torch.mean(tail_hidden,dim=0)

        #     #
        #     # head_hidden = head_entity.squeeze()
        #     # tail_hidden = tail_entity.squeeze()
        #     for idnx in range(head_idx+1,head_idx_end+1):
        #         head_hidden = head_hidden + last_hidden_state_new[i, idnx, :]
        #     head_hidden = head_hidden/(head_idx-head_idx_end+1)

        #     for idnx in range(tail_idx+1,tail_idx_end+1):
        #         tail_hidden = tail_hidden + last_hidden_state_new[i, idnx, :]
        #     tail_hidden = tail_hidden / (tail_idx - tail_idx_end + 1)
        #     # head_hidden = last_hidden_state_new[i, head_idx, :].squeeze()
        #     #
        #     # tail_hidden = last_hidden_state_new[i, tail_idx, :].squeeze()


        #     # [1536]0
        #     entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

        # (32,1536)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # (32,23)
        logits = self.classifier(entity_hidden_state)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # loss
            # return loss_fn(logits, labels.view(-1)), logits
            return a+b+c+loss_fn(logits, labels.view(-1)), logits,a,b,c
        return logits

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(61, 80)
        self.tanh = nn.Tanh()

    def forward(self, q, k, v):
        # 求g
        g_fc1 = self.fc1(v)
        g_tanh = self.tanh(g_fc1)
        g_fc2 = self.fc2(g_tanh)
        g = g_fc2.squeeze()
        # g2 = self.fc3(g1)
        # qk
        scores = torch.matmul(q, k.transpose(-1, -2))
        # 求s
        # 对注意力分数进行缩放处理
        scores_scaled = scores / torch.sqrt(torch.tensor(q.shape[1]).float())
        # 对注意力分数进行softmax归一化
        s = F.softmax(scores_scaled, dim=1)
        # weights
        # torch.Size([80, 61]) torch.Size([61]) torch.Size([80, 61]) torch.Size([61, 1])
        # print(torch.mean(s, dim=1, keepdim=True).size())
        # print(scores_scaled.size(), g.size(), s.size(), g_fc2.size())
        g = g.unsqueeze(-2)
        attn_scores = torch.add(torch.mul(s, (1 - g)), torch.mul(torch.mean(s, dim=1, keepdim=True), g)) / self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)
        # output
        attn_output = torch.matmul(attn_weights, v)
        attn_output=attn_output+q

        return attn_output, attn_weights
    
# class TVFusion(nn.Module):
#     def __init__(self, embed_dim, dropout=0):
#         super(TVFusion, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#         self.sigmoid = nn.Sigmoid()

#         self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
#         self.fc1 = nn.Linear(768, 512)
#         self.fc2 = nn.Linear(512, 1)
#         self.fc3 = nn.Linear(61, 80)
#         self.tanh = nn.Tanh()

#     def forward(self, ut, uv):
#         ut_uv = torch.mul(ut, uv)
#         score_softmax = self.softmax(ut_uv)
#         score_sigmoid = self.sigmoid(ut_uv)
#         # 求g
#         g_fc1 = self.fc1(v)
#         g_tanh = self.tanh(g_fc1)
#         g_fc2 = self.fc2(g_tanh)
#         g = g_fc2.squeeze()
#         # g2 = self.fc3(g1)
#         # qk
#         scores = torch.matmul(q, k.transpose(-1, -2))
#         # 求s
#         # 对注意力分数进行缩放处理
#         scores_scaled = scores / torch.sqrt(torch.tensor(q.shape[1]).float())
#         # 对注意力分数进行softmax归一化
#         s = F.softmax(scores_scaled, dim=1)
#         # weights
#         # torch.Size([80, 61]) torch.Size([61]) torch.Size([80, 61]) torch.Size([80])
#         # print(torch.mean(s, dim=1).size())
#         # print(scores_scaled.size(), g1.size(), s.size(), g2.size())
#         attn_scores = torch.add(torch.mul(scores_scaled, (1 - g)), torch.mul(torch.mean(s, dim=0), g)) / self.scale
#         attn_weights = self.softmax(attn_scores)
#         attn_weights = self.dropout(attn_weights)
#         # output
#         attn_output = torch.matmul(attn_weights, v)
#         attn_output=attn_output+q

#         return attn_output, attn_weights