import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import BertModel,BertConfig

from .hzk_utils import to_gpu
from .hzk_utils import ReverseLayerF

# clip
from deepke.relation_extraction.multimodal.hzk_models.CLIP import *

from typing import Any, Optional, Tuple

# loss
from deepke.relation_extraction.multimodal.hzk_models.hzk_utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD


class HZK(nn.Module):
    # embedding_size,visual_size,num_classes,dropout,activation(),hidden_size=768,use_cmd_sim,,reverse_grad_weight
    # sentences, video, lengths, bert_sent, bert_sent_type, bert_sent_mask canshu
    def __init__(self, config):
        super(HZK, self).__init__()
        # config,size
        self.config = config
        #0 self.text_size = config.embedding_size
        self.text_size =  768
        #0 self.visual_size = config.visual_size
        self.visual_size = 768

        # size
        self.input_sizes = input_sizes = [self.text_size, self.visual_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size)]
        # self.output_size = output_size = config.num_classes
        self.output_size = output_size = 23
        # dropout
        # self.dropout_rate = dropout_rate = config.dropout
        self.dropout_rate = dropout_rate = 0.2

        # activation
        # self.activation = self.config.activation()
        self.activation = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # #! bert-t
        # bertconfig = BertConfig.from_pretrained(self.config.bert_name, output_hidden_states=True)
        # self.text_config=bertconfig
        # self.text_embeddings = BertEmbeddings(bertconfig)
        # self.bertmodel = BertModel.from_pretrained(self.config.bert_name, config=bertconfig)
        # #! IFA vision model
        # self.vision_config = vision_config
        # self.device = vision_config.device
        # self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        # self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        # self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # project_t
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        # 2 1952x768
        # self.project_t = nn.Sequential()
        # self.project_t.add_module('project_t',
        #                           nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.hidden_size))
        # self.project_t.add_module('project_t_activation', self.activation)
        # self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
#         project_v
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[1], out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))
        # fusion
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 4,
                                                           out_features=self.config.hidden_size * 2))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.config.hidden_size * 2, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    #     loss
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()




    def shared_private(self, utterance_t, utterance_v):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)

    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)



    def forward(self, vision_embedding_output,text_embedding_output,lengths=None,
        inputs=None,labels=None,
        return_dict=None,):
        batch_size = 64

        # use clip
        # # def alignment(self, sentences, visual, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        # batch_size = lengths.size(0)
        # # 0 if self.config.use_bert:
        # bert_output = self.bertmodel(input_ids=bert_sent,
        #                              attention_mask=bert_sent_mask,
        #                              token_type_ids=bert_sent_type)
        # bert_output = bert_output[0]
        # # masked mean
        # masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        # mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        # bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        # utterance_text = bert_output
        #
        # # extract features from visual modality
        # # final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        # # utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        # # pixel_values = images,
        # # aux_values = aux_imgs,
        # # rcnn_values = rcnn_imgs
        # #         vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        # #         vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)
        # utterance_video = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        # utterance_video = self.vision_embeddings(utterance_video)

        # 0 Shared-private encoders
        # self.shared_private(utterance_text, utterance_video, utterance_audio)

        utterance_text, utterance_video=vision_embedding_output,text_embedding_output


        self.shared_private(utterance_text, utterance_video)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)


            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)

        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)

        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_v ) / 2.0)

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        # 4,32,61,768
        # h = torch.stack((self.utt_private_t, self.utt_private_v ,self.utt_shared_t,
        #                  self.utt_shared_v), dim=0)
        # # hzk
        # h = self.transformer_encoder(h)
        #
        # h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
        # #   ,23
        # o = self.fusion(h)
            # return o

        # o = self.alignment(sentences, video, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        # print(self.config.recon_weight)
        # return 2*self.get_recon_loss()+self.get_diff_loss()+self.get_cmd_loss()
        return self.config.diff_weight*self.get_diff_loss()+self.config.cmd_weight*self.get_cmd_loss()


        # return o

    def get_diff_loss(self):

        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        private_t = self.utt_private_t
        private_v = self.utt_private_v


        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)

        # Across privates

        loss += self.loss_diff(private_t, private_v)

        return loss

    def get_recon_loss(self, ):

        loss = self.loss_recon(self.utt_t_recon, self.utt_t_orig)
        loss += self.loss_recon(self.utt_v_recon, self.utt_v_orig)

        loss = loss / 2.0
        return loss

    def get_cmd_loss(self, ):

        # losses between shared states
        loss = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, 5)


        return loss

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def _get_resized_embeddings(
            self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


