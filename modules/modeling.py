#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: modeling.py
# @Version: version 1.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel
from modules.until_module import CrossEn

from modules.module_cross import CrossConfig
from modules.module_cross import Transformer as TransformerClip

from modules.module_clip import CLIP
from modules.module_clip import convert_weights

# logging the parameters
logger = logging.getLogger(__name__)



class CLIP2VideoPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP2VideoPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        # obtain the basic parameter in CLIP model
        if state_dict is None:
            state_dict = {}

        # obtain the basic cross config
        clip_state_dict = CLIP.get_config(clip_path=task_config.clip_path)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        #obtain the basic model and initialization
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        # initialize the parameters of clip
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        # initialize the model with other modules
        if model.sim_type == "seqTransf":

            contain_frame_position = False

            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break

            if contain_frame_position is False:
                for key, val in clip_state_dict.items():

                    # initialize for the mlp transformation
                    if key == 'visual.proj':
                        if task_config.temporal_proj == 'sigmoid_mlp':
                            state_dict['frame2t_projection'] = val[0:512].clone()

                    # initialize for the frame and type postion embedding
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        if task_config.temporal_type == 'TDB':
                            state_dict["type_position_embeddings.weight"] = val[0:2].clone()

                    # using weight of first 4 layers for initialization
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # initialize TDB's difference-level attention
                        if task_config.temporal_proj == 'sigmoid_selfA' and num_layer < 1:
                            state_dict[key.replace("transformer.", "frame2t_attention.")] = val.clone()

                        # initialize the one-layer transformer for the input of TAB
                        if (task_config.center_proj == 'TAB' or task_config.center_proj == 'TAB_TDB') and num_layer < 1:
                            state_dict[key.replace("transformer.", "actionClip.")] = val.clone()

                        # initialize the 4-layer temporal transformer
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        # init model  with loaded parameters
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


class CLIP2Video(CLIP2VideoPreTrainedModel):
    """main code for CLIP2Video
    Attributes:
        task_config: hyper-parameter from args
        center_type: indicate to whether use TAB or TDB
        temporal_type: default to use the standard type, while TDB to use the TDB block
        temporal_proj: different type to encode difference
        centerK: the center number of TAB block
        center_weight: the weight of TAB loss
        center_proj: TAB_TDB, TAB
    """

    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP2Video, self).__init__(cross_config)
        self.task_config = task_config

        # for TDB block
        self.temporal_type = task_config.temporal_type
        self.temporal_proj = task_config.temporal_proj

        # for TAB block
        self.center_type = task_config.center_type
        self.centerK = task_config.centerK
        self.center_weight = task_config.center_weight
        self.center_proj = task_config.center_proj

        # set the parameters of CLIP
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP]
        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = task_config.vocab_size #["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))

        if vocab_size == 49408:
            key_name = ["input_resolution", "context_length", "vocab_size"]
        else:
            key_name = ["input_resolution", "context_length", "vocab_size", "token_embedding.weight"]

        for key in key_name:
            if key in clip_state_dict:
                del clip_state_dict[key]

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
        ).float()
        convert_weights(self.clip)

        # set the type of similarity calculator
        self.sim_type = 'meanP'
        if hasattr(task_config, "sim_type"):
            self.sim_type = task_config.sim_type
            show_log(task_config, "\t sim_type: {}".format(self.sim_type))

        # load the max length of positional embedding
        cross_config.max_position_embeddings = context_length


        if self.sim_type == "seqTransf":

            # positional embedding for temporal transformer
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)

            # 4-layer temporal transformer
            self.transformerClip = TransformerClip(width=transformer_width,
                                                   layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )

        if self.temporal_proj == 'sigmoid_mlp':

            # initialize for mlp
            self.frame2t_projection = nn.Parameter(torch.empty(512, 512))
            nn.init.normal_(self.frame2t_projection, std=64 ** -0.5)
        elif self.temporal_proj == 'sigmoid_selfA':

            # initialize for difference-level attention
            self.frame2t_attention = TransformerClip(width=transformer_width, layers=1, heads=transformer_heads, )

        # initialize difference pipeline for 'TDB', use default to use the standard structure
        if self.temporal_type == 'TDB':
            self.type_position_embeddings = nn.Embedding(2, cross_config.hidden_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.trans_layernorm = torch.nn.LayerNorm(512)

        if self.center_type == 'TAB':

            # initialize the weight center
            self.weight_center = nn.Parameter(torch.empty(self.centerK, 512))
            nn.init.normal_(self.weight_center, std=64 ** -0.5)
            # initialize the embedding center
            self.emb_center = nn.Parameter(torch.empty(self.centerK, 512))
            nn.init.normal_(self.emb_center, std=64 ** -0.5)

            # initialize the 1-layer transformer used in TAB
            if self.center_proj == 'TAB' or self.center_proj == 'TAB_TDB':
                self.actionClip = TransformerClip(width=transformer_width, layers=1, heads=transformer_heads, )

        # initial loss (Cross entropy loss)
        self.loss_fct = CrossEn()

        self.apply(self.init_weights)


    def calc_loss(self, sequence_output, visual_output, attention_mask, video_mask):
        """ calculate  loss
        Args:
            sequence_hidden_output: token embedding
            visual_output: frame embedding
            attention_mask: caption mask
            video_mask: video mask
        Returns:
            sim_loss: loss for optimization
        """

        sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                                      video_mask, shaped=True)
        # text-to-video loss
        sim_loss1 = self.loss_fct(sim_matrix)

        # video-to-text loss
        sim_loss2 = self.loss_fct(sim_matrix.T)

        sim_loss = (sim_loss1 + sim_loss2) / 2

        return sim_loss

    def get_extra_TAB_embedding(self, embedding_out, attention_mask):
        """ obtain frame embedding concentrated with temporal embedding
        Args:
            embedding_out: token embedding
            attention_mask: frame embedding
        Returns:
            embedding_out: token embedding with temporal enhancing
            attention_mask: frame embedding with temporal enhancing
        """
        large_position_d = torch.arange(start=0, end=embedding_out.size()[1], step=2, dtype=torch.long,
                                        device=embedding_out.device)
        large_embedding_out = embedding_out[:, large_position_d, :]  # bs * 6 * 512
        large_attention_mask = attention_mask[:, large_position_d]

        # embedding_out: bs * seq * 512 | local_out: bs * seq * (k + 1)
        if self.center_proj == 'TAB' or self.center_proj == 'TAB_TDB':

            # sample in the large frame rate

            # obtain the attention mask of large frame rate
            large_attention_mask_span = large_attention_mask.squeeze(-1)
            large_attention_mask_span = large_attention_mask_span.squeeze(-1)

            # prepare the position embedding and store the input embedding
            seq_length = large_embedding_out.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=large_embedding_out.device)
            position_ids = position_ids.unsqueeze(0).expand(large_embedding_out.size(0), -1)
            TAB_position_embedding = self.frame_position_embeddings(position_ids)
            large_embedding_out_original = large_embedding_out

            if self.center_proj == 'TAB_TDB':
                # shared TDB is adopted to insert difference-enhanced token
                large_embedding_out, TAB_position_embedding, TAB_type_embedding, large_attention_mask_span = self.temporal_difference_block(
                    large_embedding_out, large_attention_mask_span)
                large_embedding_out = large_embedding_out + TAB_position_embedding + TAB_type_embedding
            else:
                large_embedding_out = large_embedding_out + TAB_position_embedding  # batch_size * 12 * 512

            extended_video_mask = (1.0 - large_attention_mask_span.unsqueeze(1)) * -1000000.0  # batch_size * 1* 12
            extended_video_mask = extended_video_mask.expand(-1, large_attention_mask_span.size(1),
                                                             -1)  # batch_size * 12 * 12

            # adopt 1-layer temporal transformer to encode representation
            large_embedding_out = large_embedding_out.permute(1, 0, 2)  # NLD -> LND # 12 * batch_size * 512
            large_embedding_out = self.actionClip(large_embedding_out, extended_video_mask)  # 12 * batch_size * 512
            large_embedding_out = large_embedding_out.permute(1, 0, 2)  # LND -> NLD # batch_size * 12 * 512

            # adopt the output of frame token if use TAB_TDB
            if self.center_proj == 'TAB_TDB':
                frame_position_id = torch.arange(start=0, end=large_embedding_out.size()[1], step=2, dtype=torch.long,
                                                 device=large_embedding_out.device)
                large_embedding_out = large_embedding_out[:, frame_position_id, :]

            # concat the original embedding and encoded embedding with temporal correlations
            large_embedding_out = large_embedding_out + large_embedding_out_original
            embedding_out = torch.cat((embedding_out, large_embedding_out), 1)
            attention_mask = torch.cat((attention_mask, large_attention_mask), 1)

        return embedding_out, attention_mask

    def get_TAB_embedding(self, embedding_out, attention_mask, type='default'):
        """ obtain aligned embedding for video and text
        Args:
            embedding_out: token embedding
            attention_mask: frame embedding
        Returns:
            cluster_embedding: aligned embedding
        """
        if type == 'visual':
            embedding_out, attention_mask = self.get_extra_TAB_embedding(embedding_out, attention_mask)


        soft_weight = F.softmax(embedding_out @ self.weight_center[0:self.centerK].t(), 2)


        cluster_embedding = soft_weight.unsqueeze(3) * (embedding_out.unsqueeze(2) - self.emb_center[0:self.centerK])
        cluster_embedding = torch.sum(cluster_embedding * attention_mask, 1)

        cluster_embedding = cluster_embedding / cluster_embedding.norm(dim=-1, keepdim=True)
        cluster_embedding = torch.mean(cluster_embedding, dim=1)
        cluster_embedding = cluster_embedding / cluster_embedding.norm(dim=-1, keepdim=True)

        return cluster_embedding


    def calc_TAB_loss(self, sequence_hidden_output, visual_output, attention_mask, video_mask):
        """ calculate TAB loss
         Args:
             sequence_hidden_output: token embedding
             visual_output: frame embedding
             attention_mask: caption mask
             video_mask: video mask
         Returns:
             sim_loss: loss for optimization
         """

        # obtain the aligned video representation
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_mask_un = video_mask_un.unsqueeze(-1)
        cluster_visual_output = self.get_TAB_embedding(visual_output, video_mask_un, type='visual')

        # obtain the aligned text representation
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        attention_mask_un = attention_mask_un.unsqueeze(-1)
        cluster_sequence_output = self.get_TAB_embedding(sequence_hidden_output, attention_mask_un, type='sequence')

        # calculate the similarity
        logit_scale = self.clip.logit_scale.exp()
        sim_matrix = logit_scale * torch.matmul(cluster_sequence_output, cluster_visual_output.t())

        # text-to-video loss
        sim_loss1 = self.loss_fct(sim_matrix)

        # video-to-text loss
        sim_loss2 = self.loss_fct(sim_matrix.T)

        sim_loss = (sim_loss1 + sim_loss2) / 2
        return sim_loss

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        """ forward method during training
        Args:
            input_ids: caption id
            token_type_ids: type id
            attention_mask: caption mask
            video_mask: video mask
            shaped: False to reshape
        Returns:
            loss: total loss
            TAB_losses: TAB loss
        """

        # reshape the input text
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        # reshape the input video
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # obtain the frame and text representation
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True,
                                                                         video_frame=video_frame)
        if self.training:
            loss = 0.
            TAB_losses = 0.

            if self.center_type == 'TAB':
                # tuple, which contain the output of cls and other tokens
                sequence_output, sequence_hidden_output = sequence_output

                # calculate the total loss
                sim_loss = self.calc_loss(sequence_output, visual_output, attention_mask, video_mask)
                loss += sim_loss * self.center_weight

                # calculate TAB loss
                TAB_loss = self.calc_TAB_loss(sequence_hidden_output, visual_output, attention_mask, video_mask)

                # combine two losses, where loss is for optimization and TAB_losses if for logging
                loss += TAB_loss * (1 - self.center_weight)
                TAB_losses += TAB_loss

            else:
                # calculate the total loss
                sim_loss, _ =  self.calc_loss(sequence_output, visual_output, attention_mask, video_mask)
                loss += sim_loss

            return loss, TAB_losses
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        """Encode text representation
        Args:
            input_ids: caption id
            token_type_ids: type id
            attention_mask: caption mask
            shaped: False to reshape
        Returns:
            sequence_output: output embedding [1,512]
            visual_output: output embedding [1,512]
        """
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)

        if self.center_type == 'TAB':
            sequence_hidden, return_hidden = self.clip.encode_text(input_ids, return_hidden=True)
            sequence_hidden = sequence_hidden.float()

            return_hidden = return_hidden.float()
            return_hidden = return_hidden.view(bs_pair, -1, return_hidden.size(-1))

            return sequence_hidden, return_hidden
        else:
            sequence_hidden = self.clip.encode_text(input_ids).float()
            sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        """Encode video representation
        Args:
            video: video frames
            video_mask: video mask
            video_frame: frame length of video
            shaped: False to reshape
        Returns:
            visual_hidden: output embedding [1,512]
        """

        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False,
                                   video_frame=-1):
        """Encode text and video representation
        Args:
            input_ids: caption id
            token_type_ids: type id
            attention_mask: caption mask
            video: video frames
            video_mask: video mask
            video_frame: frame length of video
        Returns:
            sequence_output: output embedding [1,512]
            visual_output: output embedding [1,512]
        """

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        # encode text representation
        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)

        # encode video representation
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output



    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        """average pooling for the overall text representation
        Args:
            sequence_output: embedding
            attention_mask: caption mask
        Returns:
            text_out: output embedding [1,512]
        """

        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        """average pooling for the overall video representation
        Args:
            visual_output: embedding
            video_mask: video embedding
        Returns:
            video_out: output embedding [1,512]
        """

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        """average pooling for the overall video representation
        Args:
            sequence_output: embedding
            visual_output: embedding
            attention_mask: caption mask
            video_mask: video mask
        Returns:
            text_out:output embedding [1,512]
            video_out: output embedding [1,512]
        """

        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_type="meanP"):
        """Calculate the similarity between visual and text representation
        Args:
            sequence_output: embedding
            visual_output: embedding
            attention_mask: caption mask
            video_mask: video mask
            sim_type: header for aggregate the video representation
        Returns:
            retrieve_logits: similarity
        """

        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_type == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_type == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output # batch_size * 12 * 512
            seq_length = visual_output.size(1) # 12
            # obtain positional embedding
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device) # 12
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1) # batch_size * 12
            frame_position_embeddings = self.frame_position_embeddings(position_ids) # batch_size * 12 * 512

            # add positional embedding into visual_output
            visual_output = visual_output + frame_position_embeddings # batch_size * 12 * 512

            # obtain the output of temporal transformer
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0 # batch_size * 1* 12
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1) # batch_size * 12 * 12
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND # 12 * batch_size * 512
            visual_output = self.transformerClip(visual_output, extended_video_mask) #12 * batch_size * 512
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 12 * 512
            visual_output = visual_output + visual_output_original

        # normalize the video representation
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        # normalize the text representation
        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        # calculate the similarity
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())

        return retrieve_logits

    def temporal_difference_block(self, visual_output, video_mask):
        """Calculate difference-enhanced token and inset into frame token
        Args:
            visual_output: embedding
            video_mask: video mask
        Returns:
            visual_output: frame representation
            frame_position_embeddings: position embedding
            type_embedding: type embedding
            temporal_video_mask: attention mask
        """

        seq_length = visual_output.size(1) # 12

        # obtain the positional embedding
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device) # 12
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1) # batch_size * 12
        frame_position_embeddings = self.frame_position_embeddings(position_ids) # batch_size * 12 * 512

        # obtain the type embedding to indicate the frame token and difference-enhanced token
        video_ids = torch.ones_like(position_ids)
        videoDif_ids = torch.zeros_like(position_ids)
        video_type_embedding = self.type_position_embeddings(video_ids)
        videoDif_type_embedding = self.type_position_embeddings(videoDif_ids)

        # adopt temporal_proj == sigmoid_mlp for mlp transformation
        # adopt temporal_proj == sigmoid_selfA for difference-level attention
        # adopt temporal_proj == default to use subtraction directly

        # batch size * 11 * 512
        dif_visual_output = visual_output[:, 1: seq_length, :] - visual_output[:, 0: seq_length - 1, :]
        if self.temporal_proj == 'sigmoid_mlp':
            # adopt sigmoid to transform into [-1, 1]
            dif_visual_output = 2 * self.sigmoid(self.trans_layernorm(dif_visual_output @ self.frame2t_projection)) - 1

        elif self.temporal_proj == 'sigmoid_selfA':
            # batch_size * 11 * 512
            dif_visual_output = dif_visual_output + frame_position_embeddings[:, 1:seq_length, :]
            trans_video_mask = video_mask[:,1:seq_length]
            # batch_size * 1* 11
            extend_trans_video_mask = (1.0 - trans_video_mask.unsqueeze(1)) * -1000000.0
            # batch_size * 11 * 11
            extend_trans_video_mask = extend_trans_video_mask.expand(-1, trans_video_mask.size(1), -1)

            dif_visual_output = dif_visual_output.permute(1, 0, 2)  # NLD -> LND # 11 * batch_size * 512
            dif_visual_output = self.frame2t_attention(dif_visual_output, extend_trans_video_mask)
            dif_visual_output = dif_visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 11 * 512

            dif_visual_output = 2 * self.sigmoid(self.trans_layernorm(dif_visual_output)) - 1

        # batch size * (12+11) * 512
        visual_middle = torch.cat((visual_output, dif_visual_output), 1)
        # batch size * (12+12) * 512
        frame_position_embeddings_middle = torch.cat((frame_position_embeddings, frame_position_embeddings), 1)
        temporal_video_mask_middle = torch.cat((video_mask, video_mask), 1)
        type_embedding_middle = torch.cat((video_type_embedding, videoDif_type_embedding), 1)

        # obtain the correct index to insert difference-enhanced token
        seq1_indices = torch.arange(start=0, end=seq_length, step=1, dtype=torch.long)
        seq2_indices = torch.arange(start=seq_length, end=2 * seq_length - 1, step=1, dtype=torch.long)
        seq_indices = torch.stack((seq1_indices[0], seq2_indices[0]))
        for i in range(1, seq_length - 1):
            seq_indices = torch.cat((seq_indices, seq1_indices[i].view(1), seq2_indices[i].view(1)))
        seq_indices = torch.cat((seq_indices, seq1_indices[seq_length - 1].view(1))).cuda()

        # insert difference-enhanced token between every adjacent frame token
        visual_output = visual_middle.index_select(1, seq_indices)
        frame_position_embeddings = frame_position_embeddings_middle.index_select(1, seq_indices)
        temporal_video_mask = temporal_video_mask_middle.index_select(1, seq_indices)
        type_embedding = type_embedding_middle.index_select(1, seq_indices)

        return visual_output, frame_position_embeddings, type_embedding, temporal_video_mask

    def _similarity_TDB(self, sequence_output, visual_output, attention_mask, video_mask):
        """Calculate the similarity between visual and text representation by adding TDB
        Args:
            sequence_output: embedding
            visual_output: embedding
            attention_mask: caption mask
            video_mask: video mask
        Returns:
            retrieve_logits: similarity
        """

        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        # obtain the basic embedding
        visual_output_original = visual_output # batch_size * 12 * 512

        # difference-enhanced token obtained by TDB
        visual_output, frame_position_embeddings, type_embedding, temporal_video_mask = self.temporal_difference_block(
            visual_output, video_mask)

        # obtain the output of transformer
        visual_output = visual_output + frame_position_embeddings + type_embedding # batch_size * 12 * 512
        extended_video_mask = (1.0 - temporal_video_mask.unsqueeze(1)) * -1000000.0 # batch_size * 1* 12
        extended_video_mask = extended_video_mask.expand(-1, temporal_video_mask.size(1), -1) # batch_size * 12 * 12
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND # 12 * batch_size * 512
        visual_output = self.transformerClip(visual_output, extended_video_mask) #12 * batch_size * 512
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 12 * 512

        # select the output of frame token for final video representation
        frame_position_id = torch.arange(start=0, end=visual_output.size()[1], step=2, dtype=torch.long,
                                         device=visual_output.device)
        visual_output = visual_output[:, frame_position_id, :]
        visual_output = visual_output + visual_output_original
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        # mean pooling for video representation
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1) # batch_size * 12 * 1
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        visual_output = torch.sum(visual_output, dim=1) / video_mask_un_sum

        # obtain the normalized video embedding
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        # obtain the normalized sequence embedding
        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        # calculate the similarity
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())

        return retrieve_logits


    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False):
        """get the similarity for global representation during training
          Args:
              sequence_output: embedding
              visual_output: embedding
              attention_mask: caption mask
              video_mask: video mask
              shaped: whether to shape the dimension
          Returns:
              retrieve_logits: output similarity
          """

        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if self.sim_type == 'seqTransf' and self.temporal_type == 'TDB':
            # adopting temporal transformer with TDB block
            retrieve_logits = self._similarity_TDB(sequence_output, visual_output, attention_mask, video_mask)
        else:
            # adopting mean pooling or use temporal transformer to aggregate the video representation
            assert self.sim_type in ["meanP", "seqTransf"]
            retrieve_logits = self._similarity(sequence_output, visual_output, attention_mask, video_mask,
                                               sim_type=self.sim_type)

        return retrieve_logits


    def get_inference_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False):
        """get the similarity for global and local representation during inference
         Args:
             sequence_output: embedding
             visual_output: embedding
             attention_mask: caption mask
             video_mask: video mask
             shaped: whether to shape the dimension
         Returns:
             text_out:output embedding [1,512]
             video_out: output embedding [1,512]
         """
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()

        if self.center_type == 'TAB':
            sequence_output, sequence_hidden_output = sequence_output

        if self.sim_type == 'seqTransf' and self.temporal_type == 'TDB':
            # adopting temporal transformer with TDB block
            retrieve_logits = self._similarity_TDB(sequence_output, visual_output, attention_mask, video_mask)
        else:
            # adopting mean pooling or use temporal transformer to aggregate the video representation
            assert self.sim_type in ["meanP", "seqTransf"]
            retrieve_logits = self._similarity(sequence_output, visual_output, attention_mask, video_mask,
                                               sim_type=self.sim_type)

        # calculate the similarity aggregated in TAB block
        if self.center_type == 'TAB':

            # calculate the aligned video representation
            video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
            video_mask_un = video_mask_un.unsqueeze(-1)
            cluster_visual_output = self.get_TAB_embedding(visual_output, video_mask_un, type='visual')

            # calculate the aligned text representation
            attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
            attention_mask_un[:, 0, :] = 0.
            attention_mask_un = attention_mask_un.unsqueeze(-1)
            cluster_sequence_output = self.get_TAB_embedding(sequence_hidden_output, attention_mask_un, type='sequence')

            logit_scale = self.clip.logit_scale.exp()
            sim_matrix = logit_scale * torch.matmul(cluster_sequence_output, cluster_visual_output.t())

            retrieve_logits = retrieve_logits * self.center_weight  + sim_matrix * ( 1 - self.center_weight)


        return retrieve_logits, contrastive_direction

