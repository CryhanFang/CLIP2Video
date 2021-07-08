#coding:utf-8
# @Time : 2021/6/19
# @File: module_clip.py
# @Version: version 1.0
# Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py

from collections import OrderedDict
import os

import torch
from torch import nn

# if the pretrained model doesn't exist, please download the model from the link of openai
# in this project, we adopt ViT-B/32 for pre-training
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """residual attention block used in transformer

     Attributes:
         attn: multi-head attention
         ln_1: layer normalization
         mlp: MLP
         ln_2: layer normalization
         attn_mask: attention mask
     """

    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple):
        x, video_frame = x_tuple
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)


class Transformer(nn.Module):
    """basic transformer

    Attributes:
        width: dimension for the output of every layer
        layers: total number of layers
        resblocks: residual block
    """

    def __init__(self, width, layers, heads, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x, video_frame=-1):
        return self.resblocks((x, video_frame))[0]


class VisualTransformer(nn.Module):
    """basic vision transformer

    Attributes:
        input_resolution: input resolution of image
        patch_size: patch size to split image
        width: dimension for the output of every layer
        layers: total number of layers
        heads: head for multi-head attention
        output_dim: the final output of ViT
    """

    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):

        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, x, video_frame=-1):

        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class CLIP(nn.Module):
    """basic CLIP model

    Attributes:
        input_resolution: input resolution of image
        patch_size: patch size to split image
        width: dimension for the output of every layer
        layers: total number of layers
        heads: head for multi-head attention
        output_dim: the final output of ViT
    """
    def __init__(self,
                 embed_dim,
                 # vision
                 image_resolution,
                 vision_layers,
                 vision_width,
                 vision_patch_size,
                 # text
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers,
                 ):
        super().__init__()

        # the length of caption
        self.context_length = context_length

        # set vision transformer
        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        # set the text transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(clip_path='/data/ceph_11015/ssd/howiefang/videoCLIP/CLIP2Clip/ViT-B-32.pt'):

        if os.path.exists(clip_path):
            pass
        else:
            raise RuntimeError(f"Model ViT-B/32 not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(clip_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clip_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        """build attention mask for text
        Args:
            context_length: length of caption
        Returns:
            mask: the constructed mask
        """
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        """image encoder
        Args:
            image: image
            return_hidden: whether to return hidden variable
            video_frame: frame length of video
        Returns:
            x: output embedding [1,512]
        """
        hidden = self.visual(image.type(self.dtype), video_frame=video_frame)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj

        x = hidden[:, 0, :]

        if return_hidden:
            return x, hidden

        return x

    def encode_text(self, text, return_hidden=False):
        """text encoder
        Args:
            text: caption
            return_hidden: whether to return hidden variable
        Returns:
            x: output embedding [1,512]
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x, hidden

        return x

    def forward(self, image, text):
        """forward method for CLIP
        Args:
            image: image
            text: caption
        Returns:
            logits_per_image: image-to-text similarity
            logits_per_text: text-to-image similarity
        """

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logit
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)