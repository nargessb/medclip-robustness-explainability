# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:10:04 2025

@author: narges.babadi1
"""

from torch import nn
# from .clip_model import CLIP
# from .clip_surgery_model import CLIPSurgery
from .medclip_surgery_model import MedCLIPSurgery  # Ensure this is implemented and correctly imported
# from medclip import MedCLIPModel  # assumes you have MedCLIP installed
# from medclip import MedCLIPModel, MedCLIPVisionModelViT

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
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



def build_model(name: str, state_dict: dict):
    # --- Vision part: HuggingFace ViT style ---
    if "vision_model.model.embeddings.patch_embeddings.projection.weight" in state_dict:
        vision_width = state_dict["vision_model.model.embeddings.patch_embeddings.projection.weight"].shape[0]
        vision_patch_size = state_dict["vision_model.model.embeddings.patch_embeddings.projection.weight"].shape[-1]
        image_resolution = 224  # Set this explicitly (matches your preprocessing)

        
        # image_resolution = vision_patch_size * grid_size
        vision_layers = len([k for k in state_dict if k.startswith("vision_model.model.encoder.layers.") and k.endswith(".self_attn.out_proj.bias")])
    else:
        raise ValueError("Only ViT-based MedCLIP models are supported. Required vision keys not found.")

    # --- Text part ---
        # --- Text part ---
    embed_dim = state_dict["text_model.projection_head.weight"].shape[0]
    context_length = 77  # fixed for CLIP-like models
    vocab_size = 30522   # BERT-based tokenizer like Bio_ClinicalBERT
    # transformer_width = state_dict["text_model.embeddings.word_embeddings.weight"].shape[1]
    transformer_width = state_dict["text_model.model.embeddings.word_embeddings.weight"].shape[1]
    transformer_layers = len(set(k.split(".")[4] for k in state_dict if k.startswith("text_model.model.encoder.layer")))

    transformer_heads = transformer_width // 64
    # transformer_layers = len(set(k.split(".")[3] for k in state_dict if k.startswith("text_model.encoder.layer")))

    # --- Instantiate the correct model ---
    if 'MCS-' in name:
        model = MedCLIPSurgery(
            embed_dim=512,  # Or extract from state_dict["text_model.projection_head.weight"].shape[0]
            vision_model_path=None,  # already loading from state_dict manually
            text_model_name="emilyalsentzer/Bio_ClinicalBERT",
            image_resolution=224
        )
    else:
        raise ValueError("Unsupported model name for MedCLIP Surgery")

    # Optional: delete unused metadata keys
    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    # Load state
    model.load_state_dict(state_dict, strict=False)  # allow missing keys in surgery logic
    return model.eval()
