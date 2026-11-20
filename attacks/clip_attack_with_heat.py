# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:56:23 2025

@author: narges.babadi1
"""

import clip
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

# --- Load and preprocess image ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("CS-ViT-B/16", device=device)
model.eval()
print("Image projection weight shape:", model.visual.proj.shape)
# print("Text projection weight shape:", model.transformer.proj.shape)

preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

pil_img = Image.open("demo.jpg")
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

# --- Texts ---
all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
target_texts = ['bench', 'person', 'ground', 'building']


# --- Utility: Display similarity map ---
def display_similarity(similarity_map, cv2_img, text_label, phase):
    vis = (similarity_map * 255).astype('uint8')
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    blend = cv2_img * 0.4 + vis * 0.6
    blend = cv2.cvtColor(blend.astype('uint8'), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.title(f'CLIP Surgery ({phase}) - {text_label}')
    plt.imshow(blend)
    plt.axis('off')
    plt.show()

# --- Step 1: Encode text ---
### Explain raw predictions of CLIP, which are opposite and noisy.
### Explain raw predictions of CLIP, which are opposite and noisy.
with torch.no_grad():
    # CLIP architecture surgery acts on the image encoder
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Prompt ensemble for text features with normalization
    tokenized = clip.tokenize(all_texts).to(device)
    text_features = model.encode_text(tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print("Text projection shape:", model.text_projection.shape)
  # should now be [59, 768]
    print("Image feature shape:", text_features.shape)


    
    
# --- Adversarial heatmap-targeted attack ---
# def get_patch_embeddings(model, image):
#     # Extract patch token embeddings from ViT-B/16 CLIP
#     x = model.visual.conv1(image.type(model.dtype))  # [1, 768, 14, 14]
#     x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [1, 196, 768]
    
#     class_embedding = model.visual.class_embedding.to(image.device)
#     cls_tokens = class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # [1, 1, 768]
#     x = torch.cat([cls_tokens, x], dim=1)  # [1, 197, 768]

#     x = x + model.visual.positional_embedding.to(image.device)
#     x = model.visual.ln_pre(x)
#     x = model.visual.transformer(x)
#     x = model.visual.ln_post(x)

#     return x[:, 1:, :]  # [1, 196, 512] ← patch tokens (no CLS)
def generate_adversarial_image(image, model, text_features, target_idx, steps=10, epsilon=0.01):
    adv_image = image.clone().detach().to(device)
    adv_image.requires_grad_(True)

    optimizer = torch.optim.Adam([adv_image], lr=epsilon)
    H, W = image.shape[2:]

    for step in range(steps):
        optimizer.zero_grad()

        # --- Encode image and select CLS token only ---
        image_feat = model.encode_image(adv_image)  # Could be [1, 512] or [1, 197, 512]
        print(f"Step {step+1} - Raw image_feat shape: {image_feat.shape}")
        if image_feat.dim() == 3:
            image_feat = image_feat[:, 0, :]  # Select CLS token only
            print(f"Step {step+1} - Using CLS token, image_feat shape: {image_feat.shape}")
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # --- Extract transformer token features (no grad) ---
        with torch.no_grad():
            x = model.visual.conv1(adv_image)                    # [B, C, 16, 16]
            print(f"Conv1 output shape: {x.shape}")
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, T, C]
            print(f"Tokenized shape before CLS: {x.shape}")
            cls_embed = model.visual.class_embedding.to(x.dtype)
            cls_tokens = cls_embed + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
            x = torch.cat([cls_tokens, x], dim=1)                # [B, 197, C]
            x = x + model.visual.positional_embedding.to(x.dtype)
            x = model.visual.ln_pre(x)
            token_out = model.visual.transformer(x)[-1]          # [B, 197, 768]
            feat_tokens = token_out[:, 1:, :]                    # [B, 196, 768]
            print(f"Transformer token features shape: {feat_tokens.shape}")

        # --- Project token embeddings to 512D space ---
        feat_tokens_proj = feat_tokens @ model.visual.proj       # [B, 196, 512]
        print(f"Projected feat_tokens shape: {feat_tokens_proj.shape}")

        # --- Compute similarity map ---
        text_dir = text_features[target_idx].unsqueeze(1)        # [512, 1]
        print(f"text_dir shape: {text_dir.shape}")
        sim = torch.matmul(feat_tokens_proj, text_dir).squeeze(-1)  # [B, 196]
        print(f"Similarity map shape: {sim.shape}")

        sim_min = sim.min(dim=1, keepdim=True)[0]
        sim_max = sim.max(dim=1, keepdim=True)[0]
        sim = (sim - sim_min) / (sim_max - sim_min + 1e-5)
        sim_map = sim.view(1, 14, 14)

        heatmap = torch.nn.functional.interpolate(
            sim_map.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)  # [1, H, W]
        print(f"Heatmap shape: {heatmap.shape}")

        # --- Final loss: confuse explanation but preserve class ---
        class_sim = (image_feat @ text_features.T)[0, target_idx]  # scalar
        print(f"Step {step+1} - Class similarity: {class_sim.item():.4f}")

        loss = heatmap.mean() - 0.3 * class_sim
        print(f"Step {step+1} - Loss: {loss.item():.4f}")

        (-loss).backward()
        optimizer.step()

        with torch.no_grad():
            adv_image.clamp_(0, 1)

        adv_image = adv_image.detach().requires_grad_(True)

    return adv_image.detach()



try:
    target_idx = all_texts.index("bench")
except:
    target_idx = 0

image_adv = generate_adversarial_image(image, model, text_features, target_idx)

# --- Step 3: Visualize similarity maps (clean vs adversarial) ---
# --- Utility for 2-panel comparison ---
def compare_similarity_maps(map1, map2, label1, label2, cv2_img, title):
    def blend_map(sim_map, img):
        vis = (sim_map * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        blend = img * 0.4 + vis * 0.6
        return cv2.cvtColor(blend.astype('uint8'), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.suptitle(title, fontsize=14)

    plt.subplot(1, 2, 1)
    plt.imshow(blend_map(map1, cv2_img))
    plt.title(label1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blend_map(map2, cv2_img))
    plt.title(label2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Step 3: Visualize Clean vs Adversarial ---
with torch.no_grad():
    image_features_clean = model.encode_image(image)
    image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)
    sim_map_clean = clip.get_similarity_map((image_features_clean @ text_features.T)[:, 1:, :], cv2_img.shape[:2])

    image_features_adv = model.encode_image(image_adv)
    image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)
    sim_map_adv = clip.get_similarity_map((image_features_adv @ text_features.T)[:, 1:, :], cv2_img.shape[:2])

    for n in range(sim_map_clean.shape[-1]):
        label = all_texts[n]
        if label not in target_texts:
            continue
        map_clean = sim_map_clean[0, :, :, n].cpu().numpy()
        map_adv = sim_map_adv[0, :, :, n].cpu().numpy()
        compare_similarity_maps(map_clean, map_adv, "Clean", "Adversarial", cv2_img, title=f"Similarity Maps - {label}")

# --- Step 4: Adversarial + Surgery ---
model_surgery, _ = clip.load("CS-ViT-B/16", device=device)
model_surgery.eval()

with torch.no_grad():
    image_features_adv = model_surgery.encode_image(image_adv)
    image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)

    sim_map_surgery = clip.get_similarity_map(
        clip.clip_feature_surgery(image_features_adv, text_features)[:, 1:, :],
        cv2_img.shape[:2]
    )

    for n in range(sim_map_surgery.shape[-1]):
        label = all_texts[n]
        if label not in target_texts:
            continue
        map_adv = sim_map_adv[0, :, :, n].cpu().numpy()
        map_surgery = sim_map_surgery[0, :, :, n].cpu().numpy()
        compare_similarity_maps(map_adv, map_surgery, "Adversarial", "Adversarial + Surgery", cv2_img, title=f"Surgery Maps - {label}")

# --- Step 5: Compare original / adversarial / perturbation ---
delta = (image_adv - image).squeeze().cpu()
perturb = (delta.permute(1, 2, 0) * 50 + 0.5).clamp(0, 1).numpy()
adv_img = image_adv.squeeze().cpu().permute(1, 2, 0).numpy()
orig_img = np.asarray(pil_img) / 255.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(orig_img)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(adv_img)
axes[1].set_title("Adversarial")
axes[1].axis('off')

axes[2].imshow(perturb)
axes[2].set_title("Perturbation x50")
axes[2].axis('off')

plt.tight_layout()
plt.show()
