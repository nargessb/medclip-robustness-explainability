# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 21:38:18 2025

@author: narges.babadi1
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from transformers import AutoTokenizer
from medclip import MedCLIPModel, MedCLIPVisionModelViT

# ========== Setup ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
BICUBIC = InterpolationMode.BICUBIC

# Preprocessing (match MedCLIP's expected format)
preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

# ========== Load Image ==========
pil_img = Image.open("demo.jpg").convert("RGB")
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)

# ========== Load MedCLIP ==========
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model = model.to(device)
model.eval()

# ========== Texts ==========
all_texts = ['cardiomegaly', 'pneumothorax', 'pleural effusion', 'lung opacity', 'consolidation']
target_texts = ['pleural effusion']
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
inputs = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt").to(device)
# ========== Forward Pass ==========
with torch.no_grad():
    # Encode image using ViT backbone and get patch tokens
    vision_model = model.vision_model

    # Get patch-level features (excluding CLS token)
    patch_features = vision_model(image, project=True, return_patch_tokens=True)
    patch_features = patch_features[:, 1:, :]  # remove CLS → [B, P, D]
    patch_features = F.normalize(patch_features, dim=-1)

    # Encode text
    text_features = model.encode_text(input_ids=inputs['input_ids'],
                                      attention_mask=inputs['attention_mask'])
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity [B, P, N] between each patch and each text
    similarity = patch_features @ text_features.T

    # Swin-Tiny + 224x224 → 48 patches (6 rows × 8 cols)
    H, W = 6, 8
    num_patches = similarity.shape[1]
    assert H * W == num_patches, f"Expected {H}x{W}={H*W} patches, got {num_patches}"

    # ========== Visualization ==========
    for b in range(similarity.shape[0]):
        for n in range(similarity.shape[-1]):
            if all_texts[n] not in target_texts:
                continue

            # Get patch-wise similarity map and reshape
            sim_map = similarity[b, :, n].reshape(H, W).cpu().numpy()
            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-6)
            sim_map = cv2.resize(sim_map, (cv2_img.shape[1], cv2_img.shape[0]))
            heatmap = cv2.applyColorMap((sim_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis = (0.5 * cv2_img + 0.5 * heatmap).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            print(f"MedCLIP Visual Explanation → {all_texts[n]}")
            plt.imshow(vis)
            plt.axis('off')
            plt.show()

import torch.nn.functional as F

# ========== Patch-Level Feature Surgery for MedCLIP ==========
with torch.no_grad():
    # Step 1: Get patch-level image features
    patch_features = model.vision_model(image, project=True, return_patch_tokens=True)  # [B, P+1, D]
    patch_features = patch_features[:, 1:, :]  # remove CLS → [B, P, D]
    patch_features = F.normalize(patch_features, dim=-1)  # L2 normalize

    # Step 2: Get text features
    text_features = model.encode_text(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    text_features = F.normalize(text_features, dim=-1)  # [N, D]

    # Step 3: Perform surgery on patch features (center + debias)
    # a. Centering (mean removal)
    patch_mean = patch_features.mean(dim=1, keepdim=True)  # [B, 1, D]
    patch_features_centered = patch_features - patch_mean

    # b. Projection surgery (if any): here, simply recompute similarity
    similarity = patch_features_centered @ text_features.T  # [B, P, N]

    # Step 4: Reshape & visualize
    H, W = 6, 8  # for Swin-Tiny with 224x224
    for b in range(similarity.shape[0]):
        for n in range(similarity.shape[-1]):
            if all_texts[n] not in target_texts:
                continue

            sim_map = similarity[b, :, n].reshape(H, W).cpu().numpy()
            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-6)
            sim_map = cv2.resize(sim_map, (cv2_img.shape[1], cv2_img.shape[0]))
            heatmap = cv2.applyColorMap((sim_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis = (0.5 * cv2_img + 0.5 * heatmap).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            print(f"MedCLIP Surgery Visual Explanation → {all_texts[n]}")
            plt.imshow(vis)
            plt.axis('off')
            plt.show()


# import torch.nn.functional as F
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# --- SAM Setup ---
# from segment_anything import sam_model_registry, SamPredictor

# sam_checkpoint = "F:/MedCLIP/MedCLIP-main/sam_vit_h_4b8939.pth"
# model_type = "vit_b"  # or "vit_l"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device)
# predictor = SamPredictor(sam)
# predictor.set_image(np.array(pil_img))

# # --- MedCLIP Surgery + SAM Inference ---
# with torch.no_grad():
#     # ========== Patch-level Image Features ==========
#     patch_features = model.vision_model(image, project=True, return_patch_tokens=True)  # [B, P+1, D]
#     patch_features = patch_features[:, 1:, :]  # remove CLS
#     patch_features = F.normalize(patch_features, dim=-1)  # [B, P, D]

#     # ========== Text Features ==========
#     text_features = model.encode_text(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask']
#     )
#     text_features = F.normalize(text_features, dim=-1)  # [N, D]

#     # ========== CLIP Surgery (Centering) ==========
#     patch_mean = patch_features.mean(dim=1, keepdim=True)  # [B, 1, D]
#     patch_features_centered = patch_features - patch_mean  # [B, P, D]
#     similarity = patch_features_centered @ text_features.T  # [B, P, N]

#     # ========== Visualization and SAM ==========
#     H, W = 6, 8  # Swin-Tiny patch layout for 224x224
#     similarity_map = similarity[0].reshape(H, W, -1).permute(2, 0, 1)  # [N, H, W]

#     for n in range(similarity_map.shape[0]):
#         if all_texts[n] not in target_texts:
#             continue

#         sim = similarity_map[n]  # [H, W]
#         sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)  # Normalize
#         sim_resized = cv2.resize(sim.cpu().numpy(), (cv2_img.shape[1], cv2_img.shape[0]))

#         # === Threshold to Points ===
#         threshold = 0.8
#         mask_thresh = sim_resized > threshold
#         ys, xs = np.where(mask_thresh)
#         if len(xs) == 0:
#             print(f"No points found for {all_texts[n]} at threshold {threshold}")
#             continue
#         points = np.stack([xs, ys], axis=1)
#         labels = np.ones(len(points))
#         max_points = 10  # or 20
#         if len(points) > max_points:
#             idx = np.random.choice(len(points), max_points, replace=False)
#             points = points[idx]
#             labels = labels[idx]


#         # === SAM Mask Prediction ===
#         masks, scores, logits = predictor.predict(
#             point_coords=points,
#             point_labels=labels.astype(int),
#             multimask_output=True
#         )
#         mask = masks[np.argmax(scores)]
#         mask = mask.astype('uint8')

#         # === Overlay for Visualization ===
#         vis = cv2_img.copy()
#         vis[mask > 0] = vis[mask > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
#         for i, (x, y) in enumerate(points):
#             cv2.circle(vis, (x, y), 3, (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)

#         vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
#         print(f"SAM Guided by MedCLIP Surgery → {all_texts[n]}")
#         plt.imshow(vis)
#         plt.axis('off')
#         plt.show()