# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:25:57 2025

@author: narges.babadi1
"""


# MedCLIP Surgery under Adversarial Attack
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer
from medclip import MedCLIPModel  # assumes you have MedCLIP installed
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from transformers import AutoTokenizer
import torch.nn.functional as F
import medclip
# with open("medclip_prompts.txt", "r", encoding="utf-8") as f:
#     prompts = [line.strip() for line in f if line.strip()]

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "demo.jpg"  # change as needed
texts = [
    "focal consolidation",
    "pleural effusion",
    "pneumothorax",
    "bilateral nodular opacities",
    "nipple shadows",
    "normal cardiomediastinal silhouette",
    "surgical clips in left lung",
    "normal upper abdomen",
    "chronic rib deformity",
    "posterior rib fracture",
    "no acute cardiopulmonary process"
]
# texts= prompts
target_text = "chronic rib deformity"
# target_text = texts[5]
target_idx = texts.index(target_text)

# --- Load MedCLIP model ---
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# --- Preprocessing ---
preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.5]*3, [0.5]*3)
])

pil_img = Image.open(image_path).convert("RGB")
image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
# --- Encode text ---
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
text_features = model.encode_text(inputs["input_ids"], inputs["attention_mask"])
text_features = torch.nn.functional.normalize(text_features, dim=-1).detach()

# --- Clean patch similarity ---
with torch.no_grad():
    patch_features = model.vision_model(image_tensor, project=True, return_patch_tokens=True)[:, 1:, :]
    patch_features = torch.nn.functional.normalize(patch_features, dim=-1)
    patch_mean = patch_features.mean(dim=1, keepdim=True)
    patch_features_centered = patch_features - patch_mean
    similarity_clean = patch_features_centered @ text_features.T  # B, P, N

# --- Adversarial Attack ---
def generate_adversarial_image(image_tensor, model, text_features, target_idx, steps=10, epsilon=0.01):
    adv_image = image_tensor.clone().detach().to(device)
    adv_image.requires_grad = True
    optimizer = torch.optim.SGD([adv_image], lr=epsilon)

    for _ in range(steps):
        # Reinitialize optimizer in each step to avoid accumulation issues
        # optimizer = torch.optim.SGD([adv_image], lr=epsilon)
        optimizer.zero_grad()

        # Forward pass - CLS token only
        vision_out = model.vision_model(adv_image, project=True, return_patch_tokens=False)  # [B, D]
        image_feat = torch.nn.functional.normalize(vision_out, dim=-1)  # [B, D]

        # Similarity
        sim = image_feat @ text_features.T  # [B, num_texts]

        # Loss: minimize similarity to target class
        loss = -sim[0, target_idx].sum()
        loss.backward()  # No retain_graph needed with full detach

        optimizer.step()

        with torch.no_grad():
            adv_image.clamp_(0, 1)  # Keep image values valid
            adv_image = adv_image.detach()
            adv_image.requires_grad = True

    return adv_image.detach()

# one target class to perturb (e.g., "bench")
try:
    target_idx_adv = texts.index("no acute cardiopulmonary process")
except:
    target_idx_adv = 0
image_adv = generate_adversarial_image(image_tensor, model, text_features, target_idx_adv)

# --- Surgery on Adversarial ---
with torch.no_grad():
    patch_features_adv = model.vision_model(image_adv, project=True, return_patch_tokens=True)[:, 1:, :]
    patch_features_adv = torch.nn.functional.normalize(patch_features_adv, dim=-1)
    patch_mean_adv = patch_features_adv.mean(dim=1, keepdim=True)
    patch_features_adv_centered = patch_features_adv - patch_mean_adv
    similarity_adv = patch_features_adv_centered @ text_features.T

# --- Visualization helper ---
# --- Visualization helper ---
def show_similarity_map(sim, label, base_img, phase):
    sim = sim.cpu().numpy()
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)  # Normalize
    patch_count = sim.shape[0]

    # Infer patch grid shape (e.g., Swin-Tiny is 6x8 for 224x224)
    if patch_count == 48:
        H, W = 6, 8
    elif patch_count == 49:
        H, W = 7, 7
    elif patch_count == 196:
        H, W = 14, 14
    else:
        raise ValueError(f"Unexpected patch count: {patch_count}")

    sim = (sim * 255).astype(np.uint8).reshape(H, W)
    sim = cv2.resize(sim, (base_img.shape[1], base_img.shape[0]))
    heatmap = cv2.applyColorMap(sim, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base_img, 0.4, heatmap, 0.6, 0)
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} ({phase})")
    plt.axis("off")
    plt.show()

# Helper to reshape similarity to spatial heatmap
# Utility: Convert flattened similarity vector into heatmap
def similarity_to_heatmap(sim_patch, grid_hw, img_size):
    sim = sim_patch.cpu().numpy()
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-5)
    sim = sim.reshape(grid_hw)
    sim = cv2.resize(sim, (img_size[1], img_size[0]))  # Note: cv2 uses (W, H)
    return sim

# Utility: 2-panel heatmap comparison
def compare_similarity_maps(map1, map2, label1, label2, cv2_img, title):
    def blend_map(sim_map, img):
        vis = (sim_map * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(img, 0.4, vis, 0.6, 0)
        return cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)

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

# Get image size (H, W)
grid_hw = (6, 8)  # For Swin-Tiny patch layout from 224x224 input (adjust if needed)
img_size = cv2_img.shape[:2]  # (H, W) from your original image

# Infer patch layout based on token count
patch_count = similarity_clean.shape[1]
if patch_count == 48:
    grid_hw = (6, 8)
elif patch_count == 49:
    grid_hw = (7, 7)
elif patch_count == 196:
    grid_hw = (14, 14)
else:
    raise ValueError(f"Unexpected patch count: {patch_count}")




# --- Clean (no surgery) ---
with torch.no_grad():
    patch_features_clean = model.vision_model(image_tensor, project=True, return_patch_tokens=True)[:, 1:, :]
    patch_features_clean = torch.nn.functional.normalize(patch_features_clean, dim=-1)
    similarity_clean = patch_features_clean @ text_features.T  # [B, P, N]

# --- Adversarial (no surgery) ---
with torch.no_grad():
    patch_features_adv = model.vision_model(image_adv, project=True, return_patch_tokens=True)[:, 1:, :]
    patch_features_adv = torch.nn.functional.normalize(patch_features_adv, dim=-1)
    similarity_adv = patch_features_adv @ text_features.T  # [B, P, N]

from medclip.build_model import build_model

# --- Step 2: Extract state_dict ---
state_dict = model.state_dict()

# --- Step 3: Use it to build MedCLIPSurgery ---
model_surgery = build_model("MCS-MedCLIP", state_dict).to(device).eval()

# --- Surgery on Clean ---
with torch.no_grad():
    patch_features_clean_surgery = model_surgery.visual(image_tensor, project=True, return_patch_tokens=True)[:, 1:, :]
    patch_features_clean_surgery = torch.nn.functional.normalize(patch_features_clean_surgery, dim=-1)
    patch_mean_clean = patch_features_clean_surgery.mean(dim=1, keepdim=True)
    patch_features_clean_centered = patch_features_clean_surgery - patch_mean_clean
    similarity_clean_surgery = patch_features_clean_centered @ text_features.T  # [B, P, N]

# --- Surgery on Adversarial ---
with torch.no_grad():
    patch_features_adv_surgery = model_surgery.visual(image_adv, project=True, return_patch_tokens=True)[:, 1:, :]
    patch_features_adv_surgery = torch.nn.functional.normalize(patch_features_adv_surgery, dim=-1)
    patch_mean_adv_surgery = patch_features_adv_surgery.mean(dim=1, keepdim=True)
    patch_features_adv_centered = patch_features_adv_surgery - patch_mean_adv_surgery
    similarity_adv_surgery = patch_features_adv_centered @ text_features.T  # [B, P, N]

# --- Display ---
for i, label in enumerate(texts):
    if label != target_text:
        continue

    clean_map = similarity_to_heatmap(similarity_clean[0, :, i], grid_hw, img_size)
    adv_map = similarity_to_heatmap(similarity_adv[0, :, i], grid_hw, img_size)
    clean_surg_map = similarity_to_heatmap(similarity_clean_surgery[0, :, i], grid_hw, img_size)
    adv_surg_map = similarity_to_heatmap(similarity_adv_surgery[0, :, i], grid_hw, img_size)

    # 1. MedCLIP (no surgery) Clean vs Attacked
    compare_similarity_maps(
        clean_map,
        adv_map,
        "Clean", "Adversarial",
        cv2_img,
        title=f"MedCLIP No Surgery: Clean vs Attacked"
    )

    # 2. MedCLIP (with surgery) Clean vs Attacked
    compare_similarity_maps(
        clean_surg_map,
        adv_surg_map,
        "Clean + Surgery", "Adversarial + Surgery",
        cv2_img,
        title=f" MedCLIP Surgery: Clean vs Attacked"
    )

    # 3. MedCLIP Clean (No Surgery vs Surgery)
    compare_similarity_maps(
        clean_map,
        clean_surg_map,
        "No Surgery", "With Surgery",
        cv2_img,
        title=f"Clean: No Surgery vs Surgery"
    )

    # 4. MedCLIP Adversarial (No Surgery vs Surgery)
    compare_similarity_maps(
        adv_map,
        adv_surg_map,
        "No Surgery", "With Surgery",
        cv2_img,
        title=f"Adversarial: No Surgery vs Surgery"
    )


# --- Perturbation ---
delta = (image_adv - image_tensor).squeeze().cpu()
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
