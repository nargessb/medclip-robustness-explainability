# -*- coding: utf-8 -*-
"""
Created on Tue Jul  21 14:35:23 2025

@author: narges.babadi1
"""

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
from medclip import MedCLIPModel, MedCLIPVisionModel  # or your specific MedCLIP module
from transformers import AutoTokenizer

# Load MedCLIP
model = MedCLIPModel.from_pretrained("path_or_hf_id")  # Customize this
model = model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")  # Or your medical BERT
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.5]*3, [0.5]*3)  # MedCLIP normalization may differ—adjust if needed
])


pil_img = Image.open("demo.jpg")
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

# --- Texts ---
all_texts = ['cardiomegaly', 'pneumothorax', 'pleural effusion', 'lung opacity', 'consolidation']
target_texts = ['pleural effusion']
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
with torch.no_grad():
    # CLIP architecture surgery acts on the image encoder
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Prompt ensemble for text features with normalization
    inputs = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_features = model.encode_text(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    text_features = torch.nn.functional.normalize(text_features, dim=-1)

    print("Text feature dim:", text_features.shape)  # should now be [59, 768]

    
# --- Step 2: Generate adversarial image (perturbation to fool explanation) ---
def generate_adversarial_image(image, model, text_features, target_idx, steps=10, epsilon=0.01):
    adv_image = image.clone().detach().to(device)
    adv_image.requires_grad_(True)

    optimizer = torch.optim.SGD([adv_image], lr=epsilon)

    for _ in range(steps):
        optimizer.zero_grad()
        patch_feat = model.vision_model(adv_image, project=True, return_patch_tokens=True)
        cls_token = patch_feat[:, 0, :]  # assuming CLS is retained
        cls_token = torch.nn.functional.normalize(cls_token, dim=-1)

        sim = cls_token @ text_features.T
        loss = -sim[0, target_idx]
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            adv_image.clamp_(0, 1)

        adv_image = adv_image.detach().requires_grad_(True)

    return adv_image.detach()


# one target class to perturb (e.g., "bench")
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



# --- Step 4: Clean + Adversarial Surgery ---
model_surgery, _ = clip.load("CS-ViT-B/16", device=device)
model_surgery.eval()

with torch.no_grad():
    # Clean Surgery
    image_features_clean_surgery = model_surgery.encode_image(image)
    image_features_clean_surgery = image_features_clean_surgery / image_features_clean_surgery.norm(dim=-1, keepdim=True)
    sim_map_clean_surgery = clip.get_similarity_map(
        clip.clip_feature_surgery(image_features_clean_surgery, text_features)[:, 1:, :],
        cv2_img.shape[:2]
    )

    # Adversarial Surgery
    image_features_adv_surgery = model_surgery.encode_image(image_adv)
    image_features_adv_surgery = image_features_adv_surgery / image_features_adv_surgery.norm(dim=-1, keepdim=True)
    sim_map_adv_surgery = clip.get_similarity_map(
        clip.clip_feature_surgery(image_features_adv_surgery, text_features)[:, 1:, :],
        cv2_img.shape[:2]
    )

    # Compare Clean Surgery vs Adversarial Surgery
    for n in range(sim_map_clean_surgery.shape[-1]):
        label = all_texts[n]
        if label not in target_texts:
            continue
        map_clean = sim_map_clean_surgery[0, :, :, n].cpu().numpy()
        map_adv = sim_map_adv_surgery[0, :, :, n].cpu().numpy()
        compare_similarity_maps(
            map_clean, map_adv,
            "Clean + Surgery", "Adversarial + Surgery",
            cv2_img,
            title=f"Surgery Comparison - {label}"
        )

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
