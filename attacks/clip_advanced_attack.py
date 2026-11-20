# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:30:06 2025

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
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

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
with torch.no_grad():
    # CLIP architecture surgery acts on the image encoder
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Prompt ensemble for text features with normalization
    # Right after encoding text features
    text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)  # shape: [59, 512]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Fix the shape to [D, N] => [N, D] for projection math
    text_features = text_features.to(device).float()  # ensure type and device

    print("Text feature dim:", text_features.shape)  # should now be [59, 768]

def clip_global_surgery(image_features, text_features):
    """
    Standard projection: f_surgery = T T^T f
    """
    return text_features @ text_features.T @ image_features.T  # [N x B] => transpose later
    # Transposed back to [B x D] by .T

def class_conditioned_surgery(image_features, text_features, target_idx):
    """
    Project image features onto target class t_k only
    """
    t_k = text_features[target_idx].unsqueeze(0)  # [1, D]
    t_k = t_k / t_k.norm(dim=-1, keepdim=True)
    return (image_features @ t_k.T) * t_k  # [B x 1] * [1 x D] = [B x D]

def topk_class_surgery(image_features, text_features, sim_matrix, k=3):
    """
    Project image features onto top-K text directions based on current similarity
    """
    topk_indices = torch.topk(sim_matrix, k=k, dim=-1).indices.squeeze()  # [K]
    topk_t = text_features[topk_indices]  # [K x D]
    topk_t = topk_t / topk_t.norm(dim=-1, keepdim=True)
    return (image_features @ topk_t.T) @ topk_t  # [B x D]
   
# --- Step 2: Generate adversarial image (perturbation to fool explanation) ---
def generate_adversarial_image(image, model, text_features, target_idx, steps=10, epsilon=0.01):
    adv_image = image.clone().detach().to(device)
    adv_image.requires_grad_(True)

    optimizer = torch.optim.SGD([adv_image], lr=epsilon)

    for _ in range(steps):
        optimizer.zero_grad()
        image_feat = model.encode_image(adv_image)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        sim = image_feat @ text_features.T
        loss = -sim[0, target_idx].sum()
        # reduce similarity to target class (e.g. bench)
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

# --- Step 4: Surgery Ablation Study ---
def class_conditioned_surgery(f_img, T, idx):
    t_k = T[idx].unsqueeze(0)  # [1, D]
    t_k = t_k / t_k.norm(dim=-1, keepdim=True)
    return (f_img @ t_k.T) * t_k

def topk_class_surgery(f_img, T, k=3):
    """
    Project image features onto the subspace of top-K text embeddings.
    f_img: [1, D], T: [N, D]
    """
    sims = f_img @ T.T  # [1 x N]
    topk = torch.topk(sims, k=k, dim=-1).indices.squeeze(0)  # [K]
    T_top = T[topk]  # [K x D]
    T_top = T_top / T_top.norm(dim=-1, keepdim=True)

    coeffs = f_img @ T_top.T       # [1 x K]
    f_proj = coeffs @ T_top        # [1 x D]
    return f_proj




# Make sure image_features_adv is shape [1, D]
with torch.no_grad():
    image_features_adv = model.encode_image(image_adv)
    image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)
    f_img = image_features_adv[:, 0, :]  # extract CLS token only → shape [1, D]
  # <--- Ensures [1, D]


    # Step 2: Run three types of surgery
    f_global = (f_img @ text_features.T) @ text_features
    f_class = class_conditioned_surgery(f_img, text_features, target_idx)
    f_topk = topk_class_surgery(f_img, text_features, k=3)


    # Normalize
    f_global = f_global / f_global.norm(dim=-1, keepdim=True)
    f_class = f_class / f_class.norm(dim=-1, keepdim=True)
    f_topk = f_topk / f_topk.norm(dim=-1, keepdim=True)

    # Step 3: Compute similarity maps
    sim_orig = f_img @ text_features.T
    sim_global = f_global @ text_features.T
    sim_class = f_class @ text_features.T
    sim_topk = f_topk @ text_features.T
# Step 1: Get per-patch spatial features (e.g. from CLIP's transformer)
with torch.no_grad():
    image_features_clean = model.encode_image(image)        # [1, 197, 512]
    image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)

    image_features_adv = model.encode_image(image_adv)
    image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)

def visualize_ablations(cv2_img, sim_dict, labels, target_texts):
    for i, label in enumerate(labels):
        if label not in target_texts:
            continue
        
        fig, axes = plt.subplots(1, len(sim_dict), figsize=(20, 5))
        fig.suptitle(f"Surgery Comparison - {label}", fontsize=16)
        
        for j, (name, sim) in enumerate(sim_dict.items()):
            try:
                sim_map = sim[0, i].detach().cpu().numpy()

                # Normalize safely
                sim_map = sim_map - sim_map.min()
                sim_map = sim_map / (sim_map.max() + 1e-6)
                sim_map_uint8 = (sim_map * 255).astype(np.uint8)

                # Ensure it's 2D
                if sim_map_uint8.ndim != 2:
                    sim_map_uint8 = np.squeeze(sim_map_uint8)

                # Confirm shape
                if sim_map_uint8.ndim != 2:
                    raise ValueError(f"{name}: sim_map shape not 2D after squeeze: {sim_map_uint8.shape}")

                # Final safety cast
                sim_map_uint8 = np.ascontiguousarray(sim_map_uint8)

                # Apply color map
                vis = cv2.applyColorMap(sim_map_uint8, cv2.COLORMAP_JET)

                # Blend with image
                blend = (cv2_img * 0.4 + vis * 0.6).astype(np.uint8)
                blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)

                axes[j].imshow(blend)
                axes[j].set_title(name)
                axes[j].axis('off')
            except Exception as e:
                print(f"[ERROR in {name}]:", str(e))
                axes[j].set_title(f"{name} (Error)")
                axes[j].axis('off')

        plt.tight_layout()
        plt.show()


# Prepare similarity maps for visualization
sim_dict = {
    "Adversarial": sim_orig,
    "Global Surgery": sim_global,
    "Class-Surgery": sim_class,
    "Top-K Surgery": sim_topk,
}

visualize_ablations(cv2_img, sim_dict, all_texts, target_texts)


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
