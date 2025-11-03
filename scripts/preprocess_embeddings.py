from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, os, json, numpy as np
from tqdm import tqdm
from models.style_extractor import StyleExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
proc = CLIPProcessor.from_pretrained(clip_name)
style_extractor = StyleExtractor(device)

# in_json  = "openfake-annotation/datasets/fake_balanced_filtered/metadata.json"
# out_path = "openfake-annotation/datasets/fake_balanced_filtered/cache/fused_embeddings.npz"

in_json  = "openfake-annotation/datasets/combined/metadata.json"
out_path = "openfake-annotation/datasets/combined/cache/fused_embeddings.npz"

with open(in_json, "r", encoding="utf-8") as f:
    data = json.load(f)

img_embs, txt_embs, style_embs, labels, clusters, sims = [], [], [], [], [], []

for sample in tqdm(data):
    img_path = sample["path"]
    if not os.path.exists(img_path):
        img_path = os.path.join("openfake-annotation", sample["path"])

    img = Image.open(img_path).convert("RGB")

    img_inputs = proc(images=img, return_tensors="pt").to(device)
    txt_inputs = proc(
        text=[sample["caption"]],
        return_tensors="pt",
        padding=True,
        truncation=True,
        # max_length=77  
    ).to(device)

    with torch.no_grad():
        img_emb = clip.get_image_features(**img_inputs).cpu().numpy()
        txt_emb = clip.get_text_features(**txt_inputs).cpu().numpy()

    style_vec = style_extractor(np.array(img))[None, :]
    img_embs.append(img_emb)
    txt_embs.append(txt_emb)
    style_embs.append(style_vec)
    labels.append(1 if sample["true_label"] == "real" else 0)
    clusters.append(sample.get("cluster_id_style", -1))
    sims.append(sample.get("similarity", 0.0))

os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(
    out_path,
    image=np.vstack(img_embs),
    text=np.vstack(txt_embs),
    style=np.vstack(style_embs),
    label=np.array(labels),
    cluster=np.array(clusters),
    similarity=np.array(sims),
)
print(f" Saved cached embeddings to {out_path}")
