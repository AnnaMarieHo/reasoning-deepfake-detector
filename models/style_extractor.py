import numpy as np, cv2, torch
from transformers import AutoProcessor, SiglipModel
from transformers import CLIPModel, CLIPProcessor


class StyleExtractor:
    #(blur, edge, texture variance).
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).vision_model.to(device).eval()

    def get_aesthetic_emb(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # use mean-pooled vision embeddings as "style" vector
            feats = self.model(**inputs).last_hidden_state.mean(dim=1)
        return feats.cpu().numpy().squeeze()

    def handcrafted(self, image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 100, 200).mean()
        patch_var = np.var(cv2.blur(gray, (3, 3)) - gray)
        return np.array([lap_var, edges, patch_var])

    def __call__(self, image):
        a = self.get_aesthetic_emb(image)
        h = self.handcrafted(image)
        return np.concatenate([a, h])
