import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.fusion_module import MultimodalDetector

image_path = "data/combined/images/06525.png"  
text_prompt = "synthetic\nConsistency:\n<lighting>The overall lighting on the audience and the speaker is inconsistent with unnaturally deep shadows on one side.\n<edges>The edges around audience members are uniformly soft indicating a lack of natural sharpness.\n<resolution>The image maintains a similar level of resolution in both foreground and background which is atypical for genuine photographs.\n<shadows>Shadows do not align with the light sources' angles pointing towards errors in artificial lighting simulation.\n<material properties>The clothes and surfaces in the room reflect light in an unnaturally glossy manner that is inconsistent.\n<perspective>The spatial arrangement and size of people in the room show manipulation and lack normal perspective cues.\nAuthenticity:\n<natural imperfections>The textures on individuals' clothing and faces are overly uniform lacking real-life variation.\n<environmental interaction>The placement and interaction of individuals with their surroundings appear superficial like they were superimposed.\n<temporal consistency>The lack of nuanced emotion in audience expressions suggests difficulty in capturing subtle emotional differences in synthetic generation.\n<cultural>The expressions and emotional depth across the audience are uniformly lacking showing a disconnect from realistic human responses.\n<dynamic range>The image has been identified to have unnatural uniformity across textural details and lighting.\nFinal Assessment: The given image is synthetic based on the above analysis"
target_class = 1                                # 0=real,1=synthetic,2=tampered

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cuda")

model = MultimodalDetector().to("cuda")

for name, module in model.named_children():
    print(name, "->", type(module))


model.load_state_dict(torch.load("checkpoints/multimodal_detector.pt"))
model.eval()

# last conv layer from your vision encoder
target_layer = model.vision.encoder.layer4[-1]  # for ResNet-50

# hook to capture features & grads
features, grads = [], []
def forward_hook(module, inp, out): features.append(out)
def backward_hook(module, grad_in, grad_out): grads.append(grad_out[0])
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


logits = model(image, text_prompt)
pred_class = torch.argmax(logits, dim=1).item()
print(f"Predicted class: {pred_class}")

model.zero_grad()
logit_for_class = logits[0, target_class]
logit_for_class.backward()

grad = grads[0].mean(dim=(2, 3), keepdim=True)
cam = (grad * features[0]).sum(dim=1).squeeze().cpu().detach().numpy()
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam / cam.max()

orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
orig = cv2.resize(orig, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = np.float32(heatmap) / 255 + np.float32(orig) / 255
overlay = overlay / overlay.max()

plt.imshow(overlay)
plt.title(f"Grad-CAM for class {target_class}")
plt.axis("off")
plt.show()
