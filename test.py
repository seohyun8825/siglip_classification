from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
processor = AutoImageProcessor.from_pretrained('happy8825/siglip-ecva-main')
model = AutoModelForImageClassification.from_pretrained('happy8825/siglip-ecva-main')
img = Image.open('/hub_data4/seohyun/ecva/abnormal_video_classification/296_00100111_000022_0.png').convert('RGB')
inputs = processor(images=img, return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits
pred = logits.argmax(-1).item()
print(model.config.id2label[pred])