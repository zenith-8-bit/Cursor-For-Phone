from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# Load the processor and model
model_path = "HuggingFaceTB/SmolVLM-500M-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path)

# Load and preprocess the image
image = Image.open("picture.jpg").convert("RGB")

# Prepare the input
inputs = processor(images=[image], text=["<image> Describe this image."], return_tensors="pt")

# Generate the output
outputs = model.generate(**inputs)
print(processor.decode(outputs[0], skip_special_tokens=True))
