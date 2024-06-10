from PIL import Image
import requests
import glob

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

categories = [
    "a t-shirt", "a photo of a city", "a photo of the statue of liberty",
    "photo of a receipt", "photo of a receipt of a burger restaurant",
    "museum ticket", "museum poster", "something else"
]

with open("README.md", "w") as file:
  file.write("# Examples for labels computed using CLIP embeddings\n\n")
  file.write("Paper: https://arxiv.org/pdf/2103.00020\n\n")
  file.write(f"Image|{'|'.join(categories)}\n")
  file.write(f"---|{'|'.join(['---' for c in categories])}\n")

  for fn in glob.glob("./data/*.jpg"):
    image = Image.open(fn)

    inputs = processor(text=categories,
                       images=image,
                       return_tensors="pt",
                       padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1)  # we can take the softmax to get the label probabilities

    file.write(f"![img]({fn})")
    for p in probs.tolist()[0]:
      file.write(f"|{p:.3f}")
    file.write("\n")
