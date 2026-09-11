import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


# ------------------------------------------------------------
# 1. Load pretrained ResNet-50
# ------------------------------------------------------------

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

# We only need feature extraction.
model.fc = torch.nn.Identity()

model.eval()


# ------------------------------------------------------------
# 2. Load preprocessing pipeline
# ------------------------------------------------------------

preprocess = weights.transforms()


# ------------------------------------------------------------
# 3. Load one face image
# ------------------------------------------------------------

image_path = (
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19"
    r"\face-shape-dataset\versions\2\FaceShape Dataset"
    r"\training_set\Heart\heart (1).jpg"
)

image = Image.open(image_path).convert("RGB")


# ------------------------------------------------------------
# 4. Prepare image
# ------------------------------------------------------------

input_tensor = preprocess(image)

# Add batch dimension
input_batch = input_tensor.unsqueeze(0)


# ------------------------------------------------------------
# 5. Extract embedding
# ------------------------------------------------------------

with torch.no_grad():

    embedding = model(input_batch)


# ------------------------------------------------------------
# 6. Display result
# ------------------------------------------------------------

print("=" * 70)
print("EYewear AI - VISUAL EMBEDDING TEST")
print("=" * 70)

print("\nImage:")
print(image_path)

print("\nOriginal image size:")
print(image.size)

print("\nEmbedding shape:")
print(embedding.shape)

print("\nEmbedding dimensions:")
print(embedding.shape[1])

print("\nFirst 10 embedding values:")
print(embedding[0][:10])

print("\n" + "=" * 70)
print("VISUAL EMBEDDING TEST COMPLETE")
print("=" * 70)