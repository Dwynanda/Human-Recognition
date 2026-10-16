import cv2
import torch
import torchreid
from PIL import Image
import torch.nn.functional as F

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model.eval()

transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128
)[1]

from PIL import Image

gambar = cv2.imread("./img/1.jpg")

# if gambar is not None:
#     gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
#     gambar = Image.fromarray(gambar)  # NumPy → PIL Image
#     gambar = transform(gambar).unsqueeze(0)

#     with torch.no_grad():
#         embedding = model(gambar)

#     print("Ukuran embedding:", embedding.shape)
#     print("Embedding:", embedding[0][:10])
# else:
#     print("Foto tidak ditemukan!")

####

gambar1 = cv2.imread("./img/1.jpg")
gambar2 = cv2.imread("./img/2.jpg")

def ambil_embedding(gambar):
    gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
    gambar = Image.fromarray(gambar)
    gambar = transform(gambar).unsqueeze(0)

    with torch.no_grad():
        embedding = model(gambar)

    return embedding

emb1 = ambil_embedding(gambar1)
emb2 = ambil_embedding(gambar2)

similarity = F.cosine_similarity(emb1, emb2).item()

print("Similarity:", similarity)