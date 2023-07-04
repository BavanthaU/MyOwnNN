import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('MACOSX')
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

# load an Image
img = Image.open('./cat.jpg')
fig = plt.figure()
plt.imshow(img)

# resize to imagenet size
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
print(x.shape)
x = x.unsqueeze(0)  # Returns a new tensor with a dimension of size one inserted at the specified position.
print(x.shape)
