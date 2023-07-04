import torch
from torch.nn import functional as F
import torch.nn as nn

torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# # not much efficient way  to do this
# xbow = torch.zeros((B, T, C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b, :t + 1]
#         xbow[b, t] = torch.mean(xprev, 0)

# let's use matrix multiplication to achieve the same

# torch.manual_seed(42)
# a = torch.tril(torch.ones(3,3))
# a = a / torch.sum(a, dim=1, keepdim=True)
# b = torch.randint(0,10, (3,2)).float()
# c = a @ b
# print(a)
# print(b)
# print(c)

# wei = torch.tril(torch.ones(T, T))
# wei = wei / torch.sum(wei, dim=1, keepdim=True)
# xbow2 = wei @ x  # pytorch will add missing batch dimension and perform the multiplication parallel
# print(torch.allclose(xbow, xbow2))
#
# # version 3
# tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))  # weight here begin with 0, like an interation strength
# # : how much do you want to aggregate form the previous token
# wei = wei.masked_fill(tril == 0, float('-inf'))  # -inf will block future token to be aggregated
# wei = F.softmax(wei, dim=-1)
# xbow3 = wei @ x
# print(torch.allclose(xbow, xbow3))

# version 4 self attention
# key  = What does the text contain?
# query = What am I looking for?
# value = actual information in the text embedded (here is what i communicate given the mask)
# query and key are used to compute the attention weights

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B,T, 16)
q = query(x)  # (B,T, 16)
v = value(x)  # (B,T, 16)

wei = q @ k.transpose(-1, -2)  # (B,T, 16) @ (B, 16, T) -> (B,T,T)
tri = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tri == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)  # (B,T, 16)
out = wei @ v  # (B,T,T) @ (B,T,16) -> (B,T,16)
