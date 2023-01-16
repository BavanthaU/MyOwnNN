import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
print(words[:10])

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()

# p => probability of a char represented by index being the next value given the first char tensors => Since the
# tensor in this example has a rank of 27(in other words, it is an array of arrays), its elements are vectors(arrays)
# and not scalars(a single value). N[0] will out put the first tensor array
p = N[0].float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(itos[ix])

# averaging the counts across the raws
P = (N + 1).float()
# taking sum across dimensions
# torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor
# dim (int or tuple of ints, optional) – the dimension or dimensions to reduce. If None, all dimensions are reduced.
# 0 to truncate raws - sum through columns as a raw ( 0 - first dimension)
# 1 to truncate columns - sum across raws as a columns ( 1 - second dimension)


# P.sum(1, keepdim=True).shape => torch.Size([27, 1])
# so division
# 27, 27
# 27,  1 => first raw will be divided by the sum of the count in first raw => check the matching dimensions
# dim (int or tuple of ints, optional) – the dimension or
# dimensions to reduce. If None, all dimensions are reduced.
P /= P.sum(1, keepdims=True)

for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)

log_likelihood = 0.0
n = 0

# for w in words:
for w in ["bavantha"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll / n}')


