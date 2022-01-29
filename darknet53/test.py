import torch
import math

# a = torch.Tensor([1, 2, 3, 4])
# b = a < 3  # mask
# print(b)
# print(a[b])
# print(b.nonzero())
# print(a[b.nonzero()])

a = torch.Tensor([[1, 2], [5, 6], [3, 1], [2, 8]])
# b = a < 3
# print(b)
# print(a[b])
b = a[:, 1] > 5
print(b)
print(a[b])
print(b.nonzero())

print(math.modf(3.4))


print(400/32)