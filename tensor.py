import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(x_data, "\n")

x_ones = torch.ones_like(x_data, dtype=torch.int)

print(f"tensor: {x_ones}")
print("Shape do tensor: ", x_ones.shape)
print("Type do tensor: ", x_ones.dtype)
print("Device do tensor: ", x_ones.device)

tensor = torch.ones(3,4, dtype=torch.int)
print(tensor, "\n")

tensor[:,-1] = 0
print(tensor)

concat = torch.cat([tensor,tensor], dim=1)
print(concat)

tensor = tensor.float()
print("\nMatrix Operations")
#matrix multiplications
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)

#Element wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)
print(z1)


#Converting single elements tensors
print("\nSingle-element tensors")
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


print("\nTensor to Numpy")
t = torch.ones(5)
print("tensor = ",t)
n = t.numpy()
print("np array = ", n)

t.add_(1)
print("\ntensor = ", t)
print("np array = ", n)


print("\n Numpy to tensor")
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)


nparray = np.ones(5)
print("tensor = ", t)
print("np array = ", n)
