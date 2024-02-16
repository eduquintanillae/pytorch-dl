import torch
print(torch.__version__)

scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

matrix = torch.tensor([[7, 8], [9, 10]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)

tensor_3d = torch.tensor([[[1, 2, 3], [3, 6, 9], [2,4,5]]])
print(tensor_3d)
print(tensor_3d.ndim)
print(tensor_3d.shape)

random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(random_tensor.dtype)

random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

zeros = torch.zeros(size=(2, 3))
print(zeros)
print(zeros.dtype)

ones = torch.ones(size=(2, 3))
print(ones)
print(ones.dtype)

zero_to_ten_deprecated = torch.arange(0, 10)
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None,
                                 device=None, requires_grad=False)
print(float_32_tensor)
print(float_32_tensor.dtype)
print(float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(float_16_tensor)
print(float_16_tensor.dtype)

some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Shape: {some_tensor.shape}, Dtype: {some_tensor.dtype}, Device: {some_tensor.device}")

# Basic Operations
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
print(tensor - 10)
print(torch.multiply(tensor, 10))
print(tensor + tensor)

# Matrix Multiplication
tensor = torch.tensor([1, 2, 3])
print(tensor.shape)
print(tensor*tensor)
print(torch.matmul(tensor, tensor))
print(tensor @ tensor) # not recommended

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")
print(torch.mm(tensor_A, tensor_B.T))

# Matrix Multiplication Visualization: http://matrixmultiplication.xyz/

# Linear
torch.manual_seed(42)
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# Min/Max, Sum, Mean
x = torch.arange(0, 100, 10)
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
print(torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x))

# Argmax and Argmin
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

# Change tensor datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)

tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)

tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)

# Reshaping, stacking, squeezing, unsqueezing
x = torch.arange(1., 8.)
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)
z = x.view(1, 7)
print(z, z.shape)
z[:, 0] = 5
print(z, x)
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked, x_stacked.shape)

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")