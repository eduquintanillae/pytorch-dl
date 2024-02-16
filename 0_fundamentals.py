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