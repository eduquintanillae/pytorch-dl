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