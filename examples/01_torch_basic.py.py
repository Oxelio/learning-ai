import torch

print("PyTorch :", torch.__version__)
print("CUDA build :", torch.version.cuda)
print("HIP build :", torch.version.hip)
print("GPU disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nombre de GPU :", torch.cuda.device_count())
    print("GPU :", torch.cuda.get_device_name(0))

    device = torch.device("cuda")

    x = torch.rand(5, 3, device=device)

    print("\nTensor :")
    print(x)
    print("Device :", x.device)
    print("dtype :", x.dtype)
    print("shape :", x.shape)