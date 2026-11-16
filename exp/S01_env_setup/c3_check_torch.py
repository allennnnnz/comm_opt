import torch
print("is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("torch", torch.__version__, "torch.version.cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version())
