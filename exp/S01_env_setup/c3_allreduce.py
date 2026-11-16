import os, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
x = torch.tensor([float(rank)], device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
print(f"rank {rank}/{world} sum={x.item()} expected={world*(world-1)/2}", flush=True)
assert x.item() == world * (world - 1) / 2
dist.barrier()
dist.destroy_process_group()
