# comm_opt

Research on GPU-to-GPU communication optimization, from measurement to paper. The topic is not chosen yet; it will come out of characterizing this machine with nccl-tests.

- Research rules (with sources): [docs/research_rules.md](docs/research_rules.md)
- Current status and next steps: [docs/status.md](docs/status.md)
- Decision log: [docs/decisions.md](docs/decisions.md)
- Folder layout and working conventions: [CLAUDE.md](CLAUDE.md)

## Reproducing the measurements

Requirements: CUDA 13.x, NCCL 2.32.3 (system install), Open MPI 4.1, 8× A100-SXM4.

```bash
# build nccl-tests at the pinned commit
git clone https://github.com/NVIDIA/nccl-tests.git third_party/nccl-tests
git -C third_party/nccl-tests checkout afd59ab
make -C third_party/nccl-tests -j MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1 \
  CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# M0: baseline sweep (refuses to run while other GPU processes exist)
exp/M0_baseline/run.sh
```

Raw results land in `results/<experiment>/<timestamp>/`, each with an `env.txt` snapshot of the machine.
