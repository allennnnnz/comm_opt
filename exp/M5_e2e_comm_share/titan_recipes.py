"""torchtitan 0.4.0 recipes for M5 training: Llama-3 1B / 8B architecture, random init, TP x FSDP on 8 GPUs.

Mirrors torchtitan_recipes.models.llama3.llama3_8b at tag v0.4.0 (loss, optimizer, default activation
remat, no compile), with these changes: local c4_test data and the TinyLlama tokenizer (no gated assets;
token ids < 32000 fit the fixed 128256-entry vocab, so compute and communication are unchanged),
a fixed global batch of 8 x 2048 tokens split across data-parallel ranks, and a short profiled run.

Usage: PYTHONPATH=exp/M5_e2e_comm_share torchrun --nproc-per-node 8 -m torchtitan.train \
           --module titan_recipes --config llama3_8b_tp4_dp2 --output-dir DIR
"""
from pathlib import Path

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import (
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
    TextProcessor,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]
SEQ_LEN = 2048
GLOBAL_SEQS = 8   # 8 x 2048 = 16384 tokens per training step, whatever the parallel layout
STEPS = 20        # profile the last 5 steps (16-20) after 3 profiler warmup steps
PROFILE = dict(profile_freq=STEPS, profiler_warmup=3, profiler_active=5)

C4_TEST = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="json", split="train",
        load_dataset_kwargs={"data_files": str(ROOT / "assets/c4_test/data.json")}),
    processor=TextProcessor.Config(),
    post_filters=(lambda sample: sample is not None,),
)


def _recipe(flavor: str, tp: int) -> Trainer.Config:
    dp = 8 // tp
    parallelism = ParallelismConfig(tensor_parallel_degree=tp, data_parallel_shard_degree=dp)
    model_config = build_model_config(flavor, seq_len=SEQ_LEN, parallelism=parallelism)
    tokens_per_dp_rank = GLOBAL_SEQS * SEQ_LEN // dp
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_config))),
        hf_assets_path=str(ROOT / "assets/tokenizer_tinyllama"),
        profiler=Profiler.Config(enable_profiling=True, **PROFILE),
        metrics=MetricsProcessor.Config(enable_tensorboard=False, log_freq=1),  # per-step tps for unprofiled timing
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(optimizer=OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)])),
        training=TrainingConfig(num_tokens_per_microbatch_per_dp_rank=tokens_per_dp_rank,
                                num_tokens_per_train_step=GLOBAL_SEQS * SEQ_LEN,
                                max_context_length=SEQ_LEN, steps=STEPS),
        dataloader=GrainDataLoader.Config(max_num_documents=max(16, (tokens_per_dp_rank + 511) // 512),
                                          dataset=ConcatThenSplitPackingConfig(dataset=C4_TEST)),
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        validator=None,
    )


def llama3_8b_tp8_dp1(): return _recipe("8B", 8)
def llama3_8b_tp4_dp2(): return _recipe("8B", 4)
def llama3_8b_tp2_dp4(): return _recipe("8B", 2)
def llama3_1b_tp8_dp1(): return _recipe("1B", 8)
def llama3_1b_tp4_dp2(): return _recipe("1B", 4)
def llama3_1b_tp2_dp4(): return _recipe("1B", 2)
