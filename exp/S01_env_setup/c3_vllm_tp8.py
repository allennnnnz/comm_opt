import vllm
from vllm import LLM, SamplingParams
if __name__ == "__main__":
    print("vllm.__version__", vllm.__version__, flush=True)
    llm = LLM(model="Qwen/Qwen3-0.6B", tensor_parallel_size=8, max_model_len=2048, gpu_memory_utilization=0.5)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
    for o in out:
        print("TOKENS", len(o.outputs[0].token_ids))
        print("OUTPUT_TEXT", repr(o.outputs[0].text))
