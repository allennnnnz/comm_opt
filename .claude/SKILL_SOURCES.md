# Installed skills: sources

Copied on 2026-11-02. To update, re-copy the same directories from upstream.

| Skills | Upstream | Commit | License |
|---|---|---|---|
| `nvshmem-*` (9), `nemo-mbridge-perf-*` (6) | https://github.com/NVIDIA/skills/tree/main/skills | 2534061 | Apache-2.0 / CC-BY-4.0 |
| `training-llms-megatron`, `deepspeed`, `pytorch-fsdp2` | https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/08-distributed-training | 773a529 | MIT |
| `systems-paper-writing`, `ml-paper-writing`, `academic-plotting` | https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/20-ml-paper-writing | 773a529 | MIT |
| `brainstorming-research-ideas` | https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/21-research-ideation | 773a529 | MIT |
| `research-paper-writing` | https://github.com/Master-cai/Research-Paper-Writing-Skills | 77e7c2c | MIT |

Notes:
- `academic-plotting`: the data-chart workflow (matplotlib) needs no key; the architecture-diagram workflow calls Gemini and needs `GEMINI_API_KEY`.
- MCP servers are configured in `../.mcp.json`. `paper-search-mcp@0.1.4` on PyPI breaks with `mcp` 2.x, hence `--with "mcp<2"`. Its Sci-Hub fallback is off by default; leave it off.
