# Training DeepDive with Multi-Turn RL

This folder contains DeepDive training utilities for a slime-based multi-turn
RL setup.

## Contents

```bash
training/
├── slime/                       # vendored upstream slime repository
├── tool_server/                 # DeepDive web tool server
├── reward_server/               # DeepDive strict binary reward server
├── rollout/                     # DeepDive slime rollout strategy
├── scripts/
│   ├── data/convert_qa_to_slime.py
│   ├── setup/prepare_env.sh
│   ├── setup/start_tool_server.sh
│   ├── setup/start_reward_server.sh
│   └── training/train_deepdive_slime.sh
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd training
bash scripts/setup/prepare_env.sh
pip install -r requirements.txt
pip install -r slime/requirements.txt
pip install -e slime
```

The vendored slime checkout targets Python 3.10+ and requires its own
dependencies, including Ray and SGLang router packages.

Edit `training/.env` with paths and API keys:

```bash
DEEPDIVE_ROOT=/path/to/DeepDive
SLIME_ROOT=${DEEPDIVE_ROOT}/training/slime
SEARCH_PROVIDER=serper
SERPER_API_KEY=...
# or SEARCH_PROVIDER=serpapi with SERP_API_KEY=...
JINA_API_KEY=...
REWARD_API_KEY=...
```

## Data Conversion

Convert DeepDive QA JSONL into the prompt format expected by slime rollouts:

```bash
python scripts/data/convert_qa_to_slime.py \
  --input /path/to/deepdive_rl.jsonl \
  --output /path/to/deepdive_rl_slime.jsonl \
  --source deepdive
```

Input rows should contain `question` and `answer`.
The converter writes DeepDive runtime data to `metadata.remote_env_info`, which
is where slime loads per-sample metadata.

## Tool Server

The tool server exposes the DeepDive web actions used by the inference
framework: `search`, `click`, and `open`.

```bash
bash scripts/setup/start_tool_server.sh
curl http://127.0.0.1:${TOOL_SERVER_PORT}/health
```

Example request:

```bash
curl -X POST http://127.0.0.1:${TOOL_SERVER_PORT}/tool \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"smoke","name":"search","arguments":{"query":"THUDM DeepDive"},"remote_env_info":{"forbidden_texts":[]}}'
```

`SEARCH_PROVIDER` can be `serper` or `serpapi`. `session_id` must be unique per
rollout because `click` opens result ids from the latest search in the same
session.

## Reward Server

The reward server provides a slime-compatible `/evaluate` endpoint implementing
DeepDive's strict binary reward design.

```bash
bash scripts/setup/start_reward_server.sh
curl http://127.0.0.1:${RM_TRAIN_PORT}/health
```

It returns `reward=1` only when both format correctness and answer correctness
are satisfied. Unfinished, malformed, or incorrect trajectories receive `0`.

## Slime Training

After starting the tool and reward services, launch training with the vendored
slime checkout:

```bash
PROMPT_DATA=/path/to/deepdive_rl_slime.jsonl \
HF_MODEL_PATH=/path/to/hf/model \
REF_MODEL_TORCH_DIST=/path/to/converted/torch_dist \
CKPT_DIR=/path/to/output \
bash scripts/training/train_deepdive_slime.sh
```

The script uses the local DeepDive rollout strategy:

- `training.rollout.deepdive_rollout.generate_with_tool`
- optional reward hook: `training.rollout.deepdive_rollout.reward_func`
- follows slime's official custom generation signature:
  `async generate(args, sample, sampling_params)`
- GRPO advantage estimation
- remote tool calls with retry and timeout controls
- `source`, `input_messages`, `label`, and `tools` prompt fields

Tune model-parallel and rollout settings through environment variables such as
`TP_SIZE`, `PP_SIZE`, `CP_SIZE`, `ROLLOUT_TP_SIZE`, `MAX_CONTEXT_LEN`,
`ROLLOUT_BATCH_SIZE`, and `N_SAMPLES_PER_PROMPT`.
