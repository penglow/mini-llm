# mini-llm — a tiny model that talks like your group chat

A small, from-scratch decoder-only Transformer (GPT), trained at the character
level on a WhatsApp group-chat export. It learns the group's turn-taking,
slang, emoji habits and each person's style — it does **not** have general
knowledge or reasoning. Pure PyTorch, runs on CPU.

This is a Windows/PyTorch reimplementation of the idea behind
[Doriandarko/texts-to-transformer](https://github.com/Doriandarko/texts-to-transformer).
That project is Apple-Silicon + MLX only and reads the macOS iMessage database,
so it can't run here — this repo keeps the same recipe (clean the chat → train a
small GPT → sample) using tools that work on Windows.

## What's here

| File              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `_chat.txt`       | The raw WhatsApp export (input).                               |
| `prepare_data.py` | Cleans the export into `data/corpus.txt` (`name: message`).    |
| `model.py`        | The GPT definition (~1.9M params by default).                  |
| `train.py`        | Trains on `data/corpus.txt`, writes `out/ckpt.pt`.            |
| `sample.py`       | Generates chat-style text from the checkpoint.                |

## Setup

```bash
pip install -r requirements.txt
```

(Python 3.14 on Windows only has CPU PyTorch wheels, so training runs on CPU.
For a GPU build, use Python 3.11–3.13 and install the CUDA wheel from
https://pytorch.org — the code auto-detects and uses CUDA if available.)

## Run it

```bash
python prepare_data.py     # _chat.txt -> data/corpus.txt
python train.py            # trains ~75 min on CPU, checkpoints to out/ckpt.pt
python sample.py                       # free-run a fake conversation
python sample.py --prompt "wes: "      # continue as a specific person
python sample.py --prompt "abdullah: " --temperature 0.9 --tokens 400
```

`train.py` saves the best checkpoint at every eval, so you can `Ctrl-C` early
and still sample from what it has learned.

## Training on an NVIDIA GPU (e.g. RTX 5070 Ti)

The RTX 50-series is Blackwell (sm_120) and needs the **CUDA 12.8** PyTorch
wheel and Python 3.11–3.13 (there is no CUDA wheel for 3.14 yet). On the GPU
machine:

```powershell
# 1. Install Python 3.12 (leave any other Python installed)
winget install -e --id Python.Python.3.12

# 2. Copy this whole folder over, then from inside it:
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install CUDA 12.8 PyTorch (cu128 is required for 50-series)
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. Confirm the GPU is seen (should print your 5070 Ti)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then train a bigger, sharper model — 16GB VRAM handles this in a few minutes:

```powershell
python prepare_data.py
python train.py --n_layer 6 --n_head 8 --n_embd 384 --block_size 256 `
                --batch_size 64 --max_iters 6000 --dropout 0.2
python sample.py --prompt "wes: "
```

`train.py` auto-detects CUDA — no code changes needed. If you hit an
out-of-memory error, lower `--batch_size` or `--block_size`.

## Cleaning

`prepare_data.py` parses the WhatsApp timestamp headers, joins multi-line
messages, and removes: media placeholders (`image omitted`, stickers, GIFs…),
system events (joins/leaves/admin/icon changes), polls, deleted messages,
links, the Meta AI bot, and the invisible Unicode marks WhatsApp injects.
Speaker names are stripped of emoji so they stay stable tokens. Result:
~78k messages / ~2.9M characters across 11 speakers.

## Knobs

Model size and training are CLI flags on `train.py`
(`--n_layer --n_head --n_embd --block_size --batch_size --max_iters`).
Bigger `--n_embd`/`--n_layer` = smarter but slower. Sampling randomness is
controlled by `--temperature` (higher = wilder) and `--top_k` on `sample.py`.
