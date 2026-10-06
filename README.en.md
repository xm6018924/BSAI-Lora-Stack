# BSAI-Lora-Stack (BSAI LoRA Stack)

> English | [中文](README.md)

Load multiple LoRAs through a single node, built for Qwen-Image-2.1 (and any diffusion model) workflows.

## Features

- **`lora_count` switch (1 ~ 12, default 3)**: the node shows only the first N rows, the rest are automatically hidden (compact node); set it to 4 when you need a 4th LoRA and row 4 appears immediately. The backend also honors this value — hidden rows are never loaded even if you manually override widgets.
- **Up to 12 LoRA rows**, each independent: `loraN_enable` (on/off, first 3 on by default), `loraN_name` (dropdown auto-scanning `models/loras` including subfolders), `loraN_mode` (loading method), `loraN_strength` (weight -4.0 ~ 4.0, 0 = skip, negative = invert).
- **Loading method `loraN_mode`**:
  - `standard` — weight merge (`LoraLoaderModelOnly` semantics), for normal LoRAs (Pruna 5/8-step, bfs_head, etc.).
  - `viggle` — **official ViggleTurbo runtime injection** (`y = Wx + BAx`, weights not merged), lossless for Viggle distilled LoRAs on bf16/int8 (weight merging loses ~30% of the update on bf16 and adds ~4x noise on int8). **No extra ViggleTurboLora node needed.**
- **Node master switch `master_enable`**: off = skip all LoRAs and pass the input model through (equivalent to node bypass) for easy A/B testing.
- **MODEL output**: feed it to a sampler, or chain more LoRA nodes (including `ViggleTurboLora`).

## Installation

Copy the `BSAI-Lora-Stack` folder into:

```
ComfyUI\custom_nodes\BSAI-Lora-Stack
```

Restart ComfyUI, then search `BSAI LoRA 集` in the node palette (category: loaders/BSAI).

## Usage

```
[UNETLoader / CheckpointLoader] --model--> [BSAI LoRA Stack] --MODEL--> [Sampler / more LoRAs]
```

- The first 3 rows are enabled by default (`loraN_name` defaults to None — pick a model to activate). Need more LoRAs? Set `lora_count` to the desired count (e.g. 4), and row 4 appears (max 12).
- Use `master_enable` to disable the whole node in one click.
- **Viggle distilled LoRA**: set that row's `loraN_mode` to `viggle` for official runtime-injection semantics — no extra `ViggleTurboLora` node required; the MODEL output can still be chained into a `ViggleTurboLora` node if you prefer.
- Distilled LoRAs (Viggle/Pruna): steps and sigmas are matched by the sampler and sigma nodes; this node only stacks model weights.

## Files

- `nodes_lora_stack.py` — node implementation (includes the official ViggleTurbo runtime-injection logic)
- `web/js/lora_stack.js` — frontend that shows/hides rows dynamically per `lora_count`
- `__init__.py` — registration entry

## Requirements

- ComfyUI (0.38 or newer)
- PyTorch + torch.nn.functional (needed for the `viggle` runtime-injection mode)
