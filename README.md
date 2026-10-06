# BSAI-Lora-Stack（BSAI LoRA 集）

> 中文 | [English](README.en.md)

一个节点串联加载多条 LoRA，面向 Qwen-Image-2.1（及任意扩散模型）工作流。

## 功能

- **`lora_count` 路数开关（1 ~ 12，默认 3）**：节点只显示前 N 行，其余自动隐藏（节点短而整洁）；需要第 4 个 LoRA 就把 `lora_count` 改为 4，第 4 行立即出现。后端同样按该值限定生效行数，前端隐藏只是视觉，即使手动改大也不会多加载。
- **最多 12 路 LoRA**，每路独立：`loraN_enable`（开/关，默认前 3 路开启）、`loraN_name`（下拉选模型，自动扫描 `models/loras` 目录含子目录）、`loraN_mode`（加载方式）、`loraN_strength`（权重 -4.0 ~ 4.0，0 视为跳过，负值反向）。
- **加载方式 `loraN_mode`**：
  - `standard` — 权重合并（`LoraLoaderModelOnly` 语义），适用普通 LoRA（Pruna 5/8 步、bfs_head 等）。
  - `viggle` — **官方 ViggleTurbo 运行时注入**（`y = Wx + BAx`，不融合权重），对 Viggle 蒸馏 LoRA 在 bf16/int8 下无损（权重合并会丢 ~30% 更新 / 引入 ~4x 噪声），**无需再拖 ViggleTurboLora 节点**。
- **节点总开关** `master_enable`：关闭时跳过全部 LoRA，直接输出输入模型（等价整节点 bypass），方便 A/B 对比。
- **输出 MODEL**：可接采样器，也可接到 `ViggleTurboLora.model` 等节点继续叠加。

## 安装

把 `BSAI-Lora-Stack` 文件夹放入：

```
ComfyUI\custom_nodes\BSAI-Lora-Stack
```

重启 ComfyUI 后，在节点搜索框输入 `BSAI LoRA 集` 即可找到（分类：loaders/BSAI）。

## 使用

```
[UNETLoader / CheckpointLoader] --model--> [BSAI LoRA 集] --MODEL--> [采样器/后续 LoRA]
```

- 节点默认开启前 3 路 `loraN_enable`（名称默认 None，选中模型即生效）；需要更多 LoRA 时把 `lora_count` 改为对应路数（如 4），第 4 行即显示并可配置（最多 12 路）。
- `master_enable` 用于整节点一键停用。
- **Viggle 蒸馏 LoRA**：对应行 `loraN_mode` 选 `viggle` 即可获得官方语义（运行时注入），无需再拖 `ViggleTurboLora` 节点；若仍需官方节点语义叠加，本节点输出 MODEL 也可继续接入 `ViggleTurboLora`。
- 蒸馏 LoRA（Viggle/Pruna）注意：步数与 sigma 由采样器与 sigma 节点匹配，本节点只负责模型权重叠加。

## 文件

- `nodes_lora_stack.py` — 节点实现
- `__init__.py` — 注册入口
