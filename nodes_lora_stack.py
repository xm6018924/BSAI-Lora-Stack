# -*- coding: utf-8 -*-
"""
BSAI LoRA 集（BSAI-Lora-Stack）

一个节点串联加载多条 LoRA：
  - 最多 12 路 LoRA，每路独立：启用开关 / 权重 / 模型选择 / 加载方式
  - 加载方式 standard = 权重合并（LoraLoaderModelOnly 语义，适用普通 LoRA）
  - 加载方式 viggle   = 官方 ViggleTurbo 运行时注入（y=Wx+BAx，不融合权重，
                        对 bf16 合并损失 ~30% 更新、int8 合并引入 ~4x 噪声的场景无损）
  - 节点总开关 master_enable：关闭时整体跳过（等价于节点 bypass，输出原模型）
  - 输出 MODEL，可直接接入采样器 / 继续叠加其他 LoRA 节点（含 ViggleTurboLora）
"""
import json
import os
import folder_paths
from comfy.sd import load_lora_for_models

try:
    import torch
    import torch.nn.functional as F
    import comfy.utils
    import comfy.patcher_extension
except Exception:  # 前端导入时缺依赖不致命，执行时才需要
    torch = F = comfy_utils = patcher_extension = None

LORA_MAX = 12


def _lora_names():
    try:
        names = folder_paths.get_filename_list("loras")
    except Exception:
        names = []
    return ["None"] + sorted(
        n for n in names
        if n.lower().endswith((".safetensors", ".pt", ".ckpt", ".bin"))
    )


# ---------- ViggleTurbo 官方运行时注入（unmerged）实现 ----------
# 与 ComfyUI-Qwen-Image-2.1-Prompt-Enhancer 的 viggle_turbo 官方节点同语义，
# 内置以避免外部插件依赖；wrapper key 带唯一后缀，支持同一 MODEL 叠加多个 Viggle LoRA。

def _v_lora_fwd(x, ab):
    return F.linear(F.linear(x, ab[0].to(x.dtype)), ab[1].to(x.dtype))


def _v_add_hook(mod, ab):
    return mod.register_forward_hook(lambda m, inp, out: out + _v_lora_fwd(inp[0], ab))


def _v_add_mlp_hooks(mlp, gate, up, down):
    # fused SwiGLU: gate_up = [gate_layer; proj]，out 运行在 int8/fp16 内核中会绕过钩子，
    # 因此其分支叠加在（已加 LoRA 的）gate_up 输出后的 MLP 输出上
    h = {}

    def gate_up_hook(m, inp, out):
        h["gu"] = out + torch.cat([_v_lora_fwd(inp[0], gate), _v_lora_fwd(inp[0], up)], -1)
        return h["gu"]

    def mlp_hook(m, inp, out):
        g, u = h.pop("gu").chunk(2, -1)
        return out + _v_lora_fwd(F.silu(g) * u, down)

    return [mlp.gate_up.register_forward_hook(gate_up_hook), mlp.register_forward_hook(mlp_hook)]


def _v_run_with_lora(lora, executor, *args, **kwargs):
    dm = executor.class_obj
    for ab in lora.values():
        if ab[0].device != args[0].device:
            ab[0], ab[1] = ab[0].to(args[0].device), ab[1].to(args[0].device)
    hooks = []
    for name, ab in lora.items():
        parent, _, leaf = name.rpartition(".")
        if not getattr(dm.get_submodule(parent), "fused", False):
            hooks.append(_v_add_hook(dm.get_submodule(name), ab))
        elif leaf == "out":
            hooks += _v_add_mlp_hooks(dm.get_submodule(parent),
                                      lora[parent + ".gate_layer"], lora[parent + ".proj"], ab)
    try:  # 扩散模型与其他 MODEL 输出共享，钩子不得超出本次调用
        return executor(*args, **kwargs)
    finally:
        for hk in hooks:
            hk.remove()


def _v_load(model, lora_path, strength):
    sd, meta = comfy.utils.load_torch_file(lora_path, return_metadata=True)
    cfg = json.loads((meta or {}).get("lora_adapter_metadata", "{}"))
    scale = strength * cfg.get("transformer.lora_alpha", 1) / cfg.get("transformer.r", 1)
    lora = {k.removeprefix("transformer.").removesuffix(".lora_A.weight"):
            [sd[k], sd[k.replace("lora_A", "lora_B")] * scale]
            for k in sd if k.endswith(".lora_A.weight")}
    m = model.clone()
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "viggle_turbo_lora_%d" % id(lora),
        lambda executor, *a, **kw: _v_run_with_lora(lora, executor, *a, **kw))
    return m


class BSAI_Lora_Stack:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
            "master_enable": ("BOOLEAN", {
                "default": True,
                "label_on": "启用(全部)",
                "label_off": "停用(跳过全部)",
                "tooltip": "节点总开关：关闭时跳过全部 LoRA，直接输出输入模型（等价整节点 bypass）。",
            }),
            "lora_count": ("INT", {
                "default": 3, "min": 1, "max": LORA_MAX, "step": 1,
                "tooltip": f"生效 LoRA 路数（1~{LORA_MAX}）：默认 3，需要第 4 个 LoRA 时改为 4；"
                           "前端只显示前 N 行，其余自动隐藏。",
            }),
        }
        optional = {}
        loras = _lora_names()
        for i in range(1, LORA_MAX + 1):
            optional[f"lora{i}_enable"] = ("BOOLEAN", {
                "default": i <= 3,  # 默认开启前 3 路，更多路由用户按需打开（即「自行添加」）
                "label_on": "开",
                "label_off": "关",
                "tooltip": f"第 {i} 路 LoRA 启用开关（默认前 3 路开启，需要更多路时打开对应开关即可添加）。",
            })
            optional[f"lora{i}_name"] = (loras, {
                "default": "None",
                "tooltip": f"第 {i} 路 LoRA 模型（下拉自 loras 目录，随目录刷新）。",
            })
            optional[f"lora{i}_mode"] = (["standard", "viggle"], {
                "default": "standard",
                "tooltip": "standard=权重合并（LoraLoaderModelOnly 语义）；"
                           "viggle=官方运行时注入 y=Wx+BAx（不融合权重，Viggle/蒸馏 LoRA 在 bf16/int8 下无损）。",
            })
            optional[f"lora{i}_strength"] = ("FLOAT", {
                "default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01, "round": 0.01,
                "tooltip": f"第 {i} 路 LoRA 权重（0 视为跳过；负值反向）。",
            })
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders/BSAI"
    DESCRIPTION = (
        "BSAI LoRA 集：一次串联加载最多 12 路 LoRA。"
        "每路独立「开关 + 模型 + 加载方式(standard/viggle) + 权重」，"
        "viggle 方式内置官方运行时注入语义（无需再接 ViggleTurboLora 节点）；"
        "节点级 master_enable 总开关可整体跳过；输出 MODEL 接入采样器或继续叠加 LoRA。"
    )

    def apply(self, model, master_enable, **kwargs):
        if not master_enable:
            print("[BSAI_Lora_Stack] master_enable=False → 跳过全部 LoRA，输出原模型")
            return (model,)

        cur = model
        applied = 0
        limit = int(kwargs.get("lora_count", 3) or 3)
        limit = max(1, min(LORA_MAX, limit))
        for i in range(1, limit + 1):
            en = bool(kwargs.get(f"lora{i}_enable", False))
            name = kwargs.get(f"lora{i}_name", "None") or "None"
            mode = kwargs.get(f"lora{i}_mode", "standard") or "standard"
            strength = kwargs.get(f"lora{i}_strength", 0.0) or 0.0
            if not en or name == "None" or strength == 0.0:
                continue
            try:
                lora_path = folder_paths.get_full_path("loras", name)
            except Exception:
                lora_path = None
            if not lora_path or not os.path.exists(lora_path):
                print(f"[BSAI_Lora_Stack] 警告: LoRA 文件不存在，跳过第 {i} 路: {name}")
                continue
            try:
                if mode == "viggle":
                    if _v_load is None:
                        raise RuntimeError("依赖 torch/comfy 不可用，无法使用 viggle 加载方式")
                    cur = _v_load(cur, lora_path, strength)
                    print(f"[BSAI_Lora_Stack] 已叠加 lora{i} [viggle官方注入]: {name} (strength={strength})")
                else:
                    cur, _ = load_lora_for_models(cur, None, lora_path, strength, strength)
                    print(f"[BSAI_Lora_Stack] 已叠加 lora{i} [standard合并]: {name} (strength={strength})")
                applied += 1
            except Exception as e:
                print(f"[BSAI_Lora_Stack] 警告: 第 {i} 路加载失败 {name}: {e}")

        if applied == 0:
            print("[BSAI_Lora_Stack] 未启用任何 LoRA，输出原模型")
        return (cur,)


NODE_CLASS_MAPPINGS = {
    "BSAI_Lora_Stack": BSAI_Lora_Stack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_Lora_Stack": "BSAI LoRA 集",
}
