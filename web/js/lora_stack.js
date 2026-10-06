// BSAI LoRA 集 —— 按 lora_count 动态显示/隐藏 LoRA 行
// lora_count 默认 3：只显示前 3 行，其余隐藏；输入 4 即显示 4 行。
// 隐藏实现（与 Easy-Use/Danbooru-Gallery 同款）：
//   widget.hidden = true  → LiteGraph 绘制时跳过
//   widget.type = "bsaiHidden" → 前端其他 UI 模式也跳过
//   widget.computeSize = () => [0, -4] → 不占节点高度
// 注意：本文件位于 web/js/ 子目录，相对 app.js 需三级回退。
import { app } from "../../../scripts/app.js";

const ROWS = 12;

function rowWidgets(node, i) {
  const names = [`lora${i}_enable`, `lora${i}_name`, `lora${i}_mode`, `lora${i}_strength`];
  return names
    .map((n) => node.widgets.find((w) => w.name === n))
    .filter(Boolean);
}

function setRowVisible(node, i, visible) {
  for (const w of rowWidgets(node, i)) {
    if (!w) continue;
    if (visible) {
      if (w.__bsaiOrig) {
        w.hidden = false;
        w.type = w.__bsaiOrig.type;
        w.computeSize = w.__bsaiOrig.computeSize;
        w.__bsaiOrig = null;
      }
    } else {
      if (!w.__bsaiOrig) {
        w.__bsaiOrig = { type: w.type, computeSize: w.computeSize };
      }
      w.hidden = true;
      w.type = "bsaiHidden";
      w.computeSize = () => [0, -4];
    }
  }
}

function applyCount(node, count) {
  count = Math.max(1, Math.min(ROWS, Math.trunc(Number(count) || 3)));
  for (let i = 1; i <= ROWS; i++) setRowVisible(node, i, i <= count);
  // 按剩余可见 widget 重算高度并重排
  const sz = node.computeSize();
  node.setSize([node.size[0], sz[1]]);
  if (node.graph) node.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "BSAI.LoraStack",
  nodeCreated(node) {
    const cls = node.comfyClass || (node.constructor && node.constructor.comfyClass) || node.type;
    if (cls !== "BSAI_Lora_Stack") return;
    const cw = node.widgets.find((w) => w.name === "lora_count");
    if (!cw) {
      console.warn("[BSAI-Lora-Stack] lora_count widget not found");
      return;
    }
    const origCallback = cw.callback;
    cw.callback = (v) => {
      applyCount(node, v);
      if (origCallback) origCallback.call(cw, v);
    };
    // 立即应用一次（此时 widgets 已就绪），setTimeout 作为兜底
    applyCount(node, cw.value ?? 3);
    setTimeout(() => applyCount(node, cw.value ?? 3), 0);
  },
});
