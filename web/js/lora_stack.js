// BSAI LoRA 集 —— 按 lora_count 动态显示/隐藏 LoRA 行
// lora_count 默认 3：只显示前 3 行，其余隐藏；输入 4 即显示 4 行。
//
// 兼容性说明（ComfyUI 0.4x / comfyui_frontend 1.55.x，Vue DOM 渲染）：
//   Vue DOM 行的可见性来源是 widgetValueStore 里的 visibility 副本，只改
//   live widget 的 type/computeSize 不会更新 DOM —— 旧实现在 1.55 上失效：
//   新建节点时 12 行全部常显、调 lora_count 无效（用户反馈"lora数量只到3个，
//   手动新增4个以上没出现"）。
//   1.55.x 官方可见性路径（参照 comfyui_frontend src/types/widgetVisibility.ts
//   与 widgetValueStore）：
//     - widget.hidden = true   —— setter 直接 applyLegacyHiddenWrite 到
//       _visibility（= store 里那份响应式对象），Vue 立即隐藏该 widget；
//     - widget.options.hidden = true —— options Proxy defineProperty 陷阱
//       同样转发到 _visibility；
//     - widget.setNodeId(node.id) —— 重新 registerWidget，把 visibility 同步进
//       widgetValueStore（appendNodeWidgetOrder 会去重，可安全重复调用）。
//   三条都写，保证新旧渲染器都生效。
//   旧版前端（LiteGraph 画布）读 widget.hidden 与 computeSize，照常生效。
//
//   注意：新前端里 nodeCreated 触发时 node.widgets 视图可能尚未物化（空数组），
//   所以本实现会轮询重试直到拿到 lora_count widget，并 hook
//   beforeRegisterNodeDef（原型级 onNodeCreated/onAdded/onConfigure）双保险。
import { app } from "../../../scripts/app.js";

const ROWS = 12;
const NODE_TYPE = "BSAI_Lora_Stack";

function isLoraNode(node) {
  if (!node) return false;
  const cls =
    node.comfyClass ||
    (node.constructor && node.constructor.comfyClass) ||
    node.type;
  return cls === NODE_TYPE;
}

// 一行 = enable + name + mode + strength + clip_strength（clip 列在 widget 列表尾部）
function rowNames(i) {
  return [
    `lora${i}_enable`,
    `lora${i}_name`,
    `lora${i}_mode`,
    `lora${i}_strength`,
    `lora${i}_clip_strength`,
  ];
}

function rowWidgets(node, i) {
  const list = (node && node.widgets) || [];
  return rowNames(i)
    .map((n) => list.find((w) => w && w.name === n))
    .filter(Boolean);
}

// 统一写可见性：hidden + options.hidden 双写（1.55.x 两者都会转发到 store），
// 再经 setNodeId 重新注册同步进 widgetValueStore。
function setWidgetHiddenCompat(node, w, hidden) {
  if (!w) return;
  const write = () => {
    try {
      w.options = w.options || {};
      w.options.hidden = hidden;
    } catch (e) {}
    try {
      w.hidden = hidden;
    } catch (e) {}
  };
  write();
  try {
    if (typeof w.setNodeId === "function") w.setNodeId(node?.id);
  } catch (e) {}
  // _visibility 已指向 store 的响应式对象，再写一次保证 Vue 立刻收到变更。
  write();
}

function setRowVisible(node, i, visible) {
  for (const w of rowWidgets(node, i)) {
    if (!w) continue;
    if (visible) {
      setWidgetHiddenCompat(node, w, false);
      if (w.__bsaiOrig) {
        try {
          w.type = w.__bsaiOrig.type;
        } catch (e) {}
        try {
          w.computeSize = w.__bsaiOrig.computeSize;
        } catch (e) {}
        w.__bsaiOrig = null;
      }
    } else {
      setWidgetHiddenCompat(node, w, true);
      // 旧前端兜底（无 setNodeId 的 LiteGraph 画布）：隐藏 type 标记 + 高度归零。
      // 新前端（1.55+）由 store 控制可见性，这两项无副作用。
      if (typeof w.setNodeId !== "function") {
        if (!w.__bsaiOrig) {
          w.__bsaiOrig = { type: w.type, computeSize: w.computeSize };
        }
        try {
          w.type = "bsaiHidden";
        } catch (e) {}
        try {
          w.computeSize = () => [0, -4];
        } catch (e) {}
      }
    }
  }
}

function applyCount(node, count) {
  if (!node) return;
  count = Math.max(1, Math.min(ROWS, Math.trunc(Number(count) || 3)));
  for (let i = 1; i <= ROWS; i++) setRowVisible(node, i, i <= count);
  // 按剩余可见 widget 重算高度并重排（新前端由 WidgetGrid 自行布局，此调用无副作用）
  try {
    const sz = node.computeSize ? node.computeSize() : null;
    if (sz && Array.isArray(node.size)) node.setSize([node.size[0], sz[1]]);
  } catch (e) {}
  try {
    if (node.graph) node.graph.setDirtyCanvas(true, true);
  } catch (e) {}
}

function findCountWidget(node) {
  const list = (node && node.widgets) || [];
  return list.find((w) => w && w.name === "lora_count") || null;
}

function scheduleReapply(node, cw) {
  [30, 80, 150, 300, 600, 1000, 2000, 3000].forEach((d) =>
    setTimeout(() => {
      try {
        applyCount(node, cw.value ?? 3);
      } catch (e) {}
    }, d)
  );
}

// 给 lora_count 的输入控件挂 input/change 监听（新前端值变更不一定会走
// widget.callback / onWidgetChanged，DOM 事件是最后一道保险）
function bindCountDomEvents(node, cw) {
  const bind = (el) => {
    if (!el || el.__bsaiLoraBound) return;
    el.__bsaiLoraBound = true;
    const h = () => {
      try {
        applyCount(node, cw.value ?? 3);
      } catch (e) {}
    };
    el.addEventListener("input", h);
    el.addEventListener("change", h);
  };
  try {
    bind(cw.element);
    if (cw.element && cw.element.querySelector) {
      cw.element.querySelectorAll("input,textarea,select").forEach(bind);
    }
  } catch (e) {}
}

function hookCountWidget(node, cw) {
  if (node.__bsaiLoraHooked) return;
  node.__bsaiLoraHooked = true;

  // 1) widget.callback（旧前端）
  const origCallback = cw.callback;
  cw.callback = (v) => {
    try {
      applyCount(node, v);
    } catch (e) {}
    if (origCallback) return origCallback.call(cw, v);
  };

  // 2) node.onWidgetChanged（部分前端版本）
  const origOnWidgetChanged = node.onWidgetChanged;
  node.onWidgetChanged = (w, v) => {
    if (w && w.name === "lora_count") {
      try {
        applyCount(node, v);
      } catch (e) {}
    }
    if (origOnWidgetChanged) return origOnWidgetChanged.call(node, w, v);
  };

  // 3) DOM 事件（新前端兜底）
  bindCountDomEvents(node, cw);

  // 4) 工作流载入时 configure 直接写 widget.value 不触发 callback，必须 hook
  // onConfigure（参照 WAS Node Suite 的做法），否则载入 lora_count>3 的存档时
  // 第 4+ 行不会出现。
  const origOnConfigure = node.onConfigure;
  node.onConfigure = (...args) => {
    let r;
    if (origOnConfigure) r = origOnConfigure.apply(node, args);
    try {
      applyCount(node, cw.value ?? 3);
    } catch (e) {}
    return r;
  };

  // 5) 立即应用 + 分时补跑（Vue store 注册与 configure 载入都是异步的）
  try {
    applyCount(node, cw.value ?? 3);
  } catch (e) {}
  scheduleReapply(node, cw);
  console.log(
    `[BSAI-Lora-Stack] 已安装行数联动（lora_count=${cw.value ?? 3}，共 ${ROWS} 槽）`
  );
}

function install(node) {
  if (!isLoraNode(node) || node.__bsaiLoraInstalled) return;
  node.__bsaiLoraInstalled = true;

  let done = false;
  const once = () => {
    if (done) return;
    const cw = findCountWidget(node);
    if (!cw) return; // widget 视图尚未物化，稍后重试
    done = true;
    hookCountWidget(node, cw);
  };

  once();
  if (!done) {
    // 新前端里 nodeCreated 触发时 node.widgets 可能为空，轮询直到拿到 count 控件。
    let tries = 0;
    const iv = setInterval(() => {
      if (done || ++tries > 60) {
        clearInterval(iv);
        return;
      }
      once();
    }, 100);
  }
}

app.registerExtension({
  name: "BSAI.LoraStack",
  nodeCreated(node) {
    install(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || nodeData.name !== NODE_TYPE) return;
    const proto = nodeType.prototype;
    if (proto.__bsaiLoraPatched) return;
    proto.__bsaiLoraPatched = true;

    const origOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function (...args) {
      const r = origOnNodeCreated
        ? origOnNodeCreated.apply(this, args)
        : undefined;
      install(this);
      return r;
    };

    const origOnAdded = proto.onAdded;
    proto.onAdded = function (...args) {
      const r = origOnAdded ? origOnAdded.apply(this, args) : undefined;
      install(this);
      return r;
    };

    const origOnConfigure = proto.onConfigure;
    proto.onConfigure = function (...args) {
      const r = origOnConfigure ? origOnConfigure.apply(this, args) : undefined;
      install(this);
      return r;
    };
  },
});
