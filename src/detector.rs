//! GroundingDINO ONNX 推理封装。
//!
//! 模型输入（由 tools/export_onnx.py 定义，文本 prompt 已在导出时烘焙进图）：
//!   - images: f32 [1, 3, H, W]
//! 模型输出：
//!   - pred_logits: f32 [1, NQ, L]   （query x token）
//!   - pred_boxes:  f32 [1, NQ, 4]   （归一化 cxcywh）
//!
//! 后处理与参考实现对齐：对 logits 取 sigmoid 并在 token 维取 max 作为 score，
//! 严格大于 box_threshold 才保留；pred_boxes 映射回原图并 clip；再跑 greedy NMS。

use anyhow::{anyhow, Context, Result};
use ndarray::Array3;
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

use crate::config::ModelConfig;
use crate::detection::{cxcywh_to_xyxy_clipped, greedy_nms, Detection};
use crate::preprocess::Preprocessed;

/// 英文标签到中文展示名的映射。未知标签保留英文原文。
fn label_zh(label: &str) -> String {
    match label.trim().trim_end_matches('.').to_ascii_lowercase().as_str() {
        "person" | "people" | "human" => "人".to_string(),
        "car" | "vehicle" => "车".to_string(),
        other => other.to_string(),
    }
}

pub struct Detector {
    session: Mutex<Session>,
    cfg: ModelConfig,
}

impl Detector {
    /// 加载 ONNX 模型。
    ///
    /// 执行提供者由 `cfg.execution_provider` 决定：
    ///   - `"cpu"`  ：仅 CPU。
    ///   - `"dml"`  ：必须 DirectML，注册失败直接报错。
    ///   - `"auto"` ：先试 DirectML（本机 AMD 卡 GPU 加速），失败回退 CPU 并告警。
    ///
    /// 说明：DML EP 要求运行目录存在新版 DirectML.dll（vendor 副本，见 Cargo.toml 注释）。
    /// System32 的 2020 版 DirectML.dll 过旧，会注册失败——此时 auto 模式自动放宽到 CPU。
    pub fn load(cfg: &ModelConfig, onnx_path: &std::path::Path) -> Result<Self> {
        let mode = cfg.execution_provider.trim().to_ascii_lowercase();
        let session = match mode.as_str() {
            "dml" => {
                if !directml_dll_is_fresh() {
                    warn_stale_directml();
                }
                let s = Self::build_session(cfg, onnx_path, true).with_context(|| {
                    format!(
                        "execution_provider = \"dml\"：DirectML 注册失败（确认新版 DirectML.dll \
                         与可执行文件同目录）。模型: {}",
                        onnx_path.display()
                    )
                })?;
                tracing::info!("执行提供者: DirectML (GPU)");
                s
            }
            "cpu" => {
                tracing::info!("执行提供者: CPU（配置指定）");
                Self::build_session(cfg, onnx_path, false)?
            }
            "auto" | "" => {
                // 先自检新版 DirectML.dll：缺失时直接走 CPU，不做"注册成功但
                // 静默退化"的无用尝试（见 warn_stale_directml 的背景说明）。
                if !directml_dll_is_fresh() {
                    warn_stale_directml();
                    tracing::info!("执行提供者: CPU（未启用 DirectML）");
                    Self::build_session(cfg, onnx_path, false)?
                } else {
                    match Self::build_session(cfg, onnx_path, true) {
                        Ok(s) => {
                            tracing::info!("执行提供者: DirectML (GPU)");
                            s
                        }
                        Err(e) => {
                            tracing::warn!("DirectML 不可用（{e:#}），回退 CPU 执行提供者");
                            Self::build_session(cfg, onnx_path, false)?
                        }
                    }
                }
            }
            other => {
                return Err(anyhow!(
                    "未知 execution_provider = \"{other}\"（可选: auto | cpu | dml）"
                ))
            }
        };

        Ok(Self { session: Mutex::new(session), cfg: cfg.clone() })
    }

    /// 构造一个 ORT session；`dml` 为 true 时注册 DirectML 执行提供者。
    /// ORT 会自动追加 CPU EP 作为兜底，因此 DML 未覆盖的算子自动落回 CPU。
    fn build_session(
        cfg: &ModelConfig,
        onnx_path: &std::path::Path,
        dml: bool,
    ) -> Result<Session> {
        let mut builder = Session::builder().context("创建 ORT session builder 失败")?;
        if dml {
            builder = builder
                .with_execution_providers([DirectMLExecutionProvider::default().build()])
                .context("注册 DirectML 执行提供者失败")?;
        }
        if cfg.intra_threads > 0 {
            builder = builder
                .with_intra_threads(cfg.intra_threads)
                .context("设置 ORT 线程数失败")?;
        }
        let session = builder
            .commit_from_file(onnx_path)
            .with_context(|| format!("加载 ONNX 模型失败: {}", onnx_path.display()))?;
        Ok(session)
    }

    pub fn prompt(&self) -> &str {
        &self.cfg.text_prompt
    }

    /// 对一帧预处理结果执行推理，返回过滤 + NMS 后的检测框。
    pub fn detect(&self, prep: &Preprocessed) -> Result<Vec<Detection>> {
        // ort rc.10 的 from_array 需要 owned ndarray（不接受 view），
        // 故这里 clone 一份 NCHW 张量。
        let images = prep.input.clone();

        let (logits, pred_boxes) = {
            let mut session = self.session.lock().map_err(|_| anyhow!("session 锁被污染"))?;
            let images_t = Tensor::from_array(images)
                .context("构造输入张量失败")?
                .into_dyn();
            let outputs = session
                .run(ort::inputs!["images" => images_t])
                .context("ONNX 推理执行失败")?;

            let logits = take_output(&outputs, "pred_logits")?;
            let boxes = take_output(&outputs, "pred_boxes")?;
            (logits, boxes)
        };

        let nq = logits.shape()[1];
        let n_tok = logits.shape()[2];
        let mut dets: Vec<Detection> = Vec::new();

        for q in 0..nq {
            // score = max over token of sigmoid(logit)
            let mut best = f32::NEG_INFINITY;
            for t in 0..n_tok {
                let s = sigmoid(logits[[0, q, t]]);
                if s > best {
                    best = s;
                }
            }
            if best > self.cfg.box_threshold {
                let cxcywh = [
                    pred_boxes[[0, q, 0]],
                    pred_boxes[[0, q, 1]],
                    pred_boxes[[0, q, 2]],
                    pred_boxes[[0, q, 3]],
                ];
                let bbox = cxcywh_to_xyxy_clipped(&cxcywh, prep.orig_width, prep.orig_height);
                dets.push(Detection {
                    norm_cxcywh: cxcywh,
                    bbox_xyxy: bbox,
                    confidence: best,
                    label: "person".to_string(),
                    label_zh: label_zh("person"),
                });
            }
        }

        Ok(greedy_nms(dets, self.cfg.nms_iou_threshold))
    }
}

/// DML 启动自检：确认可执行文件同级存在**新版** DirectML.dll。
///
/// 背景（实测）：Windows 的 DLL 搜索顺序以可执行文件目录优先，其次才是
/// System32。若 exe 同级没有新 DLL，会加载 System32 自带的 2020 旧版
/// （约 1.2 MB）——此时 DML EP 注册仍然“成功”，但内核编译失败并**静默**
/// 逐节点回退 CPU（实测 9.4 s/帧，与纯 CPU 无异）。因此 auto 模式据此
/// 直接选择 CPU，避免"以为在用 GPU"的假象。
fn directml_dll_is_fresh() -> bool {
    /// 新版 DirectML.dll 约 18 MB；System32 旧版约 1.2 MB。用 5 MB 作界限。
    const NEW_DLL_MIN_SIZE: u64 = 5 * 1024 * 1024;
    let Ok(exe) = std::env::current_exe() else { return false };
    let Some(dir) = exe.parent() else { return false };
    std::fs::metadata(dir.join("DirectML.dll"))
        .map(|m| m.len() >= NEW_DLL_MIN_SIZE)
        .unwrap_or(false)
}

fn warn_stale_directml() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dll = exe.parent().unwrap_or(std::path::Path::new(".")).join("DirectML.dll");
    tracing::warn!(
        "未在可执行文件同级目录找到新版 DirectML.dll（期望: {}）。\n\
         System32 的 2020 旧版会导致 DirectML 静默回退 CPU（约 9 s/帧）。\n\
         修复：复制 vendor/ort/onnxruntime/lib/DirectML.dll 到该目录。",
        dll.display()
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 从 ORT 输出中按名称取 f32 三维张量，转成 owned Array3。
///
/// 说明：rc.10 的 `SessionOutputs::get` 只接受名字（`impl AsRef<str>`），
/// **不支持按下标取**，因此这里只用名字；名字缺失时列出实际输出名以便排查。
fn take_output(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Array3<f32>> {
    let value = outputs.get(name).ok_or_else(|| {
        anyhow!(
            "模型输出缺少 `{}`。实际输出: {:?}",
            name,
            outputs.keys().collect::<Vec<_>>()
        )
    })?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("输出 `{}` 提取 f32 张量失败: {e}", name))?;
    let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    if shape.len() != 3 {
        return Err(anyhow!("输出 `{}` 期望 3 维，实际 {:?}", name, shape));
    }
    let arr = Array3::from_shape_vec((shape[0], shape[1], shape[2]), data.to_vec())
        .map_err(|e| anyhow!("输出 `{}` 重排失败: {e}", name))?;
    Ok(arr)
}
