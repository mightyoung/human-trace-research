//! 结果输出：JSONL 落盘 + 可选 webhook 告警。

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::config::Config;
use crate::detection::FrameResult;

pub struct OutputWriter {
    jsonl: BufWriter<File>,
    alert_webhook: String,
    alert_min_count: u32,
}

impl OutputWriter {
    pub fn new(cfg: &Config) -> Result<Self> {
        let path = cfg.jsonl_path();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("创建输出目录失败: {}", parent.display()))?;
            }
        }
        let file = open_append(&path)?;
        Ok(Self {
            jsonl: BufWriter::new(file),
            alert_webhook: cfg.output.alert_webhook.clone(),
            alert_min_count: cfg.output.alert_min_count,
        })
    }

    /// 写入一行 JSON，并在满足条件时触发 webhook。
    pub fn write(&mut self, result: &FrameResult) -> Result<()> {
        let line = serde_json::to_string(result)?;
        self.jsonl.write_all(line.as_bytes())?;
        self.jsonl.write_all(b"\n")?;
        self.jsonl.flush()?;

        if !self.alert_webhook.is_empty() && result.person_count as u32 >= self.alert_min_count {
            self.fire_alert(result);
        }
        Ok(())
    }

    fn fire_alert(&self, result: &FrameResult) {
        let body = serde_json::json!({
            "event": "person_detected",
            "count": result.person_count,
            "frame_index": result.frame_index,
            "timestamp_ms": result.timestamp_ms,
            "image": { "width": result.image_width, "height": result.image_height },
            "detections": result.detections,
        });
        let payload = match serde_json::to_vec(&body) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("webhook 告警序列化失败: {e}");
                return;
            }
        };
        // 告警属外部副作用：走系统 curl，失败仅记录，不影响主流程。
        if let Err(e) = post_json(&self.alert_webhook, &payload) {
            tracing::warn!("webhook 告警发送失败: {e}");
        }
    }
}

/// 通过系统 curl 发送 JSON POST。
///
/// 之所以不引入 HTTP crate：ort-sys 的 download-binaries 构建脚本要求
/// ureq 3 + TLS feature，项目再直接依赖 ureq 会造成 cargo 版本统一冲突。
/// 用 curl 可让依赖图保持干净，运行时二进制也不背 TLS 栈。
fn post_json(url: &str, payload: &[u8]) -> Result<()> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let mut child = Command::new("curl")
        .args([
            "-sS",
            "-X",
            "POST",
            "--max-time",
            "10",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            url,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| "启动 curl 失败（请确认 PATH 中存在 curl）")?;

    child
        .stdin
        .as_mut()
        .context("无法获取 curl stdin")?
        .write_all(payload)
        .context("写入 curl stdin 失败")?;

    let out = child.wait_with_output().context("等待 curl 结束失败")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("curl 退出码 {:?}: {}", out.status.code(), err.trim());
    }
    Ok(())
}

fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("打开输出文件失败: {}", path.display()))
}
