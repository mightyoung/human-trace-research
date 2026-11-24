//! 配置加载：从 TOML 读取服务配置，所有字段带默认值。

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub camera: CameraConfig,
    pub input: InputConfig,
    pub model: ModelConfig,
    pub output: OutputConfig,
    pub logging: LoggingConfig,
    /// 配置文件所在目录，用于解析相对路径（不参与反序列化）
    #[serde(skip)]
    pub base_dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub url: String,
    /// "tcp" | "udp"
    pub transport: String,
    pub sample_interval_secs: u64,
    pub read_timeout_secs: u64,
    /// ffmpeg 可执行文件路径；"ffmpeg" 表示走 PATH
    pub ffmpeg_path: String,
    /// 解码输出宽高（钉死给 ffmpeg 的 -s）
    pub width: u32,
    pub height: u32,
    pub reconnect: ReconnectConfig,
}

/// 图片输入模式配置（仅用于离线验证）。
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InputConfig {
    pub image_path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ReconnectConfig {
    pub initial_secs: u64,
    pub max_secs: u64,
    pub factor: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub onnx_path: String,
    pub text_prompt: String,
    pub box_threshold: f32,
    pub text_threshold: f32,
    pub nms_iou_threshold: f32,
    pub shortest_edge: u32,
    pub longest_edge: u32,
    pub intra_threads: usize,
    /// 执行提供者："auto"（默认，先试 DirectML 再回退 CPU）| "cpu" | "dml"
    pub execution_provider: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub jsonl_path: String,
    pub alert_webhook: String,
    pub alert_min_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            transport: "tcp".into(),
            sample_interval_secs: 10,
            read_timeout_secs: 15,
            ffmpeg_path: "ffmpeg".into(),
            width: 704,
            height: 576,
            reconnect: ReconnectConfig::default(),
        }
    }
}

impl Default for InputConfig {
    fn default() -> Self {
        Self { image_path: String::new() }
    }
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self { initial_secs: 1, max_secs: 30, factor: 2.0 }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            onnx_path: "models/onnx/groundingdino.onnx".into(),
            text_prompt: "person".into(),
            box_threshold: 0.25,
            text_threshold: 0.25,
            nms_iou_threshold: 0.7,
            shortest_edge: 800,
            longest_edge: 1333,
            intra_threads: 4,
            execution_provider: "auto".into(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self { jsonl_path: "output/detections.jsonl".into(), alert_webhook: String::new(), alert_min_count: 1 }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self { level: "info".into() }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            input: InputConfig::default(),
            model: ModelConfig::default(),
            output: OutputConfig::default(),
            logging: LoggingConfig::default(),
            base_dir: PathBuf::from("."),
        }
    }
}

impl Config {
    /// 从指定 TOML 文件加载配置。
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("读取配置文件失败: {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text)
            .with_context(|| format!("解析配置文件失败: {}", path.display()))?;
        cfg.base_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Ok(cfg)
    }

    /// 将配置中的相对路径解析为绝对路径。
    pub fn resolve(&self, rel: &str) -> PathBuf {
        let p = Path::new(rel);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.base_dir.join(p)
        }
    }

    pub fn model_path(&self) -> PathBuf {
        self.resolve(&self.model.onnx_path)
    }

    pub fn jsonl_path(&self) -> PathBuf {
        self.resolve(&self.output.jsonl_path)
    }

    pub fn image_path(&self) -> Option<PathBuf> {
        if self.input.image_path.trim().is_empty() {
            None
        } else {
            Some(self.resolve(&self.input.image_path))
        }
    }
}
