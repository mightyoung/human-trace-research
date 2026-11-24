//! human-count：RTSP 摄像头人体检测服务
//!
//! 架构：
//! 1. `capture`     —— 调用 ffmpeg 子进程拉取 RTSP / 读本地视频，解码为 BGR24
//! 2. `image_input` —— 静态图片输入（仅离线验证用）
//! 3. `preprocess`  —— 等比缩放 + 归一化，产出模型输入张量
//! 4. `detector`    —— ONNX Runtime 执行 GroundingDINO，输出 person 框
//! 5. `detection`   —— 坐标映射 + greedy NMS
//! 6. `output`      —— JSONL 落盘 + 可选 webhook 告警
//!
//! 三种运行模式：
//!   - 默认       : 连 RTSP，按 sample_interval_secs 抽帧，断流退避重连
//!   - `--video`  : 读本地视频文件，同样走 ffmpeg 解码（无摄像头时验证全链路）
//!   - `--image`  : 读单张图片跑一遍后退出（最快的链路验证）

mod capture;
mod config;
mod detection;
mod detector;
mod image_input;
mod output;
mod preprocess;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

use capture::RtspCapture;
use config::Config;
use detector::Detector;

#[derive(Parser, Debug)]
#[command(name = "human-count", about = "RTSP 摄像头人体检测服务（GroundingDINO + ONNX Runtime）")]
struct Cli {
    /// 配置文件路径
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// 仅运行一次（不循环），用于链路验证
    #[arg(long, default_value_t = false)]
    once: bool,

    /// 覆盖 config.toml 中的 RTSP 地址
    #[arg(long)]
    rtsp: Option<String>,

    /// 静态图片输入模式：读单张图片跑完整链路后退出（离线验证用）
    #[arg(long)]
    image: Option<PathBuf>,

    /// 本地视频文件输入模式：走 ffmpeg 解码并按间隔抽帧（无摄像头时验证全链路）
    #[arg(long)]
    video: Option<PathBuf>,

    /// 视频/流模式下最多处理多少帧后退出（0 表示不限）
    #[arg(long, default_value_t = 0)]
    max_frames: u64,
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut cfg = Config::load(&cli.config)
        .with_context(|| format!("加载配置失败: {}", cli.config.display()))?;
    if let Some(url) = cli.rtsp.clone() {
        cfg.camera.url = url;
    }
    if let Some(img) = cli.image.clone() {
        cfg.input.image_path = img.to_string_lossy().into_owned();
    }

    // 日志初始化
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| cfg.logging.level.clone());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    info!("human-count 启动");

    // ---- 校验并加载模型 ----
    let model_path = cfg.model_path();
    if !model_path.exists() {
        error!("未找到 ONNX 模型: {}", model_path.display());
        info!("请先运行: python tools/export_onnx.py --model-dir models/AI-ModelScope--GroundingDINO/snapshots/master --out-dir models/onnx");
        anyhow::bail!("ONNX 模型缺失，服务无法启动");
    }

    let detector = Detector::load(&cfg.model, &model_path).context("加载检测模型失败")?;
    info!("模型加载完成，已烘焙 prompt = \"{}\"", detector.prompt());

    // ---- 输出通道 ----
    let mut writer = output::OutputWriter::new(&cfg)?;

    // ---- 分支：图片 / 视频文件 / RTSP ----
    if let Some(path) = cfg.image_path() {
        return run_image_mode(&cfg, &detector, &mut writer, &path);
    }

    if let Some(video) = cli.video.clone() {
        let input = video.to_string_lossy().into_owned();
        let max_frames = if cli.max_frames == 0 { u64::MAX } else { cli.max_frames };
        info!("视频文件模式: {}（最多 {} 帧）", input, if max_frames == u64::MAX { "不限".into() } else { max_frames.to_string() });
        let cap = RtspCapture::open_file(
            &input,
            cfg.camera.width,
            cfg.camera.height,
            &cfg.camera.ffmpeg_path,
            true, // 按原生帧率读，模拟实时节奏
        )?;
        return pump_frames(&cfg, &detector, &mut writer, cap, max_frames);
    }

    run_rtsp_mode(&cli, &cfg, &detector, &mut writer)
}

/// 单张图片：读图 → 预处理 → 推理 → 写结果，跑完即退出。
fn run_image_mode(
    cfg: &Config,
    detector: &Detector,
    writer: &mut output::OutputWriter,
    path: &std::path::Path,
) -> Result<()> {
    info!("图片模式: {}", path.display());

    let (data, w, h) = image_input::load_bgr(path)?;
    info!("已读取图片 {}x{}", w, h);

    let prep = preprocess::preprocess_bgr(
        &data,
        w,
        h,
        cfg.model.shortest_edge,
        cfg.model.longest_edge,
    )
    .context("预处理失败")?;
    info!(
        "预处理后尺寸 {}x{}（{:.3} MB f32）",
        prep.resized_width,
        prep.resized_height,
        (prep.input.len() as f64 * 4.0) / 1024.0 / 1024.0
    );

    let t = Instant::now();
    let dets = detector.detect(&prep).context("推理失败")?;
    let infer_ms = t.elapsed().as_millis();

    let result = detection::FrameResult {
        frame_index: 1,
        timestamp_ms: now_ms(),
        image_width: w,
        image_height: h,
        person_count: dets.len(),
        detections: dets,
    };

    log_frame(&result, infer_ms);
    writer.write(&result)?;
    info!("结果已写入 {}", cfg.jsonl_path().display());
    Ok(())
}

/// 统一的「抽帧 → 推理 → 输出」循环。
///
/// `max_frames` 达到即返回；`read_frame` 失败（EOF/断流）也返回 Ok，
/// 由调用方决定是重连还是结束。
fn pump_frames(
    cfg: &Config,
    detector: &Detector,
    writer: &mut output::OutputWriter,
    mut cap: RtspCapture,
    max_frames: u64,
) -> Result<()> {
    let interval = Duration::from_secs(cfg.camera.sample_interval_secs);
    let mut processed: u64 = 0;

    loop {
        if processed >= max_frames {
            info!("已达到帧数上限 {}，退出", max_frames);
            return Ok(());
        }

        let tick = Instant::now();
        let (data, w, h) = match cap.read_frame() {
            Ok(v) => v,
            Err(e) => {
                warn!("读取帧失败: {e:#}");
                return Ok(());
            }
        };

        let prep = match preprocess::preprocess_bgr(
            &data,
            w,
            h,
            cfg.model.shortest_edge,
            cfg.model.longest_edge,
        ) {
            Ok(p) => p,
            Err(e) => {
                warn!("预处理失败: {e:#}");
                continue;
            }
        };

        let infer_start = Instant::now();
        let dets = match detector.detect(&prep) {
            Ok(d) => d,
            Err(e) => {
                warn!("推理失败: {e:#}");
                continue;
            }
        };
        let infer_ms = infer_start.elapsed().as_millis();
        processed += 1;

        let result = detection::FrameResult {
            frame_index: cap.frame_index(),
            timestamp_ms: now_ms(),
            image_width: w,
            image_height: h,
            person_count: dets.len(),
            detections: dets,
        };

        log_frame(&result, infer_ms);
        if let Err(e) = writer.write(&result) {
            warn!("写入结果失败: {e:#}");
        }

        // 节流到采样间隔
        let elapsed = tick.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
}

fn log_frame(result: &detection::FrameResult, infer_ms: u128) {
    info!(
        "帧 #{} | 人数={} | 推理 {}ms | {}x{}",
        result.frame_index, result.person_count, infer_ms, result.image_width, result.image_height
    );
    for (i, d) in result.detections.iter().enumerate() {
        info!(
            "  #{} conf={:.4} bbox=[{:.1}, {:.1}, {:.1}, {:.1}] {}",
            i, d.confidence, d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3], d.label_zh
        );
    }
}

/// RTSP 循环：连接 → 抽帧 → 推理 → 输出；断流则退避重连。
fn run_rtsp_mode(
    cli: &Cli,
    cfg: &Config,
    detector: &Detector,
    writer: &mut output::OutputWriter,
) -> Result<()> {
    info!("RTSP: {}", cfg.camera.url);
    info!(
        "抽帧间隔: {}s，采集尺寸: {}x{}，模型: {}",
        cfg.camera.sample_interval_secs,
        cfg.camera.width,
        cfg.camera.height,
        cfg.model_path().display()
    );

    let max_frames = if cli.once { 1 } else { u64::MAX };
    let mut backoff = cfg.camera.reconnect.initial_secs as f64;

    loop {
        match RtspCapture::open(
            &cfg.camera.url,
            &cfg.camera.transport,
            cfg.camera.read_timeout_secs,
            cfg.camera.width,
            cfg.camera.height,
            &cfg.camera.ffmpeg_path,
        ) {
            Ok(cap) => {
                backoff = cfg.camera.reconnect.initial_secs as f64;
                info!("已启动 ffmpeg 拉流: {}", cfg.camera.url);
                // 处理到断流或达到帧数上限
                pump_frames(cfg, detector, writer, cap, max_frames)?;
                if cli.once {
                    info!("--once 模式，退出");
                    return Ok(());
                }
            }
            Err(e) => {
                warn!("启动拉流失败: {e:#}；{}s 后重试", backoff as u64);
                std::thread::sleep(Duration::from_secs_f64(backoff));
                backoff =
                    (backoff * cfg.camera.reconnect.factor).min(cfg.camera.reconnect.max_secs as f64);
                continue;
            }
        }

        warn!("与 RTSP 断开，{}s 后重连", backoff as u64);
        std::thread::sleep(Duration::from_secs_f64(backoff));
        backoff = (backoff * cfg.camera.reconnect.factor).min(cfg.camera.reconnect.max_secs as f64);
    }
}
