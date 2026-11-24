//! RTSP 拉流封装（FFmpeg 子进程）。
//!
//! 设计取舍：不链接 FFmpeg 库（`ffmpeg-sys` 在 Windows 上同样要求预装 SDK，
//! 与 OpenCV 一样会把构建搞复杂），而是**调用 `ffmpeg` 可执行文件**，
//! 让其把 RTSP 解码后的原始帧按 rawvideo/BGR24 写到 stdout，Rust 侧按固定
//! 字节数读帧。好处：
//!   - 构建期零原生依赖（不再需要 OpenCV / cmake / pkg-config / libclang）
//!   - FFmpeg 的 RTSP/RTP 实现对海康兼容性最好，且能自然处理重连
//!   - 帧格式（尺寸/像素格式）由我们在命令行里钉死，读取端简单可靠
//!
//! 海康摄像头通常提供两条码流：
//!   主码流 .../Streaming/Channels/101，子码流 .../Streaming/Channels/102
//! 建议先接子码流（分辨率更低，CPU 推理更快）。

use anyhow::{bail, Context, Result};
use std::io::{BufReader, Read};
use std::process::{Child, ChildStdout, Command, Stdio};

/// 输入源类型。决定传给 ffmpeg 的输入相关参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    /// 实时流（RTSP）：需要指定 transport / rw_timeout，并可选 `-re`
    Rtsp,
    /// 本地视频文件：去掉网络相关参数，`-re` 按原生帧率读取
    File,
}

impl SourceKind {
    /// 根据地址前缀推断源类型。
    pub fn infer(input: &str) -> Self {
        let lower = input.trim_start().to_ascii_lowercase();
        if lower.starts_with("rtsp://") || lower.starts_with("rtmp://") || lower.starts_with("http") {
            SourceKind::Rtsp
        } else {
            SourceKind::File
        }
    }
}

pub struct RtspCapture {
    /// ffmpeg 子进程；Drop 时应杀掉，避免僵尸进程堆积
    child: Child,
    stdout: BufReader<ChildStdout>,
    url: String,
    width: u32,
    height: u32,
    /// 单帧字节数 = width * height * 3 (BGR24)
    frame_bytes: usize,
    frame_index: u64,
    /// 复用的读缓冲，避免每帧分配
    buf: Vec<u8>,
}

impl RtspCapture {
    /// 打开 RTSP 流。`transport` 取 "tcp" 或 "udp"。
    ///
    /// `ffmpeg_path` 为 ffmpeg 可执行文件路径（可为 "ffmpeg" 走 PATH）。
    pub fn open(
        url: &str,
        transport: &str,
        read_timeout_secs: u64,
        width: u32,
        height: u32,
        ffmpeg_path: &str,
    ) -> Result<Self> {
        Self::open_with_kind(
            url,
            transport,
            read_timeout_secs,
            width,
            height,
            ffmpeg_path,
            SourceKind::infer(url),
            false,
        )
    }

    /// 打开本地视频文件（同样走 ffmpeg 解码，用于无摄像头时端到端验证）。
    pub fn open_file(
        path: &str,
        width: u32,
        height: u32,
        ffmpeg_path: &str,
        realtime: bool,
    ) -> Result<Self> {
        Self::open_with_kind(path, "tcp", 15, width, height, ffmpeg_path, SourceKind::File, realtime)
    }

    /// 统一的打开实现。
    ///
    /// ffmpeg 命令行要点：
    ///   -rtsp_transport tcp|udp   : 传输协议（仅 RTSP）
    ///   -rw_timeout <us>          : 网络读写超时（仅 RTSP）
    ///   -re                       : 按原生帧率读取（文件模式可选，模拟实时流）
    ///   -i <input>                : 输入
    ///   -f rawvideo -pix_fmt bgr24: 输出原始 BGR24
    ///   -s WxH                    : 钉死输出尺寸（下游按此字节数读帧）
    ///   -an                       : 丢弃音频
    ///   pipe:1                    : 写到 stdout
    #[allow(clippy::too_many_arguments)]
    fn open_with_kind(
        input: &str,
        transport: &str,
        read_timeout_secs: u64,
        width: u32,
        height: u32,
        ffmpeg_path: &str,
        kind: SourceKind,
        realtime: bool,
    ) -> Result<Self> {
        if input.trim().is_empty() {
            bail!("输入地址为空（RTSP 地址或视频文件路径）");
        }
        if width == 0 || height == 0 {
            bail!("采集尺寸非法: {}x{}（需在配置中显式指定）", width, height);
        }

        let size_arg = format!("{}x{}", width, height);
        let mut args: Vec<String> = vec![
            "-hide_banner".into(),
            "-loglevel".into(), "error".into(),
            "-nostdin".into(),
        ];

        if kind == SourceKind::Rtsp {
            let timeout_us = read_timeout_secs.saturating_mul(1_000_000);
            args.extend([
                "-rtsp_transport".into(), transport.to_string(),
                "-rw_timeout".into(), timeout_us.to_string(),
            ]);
        } else if realtime {
            // 文件模式加 -re：按帧率读，模拟实时流的节奏
            args.push("-re".into());
        }

        args.extend(["-i".into(), input.to_string()]);
        args.extend([
            "-an".into(),
            "-f".into(), "rawvideo".into(),
            "-pix_fmt".into(), "bgr24".into(),
            "-s".into(), size_arg,
            "pipe:1".into(),
        ]);

        let mut child = Command::new(ffmpeg_path)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // ffmpeg 的错误直接透传到服务日志
            .spawn()
            .with_context(|| {
                format!(
                    "启动 ffmpeg 失败（路径 `{}`）。请确认已安装 ffmpeg 且在 PATH 中，\
                     或在 config.toml 的 camera.ffmpeg_path 指定完整路径",
                    ffmpeg_path
                )
            })?;

        let stdout = child
            .stdout
            .take()
            .context("无法获取 ffmpeg stdout（内部错误）")?;

        let frame_bytes = (width as usize) * (height as usize) * 3;

        Ok(Self {
            child,
            stdout: BufReader::with_capacity(frame_bytes, stdout),
            url: input.to_string(),
            width,
            height,
            frame_bytes,
            frame_index: 0,
            buf: vec![0u8; frame_bytes],
        })
    }

    /// 读取一帧并返回 BGR 字节缓冲。返回 (data, width, height)。
    ///
    /// 语义与之前 OpenCV 版本保持一致：读失败返回 Err 以触发上层重连。
    /// 用 `read_exact` 语义：只有凑满一整帧才返回，避免把半帧当帧用。
    pub fn read_frame(&mut self) -> Result<(Vec<u8>, u32, u32)> {
        let mut filled = 0usize;
        while filled < self.frame_bytes {
            match self.stdout.read(&mut self.buf[filled..]) {
                Ok(0) => {
                    // EOF：ffmpeg 进程结束（流断开或解码失败）
                    bail!("ffmpeg 输出结束（流可能已断开）");
                }
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e).context("读取 ffmpeg 输出失败"),
            }
        }

        self.frame_index += 1;
        Ok((self.buf.clone(), self.width, self.height))
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for RtspCapture {
    fn drop(&mut self) {
        // 主动杀掉 ffmpeg，避免重连时残留子进程
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
