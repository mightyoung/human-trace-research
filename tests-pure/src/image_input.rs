//! 静态图片输入（仅用于离线端到端验证）。
//!
//! 目的：在没有摄像头的环境下，验证
//! `预处理 → ONNX 推理 → 阈值 → NMS → 输出` 整条链路与 Python 参考实现一致。
//!
//! 用 `image` crate 纯 Rust 解码，产出与 `capture` 相同约定的
//! **BGR24 连续缓冲**，因此下游 `preprocess::preprocess_bgr` 可复用，
//! 保证验证路径与运行路径的预处理逻辑完全相同。

use anyhow::{bail, Context, Result};
use std::path::Path;

/// 读取图片并转换为 (BGR24 数据, width, height)。
pub fn load_bgr(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path)
        .with_context(|| format!("读取图片失败: {}", path.display()))?;

    // 统一转 8bit RGB
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w == 0 || h == 0 {
        bail!("图片尺寸非法: {}x{}", w, h);
    }

    // RGB -> BGR，与 OpenCV/ffmpeg 的通道序对齐
    let src = rgb.as_raw();
    let mut out = vec![0u8; (w as usize) * (h as usize) * 3];
    for (i, px) in src.chunks_exact(3).enumerate() {
        out[i * 3] = px[2]; // B
        out[i * 3 + 1] = px[1]; // G
        out[i * 3 + 2] = px[0]; // R
    }

    Ok((out, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgr_channel_order_is_swapped() {
        // 构造一张 1x1 的纯红 PNG，验证 RGB(255,0,0) -> BGR(0,0,255)
        let dir = std::env::temp_dir();
        let p = dir.join("hc_test_red.png");
        let mut im = image::RgbImage::new(1, 1);
        im.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        im.save(&p).unwrap();

        let (data, w, h) = load_bgr(&p).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(&data[..3], &[0u8, 0, 255], "红应转成 BGR=(0,0,255)");
        let _ = std::fs::remove_file(&p);
    }
}
