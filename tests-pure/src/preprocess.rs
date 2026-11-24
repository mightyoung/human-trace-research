//! 图像预处理：等比缩放 (shortest_edge=800, longest_edge<=1333) + 归一化。
//!
//! 与参考实现对齐：
//! - resize: 按比例缩放，使短边 >= 800 且长边 <= 1333（RandomResize([800], max_size=1333)）
//! - 缩放前将像素缩放到 [0,1]（乘 1/255）
//! - normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
//! - 输出 NCHW (f32)，并返回缩放比例用于坐标还原
//!
//! 注意：这里对 resize 的实现是「按比例缩放到目标尺寸，不 padding」，
//! 与 ONNX 导出时使用的 torchvision 等价变换保持一致。

use anyhow::{bail, Result};
use ndarray::Array4;

/// 预处理结果：NCHW 张量 + 原始尺寸与缩放后尺寸。
pub struct Preprocessed {
    pub input: Array4<f32>,
    pub orig_width: u32,
    pub orig_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 复刻 Python 内置 `round()` 的「四舍六入五成双」语义。
/// f64::round() 是四舍五入（0.5 恒向上），会在 .5 处与参考实现差 1。
fn round_half_to_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if (diff - 0.5).abs() < 1e-9 {
        // 恰为 .5：取最近的偶数
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

/// 计算等比缩放后的目标尺寸，严格复刻 GroundingDINO 的
/// `RandomResize([shortest_edge], max_size=longest_edge)`：
///
/// ```text
/// if max_size is not None:
///     if max(w,h) / min(w,h) * size > max_size:
///         size = int(round(max_size * min(w,h) / max(w,h)))
/// if w < h:  ow = size;              oh = int(size * h / w)
/// else:      oh = size;              ow = int(size * w / h)
/// ```
/// 注意第二步是**截断取整**（int()），不是四舍五入。
pub fn target_size(width: u32, height: u32, shortest_edge: u32, longest_edge: u32) -> Result<(u32, u32)> {
    if width == 0 || height == 0 {
        bail!("图像尺寸非法: {}x{}", width, height);
    }
    let w = width as f64;
    let h = height as f64;
    let min_orig = w.min(h);
    let max_orig = w.max(h);

    // 第一步：若按短边放大后长边会超过上限，则先缩小目标 size
    // 注意：Python 的 round() 是「四舍六入五成双」（banker's rounding），
    // 不能用 Rust 的 f64::round()（四舍五入），否则 .5 处会偏差 1。
    let mut size = shortest_edge as f64;
    if max_orig / min_orig * size > longest_edge as f64 {
        size = round_half_to_even(longest_edge as f64 * min_orig / max_orig);
    }

    // 第二步：按目标 size 套到长边，另一维截断取整
    let (nw, nh) = if w <= h && w == size || h <= w && h == size {
        (w, h)
    } else if w < h {
        (size, (size * h / w).floor())
    } else {
        ((size * w / h).floor(), size)
    };

    Ok((nw.max(1.0) as u32, nh.max(1.0) as u32))
}

/// 从 BGR 原始字节（OpenCV 默认通道序）构造模型输入张量。
pub fn preprocess_bgr(
    data: &[u8],
    width: u32,
    height: u32,
    shortest_edge: u32,
    longest_edge: u32,
) -> Result<Preprocessed> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() < expected {
        bail!("BGR 数据长度不足: 期望 {} 实际 {}", expected, data.len());
    }
    let (rw, rh) = target_size(width, height, shortest_edge, longest_edge)?;

    let mut input = Array4::<f32>::zeros((1, 3, rh as usize, rw as usize));
    let x_scale = width as f64 / rw as f64;
    let y_scale = height as f64 / rh as f64;

    // 双线性缩放 + 归一化，直接写入 NCHW
    for ry in 0..rh {
        let sy = ((ry as f64 + 0.5) * y_scale - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(height as usize - 1);
        let wy = (sy - y0 as f64) as f32;

        for rx in 0..rw {
            let sx = ((rx as f64 + 0.5) * x_scale - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(width as usize - 1);
            let wx = (sx - x0 as f64) as f32;

            // 四点采样（BGR）
            let idx = |x: usize, y: usize, c: usize| -> f32 {
                data[(y * width as usize + x) * 3 + c] as f32
            };

            for c in 0..3 {
                let v00 = idx(x0, y0, c);
                let v01 = idx(x1, y0, c);
                let v10 = idx(x0, y1, c);
                let v11 = idx(x1, y1, c);
                let top = v00 * (1.0 - wx) + v01 * wx;
                let bot = v10 * (1.0 - wx) + v11 * wx;
                let val = top * (1.0 - wy) + bot * wy;

                // BGR -> RGB，再归一化
                let rgb_c = 2 - c;
                let norm = (val / 255.0 - MEAN[rgb_c]) / STD[rgb_c];
                input[[0, rgb_c, ry as usize, rx as usize]] = norm;
            }
        }
    }

    Ok(Preprocessed {
        input,
        orig_width: width,
        orig_height: height,
        resized_width: rw,
        resized_height: rh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_scales_short_edge() {
        // 1280x720：短边 720 -> 长边 1280*(800/720)=1422 > 1333
        // => size = round(1333*720/1280) = round(749.8) = 750
        // => w=1333, h=int(750*720/1280)=int(421.8)... 实测参考为 750x1333
        let (w, h) = target_size(1280, 720, 800, 1333).unwrap();
        assert_eq!((w, h), (1333, 750));
    }

    #[test]
    fn target_size_small_image() {
        // 384x288：长边 384*(800/288)=1066 <= 1333，size 保持 800
        // => h=800, w=int(800*384/288)=int(1066.67)=1066（截断）
        let (w, h) = target_size(384, 288, 800, 1333).unwrap();
        assert_eq!((w, h), (1066, 800));
    }

    #[test]
    fn target_size_wide_thin_image() {
        // 64x32：比例 2:1，长边 64*(800/32)=1600 > 1333
        // => size = round(1333*32/64) = 666
        // => oh=666, ow=int(666*64/32)=1332
        let (w, h) = target_size(64, 32, 800, 1333).unwrap();
        assert_eq!((w, h), (1332, 666));
    }

    #[test]
    fn preprocess_outputs_expected_shape() {
        let (w, h) = (64u32, 32u32);
        let data = vec![128u8; (w * h * 3) as usize];
        let p = preprocess_bgr(&data, w, h, 800, 1333).unwrap();
        // 与 target_size 一致：1332x666
        assert_eq!(p.input.shape(), &[1, 3, 666, 1332]);
    }
}
