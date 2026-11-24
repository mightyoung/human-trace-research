//! 检测结果的数据结构与后处理：坐标映射、IoU、greedy NMS。
//!
//! 语义与参考实现 `DINO-code/.../minimal_frozen_detector_call.py` 对齐：
//! - 阈值判定为「严格大于」(score > threshold)
//! - NMS 为按 confidence 降序的 greedy 抑制，IoU 严格大于阈值才抑制
//! - 归一化 cxcywh -> 原图像素 xyxy，并逐坐标 clip 到 [0, w]/[0, h]

use serde::Serialize;

/// 单个检测框（像素坐标，原图尺度）。
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    /// 归一化中心点与宽高（模型原始输出，cxcywh，取值 0~1）
    pub norm_cxcywh: [f32; 4],
    /// 原图像素坐标 xyxy（已 clip）
    pub bbox_xyxy: [f32; 4],
    /// 置信度（query-token logits 的 sigmoid 最大值）
    pub confidence: f32,
    /// 类别名（英文，来自模型标签）
    pub label: String,
    /// 类别中文名（映射后的展示名）
    pub label_zh: String,
}

/// 一帧的检测结果。
#[derive(Debug, Clone, Serialize)]
pub struct FrameResult {
    pub frame_index: u64,
    /// 该帧的采集时间（毫秒时间戳，UTC）
    pub timestamp_ms: u128,
    pub image_width: u32,
    pub image_height: u32,
    pub person_count: usize,
    pub detections: Vec<Detection>,
}

/// 计算两个 xyxy 框的 IoU。空框（面积为 0）返回 0。
pub fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let inter_h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// 归一化 cxcywh 转原图像素 xyxy，并 clip 到图像边界。
///
/// 与参考实现（`post_process_grounded_object_detection` 传 target_sizes）一致：
/// 先把归一化 cxcywh 转成 xyxy，再**乘以图像宽高**还原到像素尺度，
/// 最后逐坐标 clip 到 [0, width] / [0, height]。
pub fn cxcywh_to_xyxy_clipped(cxcywh: &[f32; 4], width: u32, height: u32) -> [f32; 4] {
    let (cx, cy, w, h) = (cxcywh[0], cxcywh[1], cxcywh[2], cxcywh[3]);
    let (fw, fh) = (width as f32, height as f32);

    // 归一化 xyxy
    let nx1 = cx - w / 2.0;
    let ny1 = cy - h / 2.0;
    let nx2 = cx + w / 2.0;
    let ny2 = cy + h / 2.0;

    // 还原到像素尺度并 clip
    [
        (nx1 * fw).clamp(0.0, fw),
        (ny1 * fh).clamp(0.0, fh),
        (nx2 * fw).clamp(0.0, fw),
        (ny2 * fh).clamp(0.0, fh),
    ]
}

/// Greedy NMS：按 confidence 降序，与任一已保留框 IoU 严格大于阈值则抑制。
/// 输入需已按类别过滤（本服务仅 person，单类）。
pub fn greedy_nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // 降序；等分时按索引稳定排序，保持与参考实现的确定性
    dets.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::new();
    for cand in dets.into_iter() {
        let suppressed = kept
            .iter()
            .any(|k| box_iou(&cand.bbox_xyxy, &k.bbox_xyxy) > iou_threshold);
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(conf: f32, bbox: [f32; 4]) -> Detection {
        Detection {
            norm_cxcywh: [0.0; 4],
            bbox_xyxy: bbox,
            confidence: conf,
            label: "person".into(),
            label_zh: "人".into(),
        }
    }

    #[test]
    fn iou_disjoint_is_zero() {
        assert_eq!(box_iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]), 0.0);
    }

    #[test]
    fn iou_identical_is_one() {
        assert!((box_iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_overlap_above_threshold() {
        // 两个高度重叠的框，保留高分
        let dets = vec![
            det(0.9, [0.0, 0.0, 10.0, 10.0]),
            det(0.8, [1.0, 1.0, 11.0, 11.0]),
        ];
        let kept = greedy_nms(dets, 0.5);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_overlap_below_threshold() {
        // 两个框 IoU≈1/3；阈值 0.5 时不抑制，应全部保留
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 0.0, 15.0, 10.0]; // 交集 50，并集 150 -> IoU=1/3≈0.333
        let kept = greedy_nms(vec![det(0.9, a), det(0.8, b)], 0.5);
        assert_eq!(kept.len(), 2, "IoU(0.333) 不大于 0.5 时应全部保留");
    }

    #[test]
    fn nms_suppresses_when_above_threshold() {
        // 同上两个框，阈值降到 0.25 时 IoU(0.333)>0.25，应抑制低分框
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 0.0, 15.0, 10.0];
        let kept = greedy_nms(vec![det(0.9, a), det(0.8, b)], 0.25);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn cxcywh_maps_and_clips() {
        // 中心 (0.5,0.5) 宽高 (1.0,1.0) -> 整图
        let b = cxcywh_to_xyxy_clipped(&[0.5, 0.5, 1.0, 1.0], 100, 50);
        assert_eq!(b, [0.0, 0.0, 100.0, 50.0]);

        // 中心 (0.05,0.5)、宽 0.2 -> x1=0.05-0.1=-0.05 应 clip 到 0
        let b2 = cxcywh_to_xyxy_clipped(&[0.05, 0.5, 0.2, 0.2], 100, 50);
        assert_eq!(b2[0], 0.0, "左边界应被 clip 到 0");
        assert!(b2[2] > 0.0 && b2[2] <= 100.0);

        // 越界右下角应 clip 到边界
        let b3 = cxcywh_to_xyxy_clipped(&[0.98, 0.98, 0.2, 0.2], 100, 50);
        assert!((b3[2] - 100.0).abs() < 1e-6);
        assert!((b3[3] - 50.0).abs() < 1e-6);
    }
}
