//! 纯逻辑模块的独立测试 crate。
//!
//! 目的：主工程因 ONNX Runtime 预编译库与本机 MSVC STL 版本不匹配（见根 README
//! 第 6 节），链接阶段无法通过，导致 `cargo test` 一并失败。
//! 这里把**不依赖 ort** 的三个模块（预处理、后处理、图片读入）抽出来单独编译，
//! 用于验证其数值/逻辑正确性。
pub mod detection;
pub mod image_input;
pub mod preprocess;
