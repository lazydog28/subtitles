use crate::funasr::utils::{fbank, Frame};
use ndarray::{concatenate, Array1, Axis};

/// 预处理音频数据
/// 将音频数据转换为 [-32768,32768] 范围，并与剩余音频数据拼接
/// 返回特征和剩余音频数据
/// # 参数
/// - `waveform`: 当前音频数据
/// - `reserve_waveforms`: 上一次处理剩余的音频数据
/// - `noise`: 是否添加噪声
/// # 返回值
/// - `features`: 提取的特征
/// - `waveform`: 特征对应的音频数据
/// - `remaining_waveform`: 剩余的音频数据
pub fn pretreatment(
    waveform: Vec<f32>,
    reserve_waveforms: Array1<f32>,
) -> (Vec<Frame>, Array1<f32>) {
    let mut waveform = Array1::from_vec(waveform);
    // 将音频数据转换为 [-32768,32768]
    waveform.mapv_inplace(|x| x * 32768.0f32);
    // 将上次剩余的音频数据与当前音频数据拼接
    waveform = concatenate![Axis(0), reserve_waveforms, waveform];
    let (frames, remaining_waveform) = fbank(waveform);
    (frames, remaining_waveform)
}
// /// 预处理音频数据 在音频数据尾端填充空白 以保证所有数据都被处理
// ///
// /// 这里的处理非常粗暴，理论上应该缺多少补多少，但是我懒
// pub fn pretreatment_with_pad(mut waveform: Vec<f32>) -> (Vec<Frame>) {
//     // 尾部添加 FRAME_LENGTH 长度的空白
//     waveform.extend(vec![0.0; FRAME_LENGTH]);
//     // 尾部添加 SPEECH_RECOGNITION_LFR_M * FRAME_SHIFT 个空白
//     waveform.extend(vec![0.0; SPEECH_RECOGNITION_LFR_M * FRAME_SHIFT]);
//     let mut waveform = Array1::from_vec(waveform);
//     // 将音频数据转换为 [-32768,32768]
//     waveform.mapv_inplace(|x| x * 32768.0f32);
//     let (frames, _) = fbank(waveform);
//     frames
// }
