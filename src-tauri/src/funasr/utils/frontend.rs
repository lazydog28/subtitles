use crate::funasr::utils::constant::MEL_BINS;
use crate::funasr::utils::fbank::Frame;
use ndarray::{concatenate, s, Array2, ArrayView1, Axis};

/// Cepstral Mean and Variance Normalization
#[derive(Clone)]
pub struct CMVN {
    pub means: Vec<f32>,
    pub vars: Vec<f32>,
}

pub struct WavFrontend {
    /// 均值方差归一化
    cmvn: CMVN,
    /// LFR(Low Frame Rate)上下文帧数，默认5
    /// 用于降低帧率的上下文窗口大小，将多个连续帧拼接成一个特征向量
    lfr_m: usize,
    /// LFR帧移步长，默认1
    /// LFR处理时的步长，控制降采样的程度
    lfr_n: usize,
}

impl WavFrontend {
    pub fn new(cmvn: CMVN, lfr_m: usize, lfr_n: usize) -> Self {
        let frontend = Self { cmvn, lfr_m, lfr_n };
        frontend
    }
    /// 提取 特征
    /// 参数:
    ///     features: fbank 提取的特征
    /// 返回：
    ///     features：LFR处理+倒谱均值和方差归一化 后的特征向量
    ///     reserve_frames：剩余的帧
    pub fn extract_features(&self, frames: &Vec<Frame>) -> (Array2<f32>, Vec<Frame>) {
        let (features, frames) = self.apply_lfr(frames);
        (self.apply_cmvn(features), frames)
    }

    /// LFR处理
    fn apply_lfr(&self, frames: &Vec<Frame>) -> (Array2<f32>, Vec<Frame>) {
        let frames_count = frames.len();
        // 计算LFR处理后的帧数
        // 公式：(总帧数 - 上下文帧数) / 步长，向下取整
        let t_lfr = ((frames_count as f32 - self.lfr_m as f32) / self.lfr_n as f32) as usize;
        // 计算LFR处理后每帧的长度
        let lfr_len = MEL_BINS * self.lfr_m;
        let mut lfr_input: Array2<f32> = Array2::zeros((t_lfr, lfr_len));
        for t in 0..t_lfr {
            let start = t * self.lfr_n;
            let end = start + self.lfr_m;
            let features_cloned = frames[start..end]
                .iter()
                .map(|frame| frame.feature.view())
                .collect::<Vec<ArrayView1<f32>>>();
            let features = concatenate(Axis(0), &features_cloned).unwrap();
            lfr_input.slice_mut(s![t, ..]).assign(&features);
        }
        // 将剩余特征保存
        let frames = frames[t_lfr * self.lfr_n..].to_vec();
        (lfr_input, frames)
    }

    /// 应用CMVN 倒谱均值和方差归一化
    fn apply_cmvn(&self, features: Array2<f32>) -> Array2<f32> {
        let (frames, feature) = features.dim();
        let mut new_features: Array2<f32> = Array2::zeros((frames, feature));
        for t in 0..frames {
            let mut new_frame = features.slice(s![t, ..]).to_owned();
            for i in 0..feature {
                // 应用CMVN变换：(特征值 + 均值) * 方差倒数
                new_frame[i] = (new_frame[i] + self.cmvn.means[i]) * self.cmvn.vars[i];
            }
            new_features.slice_mut(s![t, ..]).assign(&new_frame);
        }
        new_features
    }
}
