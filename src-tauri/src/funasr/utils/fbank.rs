use crate::funasr::utils::constant::{
    FRAME_LENGTH, FRAME_SHIFT, HAMMING_WINDOW, MEL_BANKS, PADDED_SIZE, PREEMPH_COEFF,
};
use ndarray;
use ndarray::{s, Array1, Axis, Zip};

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::{Arc, LazyLock};

static FFT: LazyLock<Arc<dyn Fft<f32>>> = LazyLock::new(|| {
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(PADDED_SIZE)
});

#[derive(Clone)]
pub struct Frame {
    pub feature: Array1<f32>, // 梅尔频谱
    pub decibel: f32,         // 帧的分贝值
}

impl Frame {
    pub fn new(mut waveform: Array1<f32>) -> Self {
        let decibel = compute_decibel(&waveform);
        // 移除直流分量，消除信号中的直流偏移
        let mean = waveform.mean().unwrap();
        waveform.mapv_inplace(|x| x - mean);

        // 预加重 preemph_coeff
        for j in (1..waveform.len()).rev() {
            waveform[j] -= PREEMPH_COEFF * waveform[j - 1];
        }
        let hamming_window = HAMMING_WINDOW.to_owned();
        // 应用汉明窗
        Zip::from(&mut waveform)
            .and(&hamming_window)
            .for_each(|x, &h| {
                *x *= h;
            });
        // 计算特征
        let feature = compute_features(waveform);
        Self { feature, decibel }
    }
}

/// 计算 Filter Bank 特征
pub fn fbank(waveform: Array1<f32>) -> (Vec<Frame>, Array1<f32>) {
    let waveform_length = waveform.len();
    // 如果音频数据的长度小于窗口长度，则返回空
    if waveform_length < FRAME_LENGTH {
        return (Vec::new(), waveform);
    }
    let frame_count = 1 + (waveform_length - FRAME_LENGTH) / FRAME_SHIFT;
    // 获取剩余的数据
    let remaining_waveform: Array1<f32> =
        waveform.slice(s![frame_count * FRAME_SHIFT..]).into_owned();
    let mut frames: Vec<Frame> = Vec::new();
    for i in 0..frame_count {
        let frame = Frame::new(
            waveform
                .slice(s![i * FRAME_SHIFT..i * FRAME_SHIFT + FRAME_LENGTH])
                .into_owned(),
        );
        frames.push(frame)
    }
    (frames, remaining_waveform)
}

fn compute_features(window: Array1<f32>) -> Array1<f32> {
    // 将输入调整到2的幂
    let mut fft_input = Array1::zeros(PADDED_SIZE);
    fft_input.slice_mut(s![..FRAME_LENGTH]).assign(&window);
    let mut buffer: Vec<Complex<f32>> = fft_input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    FFT.process(&mut buffer);
    let power_spectrum = compute_power_spectrum(&buffer);

    let mut mel_energies = compute_mel(&power_spectrum);

    // 应用对数变换
    mel_energies.mapv_inplace(|x| x.max(1e-10).ln());
    mel_energies
}

/// 计算梅尔频谱
fn compute_mel(power_spectrum: &Array1<f32>) -> Array1<f32> {
    let mut mel_energies = Array1::zeros(MEL_BANKS.shape()[0]);
    // 对每个梅尔滤波器进行计算
    for (i, mel_bank) in MEL_BANKS.axis_iter(Axis(0)).enumerate() {
        let mut sum = 0.0;
        for (j, &mel_filter) in mel_bank.iter().enumerate() {
            sum += power_spectrum[j] * mel_filter;
        }
        mel_energies[i] = sum;
    }
    mel_energies
}

/// 计算功率谱
fn compute_power_spectrum(complex_fft: &Vec<Complex<f32>>) -> Array1<f32> {
    let mut power_spectrum = Array1::zeros(complex_fft.len());
    for (i, &c) in complex_fft.iter().enumerate() {
        power_spectrum[i] = c.norm_sqr();
    }
    power_spectrum
}

/// 计算分贝
fn compute_decibel(waveform: &Array1<f32>) -> f32 {
    // 计算能量
    let energy = waveform.map(|x| x.powi(2)).sum();
    // 转换为分贝值，添加小常数避免log(0)
    10.0 * (energy + 1e-10).log10()
}
