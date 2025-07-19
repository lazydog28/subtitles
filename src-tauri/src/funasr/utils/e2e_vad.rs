use crate::funasr::utils::constant::{
    SIL_TO_SPEECH_FRMCNT_THRES, SPEECH_TO_SIL_FRMCNT_THRES, WIN_SIZE_FRAME,
};
use crate::funasr::utils::fbank::Frame;
use ndarray::{Array2, ArrayView1};

const NOISE_FRAME_NUM_USED_FOR_SNR: f32 = 100.0;
const SNR_THRES: f32 = -100.0; // 信噪比阈值(dB) - 低于此值的帧被认为是噪声
const DECIBEL_THRES: f32 = -100.0; // 分贝阈值(dB) - 音频能量阈值，低于此值的帧被认为是静音
const SPEECH_NOISE_THRES: f32 = 0.6; // 语音/噪声概率阈值 - 神经网络输出的概率判别阈值
const SPEECH_2_NOISE_RATIO: f32 = 1.0; // 语音/噪声比 - 用于调整语音和噪声的判别阈值

/// 帧状态枚举
/// 定义每一帧音频的分类结果
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameState {
    /// 语音帧：当前帧包含语音信号
    KFrameStateSpeech = 1,

    /// 静音帧：当前帧为静音或背景噪声
    KFrameStateSil = 0,
}

// 每帧的最终语音/非语音状态
/// 音频状态变化枚举
/// 描述连续帧之间的语音状态转换
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioChangeState {
    /// 语音到语音：连续的语音帧
    KChangeStateSpeech2Speech = 0,

    /// 语音到静音：从语音转为静音
    KChangeStateSpeech2Sil = 1,

    /// 静音到静音：连续的静音帧
    KChangeStateSil2Sil = 2,

    /// 静音到语音：从静音转为语音（语音起点）
    KChangeStateSil2Speech = 3,

    /// 无效状态：异常情况
    KChangeStateInvalid = 5,
}

/// 滑动窗口检测器
/// 使用滑动窗口技术平滑语音检测结果，减少误检
#[derive(Debug, Clone)]
pub struct WindowDetector {
    /// 窗口内语音帧总数
    win_sum: usize,

    /// 窗口状态数组，存储每帧的状态(0-静音, 1-语音)
    win_state: Vec<usize>,

    /// 当前窗口位置指针，实现循环缓冲区
    cur_win_pos: usize,

    /// 前一帧状态
    pre_frame_state: FrameState,
}

impl Default for WindowDetector {
    fn default() -> Self {
        Self {
            win_sum: 20,
            win_state: vec![0; 20], // 初始化窗口状态为20个静音帧
            cur_win_pos: 0,
            pre_frame_state: FrameState::KFrameStateSil,
        }
    }
}

impl WindowDetector {
    /// 重置检测器状态
    /// 清空所有计数器和状态信息
    pub fn reset(&mut self) {
        self.cur_win_pos = 0;
        self.win_sum = 0;
        self.win_state.fill(0);
        self.pre_frame_state = FrameState::KFrameStateSil;
    }

    /// 检测单帧音频的状态变化
    ///
    /// # 参数
    /// * `frame_state` - 当前帧的状态(语音/静音)
    /// * `_frame_count` - 帧计数器(当前未使用)
    ///
    /// # 返回值
    /// 返回音频状态变化类型
    pub fn detect_one_frame(&mut self, frame_state: FrameState) -> AudioChangeState {
        // 将帧状态转换为数值(语音=1, 静音=0)
        let cur_frame_state = match frame_state {
            FrameState::KFrameStateSpeech => 1,
            FrameState::KFrameStateSil => 0,
        };

        // 更新滑动窗口：移除旧帧，添加新帧
        self.win_sum -= self.win_state[self.cur_win_pos]; // 如果上一帧是语音，则减去1，否则减去0
        self.win_sum += cur_frame_state; // 如果当前帧是语音，则加上1，否则加上0
        self.win_state[self.cur_win_pos] = cur_frame_state; // 更新当前帧的状态
        self.cur_win_pos = (self.cur_win_pos + 1) % WIN_SIZE_FRAME; // 更新当前窗口位置 +1

        // 检测静音到语音的转换
        if self.pre_frame_state == FrameState::KFrameStateSil
            && self.win_sum >= SIL_TO_SPEECH_FRMCNT_THRES
        {
            self.pre_frame_state = FrameState::KFrameStateSpeech;
            return AudioChangeState::KChangeStateSil2Speech;
        }

        // 检测语音到静音的转换
        if self.pre_frame_state == FrameState::KFrameStateSpeech
            && self.win_sum <= SPEECH_TO_SIL_FRMCNT_THRES
        {
            self.pre_frame_state = FrameState::KFrameStateSil;
            return AudioChangeState::KChangeStateSpeech2Sil;
        }

        // 返回当前持续状态
        if self.pre_frame_state == FrameState::KFrameStateSil {
            return AudioChangeState::KChangeStateSil2Sil;
        }
        if self.pre_frame_state == FrameState::KFrameStateSpeech {
            return AudioChangeState::KChangeStateSpeech2Speech;
        }
        AudioChangeState::KChangeStateInvalid
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointType {
    /// 语音段起点
    Start,
    /// 语音段终点
    End,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub segment_type: PointType,
}
pub struct E2EVadModel {
    /// 滑动窗口检测器，用于平滑检测结果
    windows_detector: WindowDetector,
    /// 噪声平均分贝值，用于动态调整阈值
    noise_average_decibel: f32,
}

impl Default for E2EVadModel {
    fn default() -> Self {
        Self {
            windows_detector: WindowDetector {
                win_sum: 0,
                win_state: vec![0; 20], // 初始化窗口状态为20个静音帧
                cur_win_pos: 0,
                pre_frame_state: FrameState::KFrameStateSil,
            },
            noise_average_decibel: -100.0, // 初始噪声分贝值设为-100dB
        }
    }
}

impl E2EVadModel {
    pub fn reset(&mut self) {
        self.windows_detector.reset();
    }
    /// E2E VAD主调用接口
    /// 处理输入的音频数据和神经网络得分，返回检测到的语音段
    /// ## 参数
    ///     * `scores`: 神经网络输出的得分矩阵 [time_steps, num_classes] num_classes为 248
    ///     * `waveform`: 特征的原始音频信号，该数据从`WavFrontend`中调用函数返回
    pub fn call(&mut self, scores: Array2<f32>, frames: &Vec<Frame>) -> Vec<Segment> {
        let (frame_count, _) = scores.dim();
        // 计算音频分贝值序列
        let mut segments: Vec<Segment> = Vec::new();
        // 进行帧检测
        for i in 0..frame_count {
            let score = scores.row(i);
            let decibel = frames[i].decibel;
            // 获取当前帧的语音状态
            let frame_state = self.get_frame_state(score, decibel);
            // 检测单帧的状态变化并更新VAD状态机
            if let Some(segment) = self.detect_one_frame(frame_state) {
                segments.push(segment);
            }
        }
        segments
    }

    /// 判断当前帧的语音状态
    fn get_frame_state(&mut self, score: ArrayView1<f32>, decibel: f32) -> FrameState {
        // 初始化帧状态
        // 计算当前帧的信噪比（SNR）= 当前音量 - 噪声平均音量
        let snr = decibel - self.noise_average_decibel;
        // 如果当前帧的音量低于设定的阈值，则直接判定为静音帧
        if decibel < -100.0 {
            return FrameState::KFrameStateSil;
        }
        let p_silence: f32 = score[0];
        let p_speech: f32 = 1.0 - p_silence;
        let p_silence = p_silence.ln() * SPEECH_2_NOISE_RATIO;
        let p_speech = p_speech.ln();

        // 判断当前帧的语音概率是否高于静音概率+阈值 speech_noise_thres 0.6
        if p_speech.exp() >= p_silence.exp() + SPEECH_NOISE_THRES {
            // 信噪比 SNR 大于设定的阈值 -100 ，并且 分贝值大于设定的阈值 -100，则判断为语音帧
            if snr >= SNR_THRES && decibel >= DECIBEL_THRES {
                FrameState::KFrameStateSpeech
            } else {
                FrameState::KFrameStateSil
            }
        } else {
            if self.noise_average_decibel < -99.9 {
                self.noise_average_decibel = decibel;
            } else {
                // noise_frame_num_used_for_snr 为 100
                self.noise_average_decibel = (decibel
                    + self.noise_average_decibel * (NOISE_FRAME_NUM_USED_FOR_SNR - 1.0))
                    / NOISE_FRAME_NUM_USED_FOR_SNR;
            }
            FrameState::KFrameStateSil
        }
    }

    /// 检测单帧音频的状态变化
    /// # 参数
    /// * `frame_state`: 当前帧的语音状态
    fn detect_one_frame(&mut self, frame_state: FrameState) -> Option<Segment> {
        // 使用窗口检测器
        let state_change = self.windows_detector.detect_one_frame(frame_state);

        let mut state: Option<Segment> = None;
        // 根据状态变化进行相应的处理 TODO
        match state_change {
            AudioChangeState::KChangeStateSpeech2Speech => {
                // 语音到语音，持续语音段，无需特殊处理
            }
            AudioChangeState::KChangeStateSpeech2Sil => {
                // 语音到静音，结束当前语音段
                state = Some(Segment {
                    segment_type: PointType::End,
                });
            }
            AudioChangeState::KChangeStateSil2Sil => {
                // 静音到静音，持续静音段，无需特殊处理
            }
            AudioChangeState::KChangeStateSil2Speech => {
                // 静音到语音，开始新的语音段
                state = Some(Segment {
                    segment_type: PointType::Start,
                });
            }
            AudioChangeState::KChangeStateInvalid => {
                // 无效状态变化或未开始，无需处理
            }
        }
        // 多语音段检测模式：如果检测到语音结束且配置为多段模式，重置状态准备下一段
        if let Some(segment) = &state {
            if matches!(segment.segment_type, PointType::End) {
                self.reset();
            }
        }
        state
    }
}
