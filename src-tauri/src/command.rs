use crate::funasr::utils::{download, microphone_stream};
use crate::funasr::{
    pretreatment, Cache, Frame, PointType, Recorder, ENDPOINT_DETECTION_WAV_FRONTEND,
    SPEECH_RECOGNITION_WAV_FRONTEND,
};
use crate::global::{
    init_lazy_lock, CONFIG, PARAFORMER, SENSE_VOICE, STOP_SPEECH_RECOGNITION, VAD
};
use crate::tray_icon::get_device_by_name;
use anyhow::Result;
use log::info;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::mem::take;
use std::sync::{Arc, Mutex};
use tauri::ipc::Channel;

/// 检查模型文件是否存在
#[tauri::command]
pub async fn models_exists() -> bool {
    if let Ok(exists) = download::models_exists() {
        exists
    } else {
        false
    }
}

/// 下载模型
#[tauri::command]
pub async fn download_models(on_event: Channel<f32>) -> Result<(), String> {
    let on_progress = |progress: f32| {
        on_event.send(progress).expect("发送进度事件失败");
    };
    download::download_and_extract_model(on_progress)
        .await
        .map_err(|e| e.to_string())
}

/// 初始化所有变量
#[tauri::command]
pub async fn init() -> Result<(), String> {
    if !models_exists().await {
        // 模型不存在则返回异常
        return Err("模型文件不存在，请重启软件".to_string());
    }
    // 初始化变量
    init_lazy_lock();
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub enum SubtitlesType {
    Online = 1,
    Offline = 2,
}

#[derive(Serialize, Deserialize)]
pub struct Subtitles {
    pub type_: SubtitlesType,
    pub msg: String,
}

#[tauri::command]
pub async fn start_speech_recognition(on_event: Channel<Subtitles>) -> Result<(), String> {
    let select_device_name = CONFIG.lock().unwrap().select_device_name.clone().unwrap();
    let device = get_device_by_name(select_device_name).unwrap();
    let recorder = Recorder::new(device);
    let recorder = Arc::new(Mutex::new(recorder));
    let stream = microphone_stream(Arc::clone(&recorder));

    // 音频数据缓存

    let mut recorder_waveform: bool = false; // 是否缓存录音数据
    let mut last_300ms: Vec<Frame> = Vec::new(); // 缓存最后30帧
    let mut remaining_waveform = Array1::<f32>::zeros(0); // 录音数据转为音频帧时剩余音频数据

    let mut vad_remaining_frames: Vec<Frame> = Vec::new(); // 缓存 vad 提取特征剩余音频帧

    let mut paraformer_frames: Vec<Frame> = Vec::new(); // 为 paraformer 储存音频帧
    let mut cache = Cache::default(); // 缓存 paraformer 的中间结果
    let mut paraformer_remaining_frames: Vec<Frame> = Vec::new();

    let mut sense_voice_frames: Vec<Frame> = Vec::new(); // 为 sense_voice 储存音频帧

    for audio_data in stream {
        // 检查退出标志
        let should_exit = {
            let flag = STOP_SPEECH_RECOGNITION.lock().expect("获取退出标志锁失败");
            *flag
        };
        if should_exit {
            info!("停止语音识别运行");
            break;
        }

        // 提取音频帧
        let (frames, remaining_waveform_tmp) = pretreatment(audio_data, remaining_waveform);
        remaining_waveform = remaining_waveform_tmp;

        // 如果处于录音状态 则缓存
        if recorder_waveform {
            sense_voice_frames.extend(frames.clone());
            paraformer_frames.extend(frames.clone());
        }
        // 提取VAD特征 将音频帧转为特征向量
        vad_remaining_frames.extend(frames.clone()); // 将本次帧加入缓存
        let (features, vad_remaining_frames_tmp) =
            ENDPOINT_DETECTION_WAV_FRONTEND.extract_features(&vad_remaining_frames);

        last_300ms.extend(frames.clone());
        if last_300ms.len() > 30 {
            // 只保留最新的300ms音频
            last_300ms.drain(0..last_300ms.len() - 30);
        }


        let segments = VAD.lock().unwrap().call(features, &vad_remaining_frames).map_err(|e| e.to_string())?;
        vad_remaining_frames = vad_remaining_frames_tmp; // 缓存剩余的帧
        for segment in segments {
            match segment.segment_type {
                PointType::Start => {
                    recorder_waveform = true;
                    paraformer_frames = last_300ms.clone();
                    sense_voice_frames = take(&mut last_300ms);
                }
                PointType::End => {
                    recorder_waveform = false;
                    let (features, _) =
                        SPEECH_RECOGNITION_WAV_FRONTEND.extract_features(&sense_voice_frames);
                    let result = SENSE_VOICE.lock().unwrap().call(features).map_err(|e| e.to_string())?;

                    on_event
                        .send(Subtitles {
                            type_: SubtitlesType::Offline,
                            msg: result,
                        }).map_err(|e| e.to_string())?;
                    take(&mut paraformer_frames);
                    take(&mut paraformer_remaining_frames);
                    take(&mut sense_voice_frames);
                    cache = Cache::default();
                }
            }
        }

        if recorder_waveform && paraformer_frames.len() > 60 {
            paraformer_remaining_frames.extend(paraformer_frames);
            let (features, paraformer_remaining_frames_tmp) =
                SPEECH_RECOGNITION_WAV_FRONTEND.extract_features(&paraformer_remaining_frames);
            paraformer_remaining_frames = paraformer_remaining_frames_tmp;
            let word = PARAFORMER.lock().unwrap().call(features, &mut cache).map_err(|e| e.to_string())?;
            on_event
                .send(Subtitles {
                    type_: SubtitlesType::Online,
                    msg: word,
                })
                .expect("发送消息事件失败");
            paraformer_frames = Vec::new();
        }
    }

    Ok(())
}

#[tauri::command]
pub fn stop_speech_recognition() {
    let mut flag = STOP_SPEECH_RECOGNITION.lock().unwrap();
    *flag = true;
}