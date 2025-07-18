use crate::funasr::{default_device, Language, Paraformer, SenseVoice, Vad};
use cpal::traits::DeviceTrait;
use std::sync::{Arc, LazyLock, Mutex};

// 将所有 LazyLock 变量初始化
pub fn init_lazy_lock() {
    let _ = &*CONFIG;
    let _ = &*VAD;
    let _ = &*SENSE_VOICE;
    let _ = &*PARAFORMER;
}

pub static CONFIG: LazyLock<Arc<Mutex<Config>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Config::default())));

pub static VAD: LazyLock<Arc<Mutex<Vad>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(
        Vad::new(None).expect("Failed to create Vad instance"),
    ))
});

pub static SENSE_VOICE: LazyLock<Arc<Mutex<SenseVoice>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(
        SenseVoice::new(None, None).expect("Failed to create SenseVoice instance"),
    ))
});

pub static PARAFORMER: LazyLock<Arc<Mutex<Paraformer>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(
        Paraformer::new(None).expect("Failed to create Paraformer instance"),
    ))
});

pub static STOP_SPEECH_RECOGNITION:LazyLock<Arc<Mutex<bool>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(false))
});

pub struct Config {
    pub select_device_name: Option<String>,
    pub language: Language,
}

impl Default for Config {
    fn default() -> Self {
        let default_device_name: Option<String> = {
            if let Some(device) = default_device() {
                if let Ok(device_name) = device.name() {
                    Some(device_name)
                } else {
                    None
                }
            } else {
                None
            }
        };

        Self {
            select_device_name: default_device_name,
            language: Language::Chinese,
        }
    }
}
