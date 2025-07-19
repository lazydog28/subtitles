use crate::funasr::utils::{read_token, OrtInferSession, TokenIdConverter};
use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use ort::inputs;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub struct SenseVoice {
    session: OrtInferSession,
    token_converter: TokenIdConverter,
    pub language: Language,
}
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Language {
    Auto = 0,
    Chinese = 3,
    English = 4,
    Cantonese = 7,
    Japanese = 11,
    Korean = 12,
    NoSpeech = 13,
}
impl Language {
    pub fn to_string(&self) -> String {
        match self {
            Language::Auto => "自动".to_string(),
            Language::Chinese => "中文".to_string(),
            Language::English => "英文".to_string(),
            Language::Cantonese => "粤语".to_string(),
            Language::Japanese => "日语".to_string(),
            Language::Korean => "韩语".to_string(),
            Language::NoSpeech => "无语音".to_string(),
        }
    }

    pub fn all() -> Vec<Language> {
        vec![
            Language::Auto,
            Language::Chinese,
            Language::English,
            Language::Cantonese,
            Language::Japanese,
            Language::Korean,
            Language::NoSpeech,
        ]
    }

    pub fn from_str(name: &str) -> Result<Language> {
        match name {
            "自动" => Ok(Language::Auto),
            "中文" => Ok(Language::Chinese),
            "英文" => Ok(Language::English),
            "粤语" => Ok(Language::Cantonese),
            "日语" => Ok(Language::Japanese),
            "韩语" => Ok(Language::Korean),
            "无语音" => Ok(Language::NoSpeech),
            _ => Err(anyhow!("无法识别的语言: {}", name)),
        }
    }
}

const BLANK_ID: usize = 0;

impl SenseVoice {
    pub fn new(model_dir: Option<PathBuf>, language: Option<Language>) -> Result<Self> {
        let model_dir = model_dir.unwrap_or_else(|| PathBuf::from("models"));
        if !model_dir.exists() {
            return Err(anyhow!("Model directory does not exist: {:?}", model_dir));
        }
        let model_file = model_dir.join("sense-voice.onnx");
        let session = OrtInferSession::new(model_file)?;
        let token_converter = read_token(model_dir.join("sense-voice-tokens.txt"))?;
        let language = language.unwrap_or(Language::Chinese);
        Ok(Self {
            session,
            token_converter,
            language,
        })
    }
    /// 参数
    /// - `features`:特征
    pub fn call(&mut self, features: Array2<f32>) -> Result<String> {
        self.inter(features)
    }

    fn inter(&mut self, feats: Array2<f32>) -> Result<String> {
        let x_length = feats.shape()[0];
        let feats = feats.insert_axis(Axis(0));
        let language = self.language.clone() as i32;
        let logits = {
            let mut result = self.session.run(inputs![
              "speech"=>Tensor::from_array(feats)?,
              "speech_lengths"=>Tensor::from_array(([1], vec![x_length as i32]))?,
              "language"=>Tensor::from_array(([1], vec![language]))?,
              "textnorm"=>Tensor::from_array(([1], vec![15i32]))?,
            ])?;
            let logits_tensor = result.remove("ctc_logits").unwrap();
            let logits = logits_tensor.try_extract_array()?;
            let logits: Array2<f32> = logits
                .to_shape((logits.shape()[1], logits.shape()[2]))?
                .to_owned();
            logits
        };
        let decoded_text = self.decode(logits)?;
        Ok(decoded_text)
    }

    fn decode(&self, logits: Array2<f32>) -> Result<String> {
        // 获取每个时间步长中概率最大的
        let max_indices = logits
            .axis_iter(Axis(0))
            .map(|t| {
                t.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as i32)
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let mut merged_indices = Vec::new();
        let mut current = max_indices[0];
        for &index in &max_indices[1..] {
            if index == current {
                continue;
            }
            merged_indices.push(current);
            current = index;
        }
        merged_indices.push(current);
        // 移除 BLANK_ID
        merged_indices.retain(|&id| id != BLANK_ID as i32);
        // 将 merged_indices 转换为文本
        let merged_indices_usize: Vec<usize> = merged_indices.iter().map(|&x| x as usize).collect();
        let tokens: Vec<String> = self.token_converter.ids2tokens(&merged_indices_usize);
        let decoded_text = tokens.join("");
        Ok(decoded_text)
    }
}
