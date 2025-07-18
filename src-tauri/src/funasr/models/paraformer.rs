use crate::funasr::utils::constant::{MEL_BINS, SPEECH_RECOGNITION_LFR_M};
use crate::funasr::utils::{read_token, OrtInferSession, TokenIdConverter};
use anyhow::{anyhow, Ok, Result};
use ndarray::{concatenate, s, Array1, Array2, Array3, ArrayView1, Axis};
use ort::inputs;
use ort::value::{Tensor, Value};
use std::f32::consts::E;
use std::path::PathBuf;

const CHUNK_SIZE_PRE: usize = 5; // 5帧前置
const CHUNK_SIZE_BACK: usize = 5; // 5帧后置
const CHUNK_SIZE: usize = 10; // 10帧中间
const OUTPUT_SIZE: usize = 512; // 输出特征维度 config["encoder_conf"]["output_size"]
const FSMN_LAYER: usize = 16; // fsmn层数
const FSMN_LORDER: usize = 10; // config["decoder_conf"]["kernel_size"] - 1
const CIF_THRESHOLD: f32 = 1.0; // cif阈值

pub struct Cache {
    feats: Array2<f32>,
    start_idx: usize,
    cif_hidden: Array1<f32>,
    cif_alphas: f32,
    decoder_fsmn: Array3<f32>, // 16,512,10
}
impl Default for Cache {
    fn default() -> Self {
        Self {
            feats: Array2::zeros((
                CHUNK_SIZE_PRE + CHUNK_SIZE_BACK,
                SPEECH_RECOGNITION_LFR_M * MEL_BINS,
            )),
            start_idx: 0,
            cif_hidden: Array1::zeros(OUTPUT_SIZE),
            cif_alphas: 0f32,
            decoder_fsmn: Array3::zeros((FSMN_LAYER, OUTPUT_SIZE, FSMN_LORDER)),
        }
    }
}

pub struct Paraformer {
    token_converter: TokenIdConverter,
    encoder_session: OrtInferSession,
    decoder_session: OrtInferSession,
}

impl Paraformer {
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        let model_dir = model_dir.unwrap_or_else(|| PathBuf::from("models"));
        if !model_dir.exists() {
            return Err(anyhow!("Model directory does not exist: {:?}", model_dir));
        }
        let encoder_model_file = model_dir.join("paraformer-encoder.onnx");
        let decoder_model_file = model_dir.join("paraformer-decoder.onnx");
        let encoder_session = OrtInferSession::new(encoder_model_file)?;
        let decoder_session = OrtInferSession::new(decoder_model_file)?;

        let token_converter = read_token(model_dir.join("paraformer-tokens.txt"))?;
        Ok(Self {
            token_converter,
            encoder_session,
            decoder_session,
        })
    }

    /// 处理音频特征并进行推理
    /// 返回识别结果
    /// # 参数
    /// - `features`: 特征
    /// - `cache`: 缓存
    pub fn call(&mut self, mut features: Array2<f32>, cache: &mut Cache) -> Result<String> {
        // 检查输入是否小于   chunk_len = chunk_size[1]*frame_shift*lfr_n*offline_handle_->GetAsrSampleRate()/1000;
        if features.shape()[0] < 10 {
            println!("Paraformer 实时识别 理想输入长度为600ms语音")
        }
        // 特征缩放 feats *= self.encoder_output_size**0.5
        features.mapv_inplace(|x| x * (OUTPUT_SIZE as f32).sqrt());
        // fbank -> position encoding -> overlap chunk
        features = forward(features, cache.start_idx);
        let features_count = features.shape()[0];
        features = self.add_overlap_chunk(features, cache);
        let results = self.infer(features, cache)?;
        cache.start_idx += features_count;
        Ok(results)
    }
    /// 添加重叠块
    fn add_overlap_chunk(&mut self, features: Array2<f32>, cache: &mut Cache) -> Array2<f32> {
        let features = concatenate![Axis(0), cache.feats, features];
        let cache_feats_start = features.shape()[0] - CHUNK_SIZE_PRE - CHUNK_SIZE_BACK;
        cache.feats = features.slice(s![cache_feats_start.., ..]).to_owned();
        features
    }

    /// infer预测
    fn infer(&mut self, features: Array2<f32>, cache: &mut Cache) -> Result<String> {
        // features 添加批次维度
        let features = features.insert_axis(Axis(0));
        let features_len = Array1::from(vec![features.shape()[1] as i32]);
        let (enc, enc_len, alphas) = {
            let result = self.encoder_session.run(inputs![
                Tensor::from_array(features)?,
                Tensor::from_array(features_len)?,
            ])?;

            let enc_value = result.get("enc").unwrap().try_extract_array()?;

            let enc: Array2<f32> = enc_value
                .to_shape((enc_value.shape()[1], enc_value.shape()[2]))?
                .to_owned();

            let enc_len_value = result.get("enc_len").unwrap().try_extract_array()?;

            let enc_len: Array1<i32> = enc_len_value.to_shape(enc_value.shape()[0])?.to_owned();

            let alphas_value = result.get("alphas").unwrap().try_extract_array()?;
            let alphas: Array1<f32> = alphas_value
                .to_shape((alphas_value.shape()[1],))?
                .to_owned();

            (enc, enc_len, alphas)
        };
        let acoustic_embeds = self.cif_search(enc.clone(), alphas, cache)?;

        if acoustic_embeds.shape()[0] > 0 {
            let logits = {
                // 准备解码器输入
                let enc_3d = enc.insert_axis(Axis(0)); // 添加批次维度
                let enc_len_array = Array1::from(vec![enc_len[0]]);
                let acoustic_embeds_len = acoustic_embeds.shape()[0] as i32;
                let acoustic_embeds_3d = acoustic_embeds.insert_axis(Axis(0)); // 添加批次维度
                let acoustic_embeds_len_array = Array1::from(vec![acoustic_embeds_len]);

                // 使用宏构建解码器输入
                let mut decoder_inputs = inputs![
                    "enc" => Tensor::from_array(enc_3d)?,
                    "enc_len" => Tensor::from_array(enc_len_array)?,
                    "acoustic_embeds" => Tensor::from_array(acoustic_embeds_3d)?,
                    "acoustic_embeds_len" => Tensor::from_array(acoustic_embeds_len_array)?,
                ];
                // 添加缓存
                for (index, decoder_fsmn) in cache.decoder_fsmn.axis_iter(Axis(0)).enumerate() {
                    let in_cache_3d = decoder_fsmn.insert_axis(Axis(0));
                    decoder_inputs.push((
                        format!("in_cache_{}", index).into(),
                        Value::from_array(in_cache_3d.to_owned())?.into(),
                    ));
                }

                let decoder_result = self.decoder_session.run(decoder_inputs)?;
                let logits_value = &decoder_result.get("logits").unwrap().try_extract_array()?;
                let logits: Array2<f32> = logits_value
                    .to_shape((logits_value.shape()[1], logits_value.shape()[2]))?
                    .to_owned();
                for i in 0..FSMN_LAYER {
                    // tensor: float32[batch_size,512,Sliceout_cache_0_dim_2]
                    let out_cache_value = &decoder_result[i + 2].try_extract_array()?;
                    let out_cache: Array2<f32> = out_cache_value
                        .to_shape((out_cache_value.shape()[1], out_cache_value.shape()[2]))?
                        .to_owned(); //512 10
                    cache
                        .decoder_fsmn
                        .slice_mut(s![i, .., ..])
                        .assign(&out_cache);
                }
                logits
            };
            let result = self.decode(logits);
            return Ok(result);
        }
        Ok("".to_string())
    }

    fn cif_search(
        &mut self,
        hidden: Array2<f32>,
        mut alphas: Array1<f32>,
        cache: &mut Cache,
    ) -> Result<Array2<f32>> {
        let (_, hidden_size) = hidden.dim();

        // 初始化变量
        let mut list_frame = Vec::new();
        let mut frame_timestamp = Vec::new();

        // alphas[:self.chunk_size[0]] = 0.0
        for i in 0..CHUNK_SIZE_PRE.min(alphas.len()) {
            alphas[i] = 0.0; // 0~5
        }
        // alphas[sum(self.chunk_size[:2]):] = 0.0
        let sum_first_two = CHUNK_SIZE_PRE + CHUNK_SIZE;
        for i in sum_first_two..alphas.len() {
            alphas[i] = 0.0; // 15~最后
        }

        // 处理缓存中的数据
        let (final_hidden, final_alphas) = if cache.cif_alphas != 0.0 {
            // 如果缓存中有数据，进行拼接
            let cache_hidden = cache.cif_hidden.clone().insert_axis(Axis(0));
            let cache_alphas_array = Array1::from(vec![cache.cif_alphas]);

            let concatenated_hidden = ndarray::concatenate![Axis(0), cache_hidden, hidden];
            let concatenated_alphas = ndarray::concatenate![Axis(0), cache_alphas_array, alphas];

            (concatenated_hidden, concatenated_alphas)
        } else {
            (hidden, alphas)
        };

        let len_time = final_alphas.len();

        // CIF 搜索主逻辑
        let mut integrate = 0.0f32;
        let mut frames = Array1::zeros(hidden_size);

        for t in 0..len_time {
            let alpha = final_alphas[t];
            // 如果alpha + integrate < CIF_THRESHOLD 则继续累加
            if alpha + integrate < CIF_THRESHOLD {
                integrate += alpha;
                // frames += alpha * hidden[t]
                let weighted_hidden = &final_hidden.row(t) * alpha;
                frames = &frames + &weighted_hidden;
            } else {
                // 记录触发峰值的帧数 在字符开始时触发  因此相邻两个峰值之间的帧被视为前一个字符的持续时间
                // frames += (CIF_THRESHOLD - integrate) * hidden[t]
                let weight = CIF_THRESHOLD - integrate;
                let weighted_hidden = &final_hidden.row(t) * weight;
                frames = &frames + &weighted_hidden;

                list_frame.push(frames.clone());

                frame_timestamp.push(cache.start_idx + t);

                integrate += alpha;
                integrate -= CIF_THRESHOLD;

                // frames = integrate * hidden[t]
                frames = &final_hidden.row(t) * integrate;
            }
        }

        // 更新缓存
        cache.cif_alphas = integrate;
        if integrate > 0.0 {
            cache.cif_hidden = &frames / integrate;
        } else {
            cache.cif_hidden = frames.clone();
        }

        let token_length = list_frame.len() as i32;

        // 创建结果数组
        let result_frames = if !list_frame.is_empty() {
            let mut result = Array2::zeros((token_length as usize, hidden_size));
            for (i, frame) in list_frame.iter().enumerate() {
                result.row_mut(i).assign(frame);
            }
            result
        } else {
            Array2::zeros((0, hidden_size))
        };

        Ok(result_frames)
    }

    fn decode(&self, logits: Array2<f32>) -> String {
        // 获取每个时间步的最大概率索引 (argmax)
        let token_int: Vec<usize> = logits
            .outer_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            })
            .collect();

        // 过滤掉空白符号 (0) 和 EOS (2)
        let filtered_token_int: Vec<usize> = token_int
            .into_iter()
            .filter(|&x| x != 0 && x != 2)
            .collect();

        // 将整数ID转换为token
        let tokens = self.token_converter.ids2tokens(&filtered_token_int);
        let tokens: String = tokens.join("");
        tokens
    }
}

/// 位置编码
pub fn forward(x: Array2<f32>, start_idx: usize) -> Array2<f32> {
    pub fn encode(positions: ArrayView1<f32>, depth: usize) -> Array2<f32> {
        let timesteps = positions.len();
        let positions = positions.to_owned();

        // 计算时间尺度增量
        let log_timescale_increment = 10000.0_f32.ln() / (depth as f32 / 2.0 - 1.0);

        // 计算逆时间尺度
        let inv_timescales: Array1<f32> = Array1::from_iter(
            (0..depth / 2).map(|i| E.powf(i as f32 * (-log_timescale_increment))),
        );

        // 重塑逆时间尺度 - 现在是 (timesteps, depth/2)
        let inv_timescales = inv_timescales
            .broadcast((timesteps, depth / 2))
            .unwrap()
            .to_owned();

        // 计算缩放时间
        let positions_reshaped = positions.insert_axis(Axis(1));
        let scaled_time = &positions_reshaped * &inv_timescales;

        // 计算正弦和余弦编码
        let sin_encoding = scaled_time.map(|&x| x.sin());
        let cos_encoding = scaled_time.map(|&x| x.cos());

        // 沿着最后一个轴连接
        let encoding = ndarray::concatenate![Axis(1), sin_encoding, cos_encoding];

        encoding
    }

    let (timesteps, input_dim) = x.dim();

    // 创建位置数组
    let positions: Array1<f32> =
        Array1::from_iter((1..timesteps + 1 + start_idx).map(|i| i as f32));

    // 获取位置编码
    let position_encoding = encode(positions.view(), input_dim);

    // 截取相应的时间步
    let sliced_encoding = position_encoding.slice(s![start_idx..start_idx + timesteps, ..]);

    // 添加位置编码到输入
    x + &sliced_encoding
}
