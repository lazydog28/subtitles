use crate::funasr::utils::{E2EVadModel, Frame, OrtInferSession, Segment};
use anyhow::{anyhow, Ok, Result};
use ndarray::{Array2, Axis};
use ort::inputs;
use ort::value::{Tensor, Value};
use std::path::PathBuf;
use std::usize;
const FSMN_LAYERS: usize = 4;
const PROJ_DIM: usize = 128;
const LORDER: usize = 20;

pub struct Vad {
    session: OrtInferSession,
    scorer: E2EVadModel,

    in_cache: Vec<Value>,
}

impl Vad {
    /// 创建 VAD 实例
    /// # Arguments
    /// * `model_dir` - 模型文件目录，默认 "models"
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        let model_dir = model_dir.unwrap_or_else(|| PathBuf::from("models"));
        let model_path = model_dir.join("vad.onnx");
        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {}", model_path.display()));
        }
        let session = OrtInferSession::new(model_path)?;

        let mut in_cache = Vec::with_capacity(FSMN_LAYERS);
        for _ in 0..FSMN_LAYERS {
            in_cache.push(
                Tensor::from_array(ndarray::Array4::<f32>::zeros((1, PROJ_DIM, LORDER - 1, 1)))?
                    .into(),
            );
        }
        Ok(Self {
            session,
            scorer: E2EVadModel::default(),
            in_cache,
        })
    }

    pub fn call(&mut self, features: Array2<f32>, frames: &Vec<Frame>) -> Result<Vec<Segment>> {
        let scores = self.infer(features)?;
        Ok(self.scorer.call(scores, frames))
    }

    fn infer(&mut self, features: Array2<f32>) -> Result<Array2<f32>> {
        // 构造输入张量
        let mut inputs = inputs![
            "speech"=>Tensor::from_array(features.insert_axis(Axis(0)))?,
        ];
        for (i, cache) in self.in_cache.iter().enumerate() {
            inputs.push((format!("in_cache{}", i).into(), cache.into()));
        }
        let mut result = self.session.run(inputs)?;
        let scores = Self::extract_scores(&result[0])?;
        let mut new_caches: Vec<ort::value::Value> = Vec::new();
        for i in 0..FSMN_LAYERS {
            new_caches.push(
                result
                    .remove(format!("out_cache{}", i))
                    .expect("语音端点检测提取缓存失败"),
            );
        }
        self.in_cache = new_caches;
        Ok(scores)
    }

    /// 提取得分数据
    fn extract_scores(scores_tensor: &ort::value::Value) -> Result<Array2<f32>> {
        let scores_array = scores_tensor.try_extract_array()?;
        let shape = scores_array.shape();
        let dim = shape[1];
        let cls = shape[2];
        let scores: Array2<f32> = scores_array.to_shape((dim, cls))?.to_owned();
        Ok(scores)
    }
}
