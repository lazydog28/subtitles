use crate::funasr::utils::constant::INTRA_THREADS;
use anyhow::{anyhow, Result};
use ort::error::Result as OrtResult;
use ort::logging::LogLevel;
use ort::session::SessionInputs;
use ort::session::SessionOutputs;
use ort::{
    execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider},
    init,
    session::{builder::GraphOptimizationLevel, Session},
};
use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

/// ONNX运行时推理会话
pub struct OrtInferSession {
    session: Session,
}

impl OrtInferSession {
    pub fn new(model_file: PathBuf) -> Result<Self> {
        Self::verify_model(&model_file)?;
        // 初始化ORT环境 (全局执行一次)
        INIT.call_once(|| {
            let _ = init();
        });
        let session = Session::builder()?
            .with_intra_threads(INTRA_THREADS)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_memory_pattern(false)?
            .with_log_level(LogLevel::Fatal)?
            .with_memory_pattern(false)?
            .with_execution_providers([
                DirectMLExecutionProvider::default()
                    .build()
                    .error_on_failure(), // 如果DirectML不可用，则使用CPU执行提供程序
                // WebGPUExecutionProvider::default().build().error_on_failure(), // 如果DirectML不可用，则使用CPU执行提供程序
                CPUExecutionProvider::default().build().error_on_failure(), // 如果DirectML不可用，则使用CPU执行提供程序
            ])?
            .commit_from_file(model_file.clone())?;

        Ok(Self { session })
    }

    pub fn run<'s, 'i, 'v: 'i, const N: usize>(
        &'s mut self,
        input_values: impl Into<SessionInputs<'i, 'v, N>>,
    ) -> OrtResult<SessionOutputs<'s>> {
        self.session.run(input_values)
    }
    pub fn get_input_names(&self) -> Vec<&str> {
        self.session
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect()
    }

    fn verify_model(model_path: &PathBuf) -> Result<()> {
        if !model_path.exists() {
            return Err(anyhow!("The {:?} does not exist.", model_path));
        }

        if !model_path.is_file() {
            return Err(anyhow!("The {:?} is not a file.", model_path));
        }

        Ok(())
    }
}
