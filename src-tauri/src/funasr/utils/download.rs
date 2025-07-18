use anyhow::{anyhow, Result};
use md5::{Digest, Md5};
use reqwest::Client;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::LazyLock;
use tokio::fs::{remove_file, File as AsyncFile};
use tokio::io::AsyncWriteExt;
use zip;

// const MODEL_FILE_URLS: [&str; 1] = ["http://rustfs.lazydog.site/download/funasr.zip"];
const MODEL_FILE_URLS: [&str; 1] = ["http://rustfs.lazydog.site/download/funasr.zip"];
const MODEL_FILE_MD5: &str = "a8f50347f388863b59dcc42927428e7f";
const MODEL_FILE_NAME: &str = "funasr.zip";
const MODEL_PATH: &str = "models";
static MODEL_FILES: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "paraformer-decoder.onnx",
        "a9778230fe187338c21b2cecba92ccea",
    );
    map.insert(
        "paraformer-encoder.onnx",
        "2016e5686aeadd194572e17c77aaae2f",
    );
    map.insert("paraformer-tokens.txt", "f51cac5badc6cf4a2fd25e0910f20cd5");
    map.insert("sense-voice.onnx", "120cc8b9885544d3b89546dd66df03ab");
    map.insert("sense-voice-tokens.txt", "17cdd9d527315e28553153706d23c719");
    map.insert("vad.onnx", "8bb44d7f59905c08bad70840d0290c33");
    map
});

/// 判断模型文件是否存在
pub fn models_exists() -> Result<bool> {
    let mut need_download: bool = false;
    // 判断 MODEL_PATH 目录下是否存在 MODEL_FILES 且md5是否一致
    for (file, md5_value) in MODEL_FILES.iter() {
        if let Ok(success) = check_md5(&format!("models/{}", file), md5_value) {
            // 如果不一致则需要下载
            need_download = !success
        }
        if need_download {
            break;
        }
    }
    // 如果需要下载则文件不存在
    if need_download {
        Ok(false)
    } else {
        Ok(true)
    }
}

/// 函数流式下载模型文件到指定路径
/// # 参数
/// - `url`: 模型文件的下载链接
/// - `dest`: 下载后保存的目标路径
/// # 返回
///     迭代器 实时返回Result 下载进度，成功时为下载进度，失败时为错误信息
async fn download_model_file<F>(url: &str, progress_callback: F) -> Result<()>
where
    F: Fn(f32) -> (),
{
    let client = Client::new();
    let url = reqwest::Url::parse(url).expect("下载链接异常");
    let dest = Path::new("funasr.zip");
    if dest.exists() {
        // 删除文件
        remove_file(dest).await.expect("删除已存在文件失败");
    }
    let mut response = client.get(url).send().await.expect("发送下载请求失败");
    let total_size = response.content_length().expect("获取文件大小失败");
    let mut file = AsyncFile::create(dest).await.expect("创建文件失败");
    let mut downloaded: u64 = 0;

    while let Ok(chunk) = response.chunk().await {
        if let Some(chunk) = chunk {
            file.write_all(&chunk).await.expect("文件写入时失败");
            downloaded += chunk.len() as u64;
            let progress = downloaded as f32 / total_size as f32;
            progress_callback(progress);
        } else {
            break;
        }
    }
    file.flush().await.expect("文件写入失败");
    Ok(())
}

/// 解压缩文件到指定目录
/// # 参数
/// - `src`: 源文件路径
/// - `dest`: 目标目录路径
fn unzip(src: &str, dest: &str) -> Result<()> {
    let file = File::open(src).expect("打开压缩文件失败");
    let mut archive = zip::ZipArchive::new(file).expect("创建ZipArchive失败");
    archive.extract(dest).expect("解压缩文件失败");
    Ok(())
}

/// 检查文件的MD5值是否与预期匹配
/// # 参数
/// - `file_path`: 要检查的文件路径
/// - `expected_md5`: 预期的MD5值
/// # 返回
/// - `Result<bool>`: 如果文件的MD5值与预期匹配，返回`Ok(true)`，否则返回`Ok(false)`，如果发生错误则返回`Err`
fn check_md5(file_path: &str, expected_md5: &str) -> Result<bool> {
    // 判断文件是否存在
    if !Path::new(file_path).exists() {
        return Ok(false);
    }
    let mut file = File::open(file_path).expect("打开文件失败");
    let mut hasher = Md5::new();
    std::io::copy(&mut file, &mut hasher).expect("计算MD5失败");
    let result = hasher.finalize();
    let md5_str = format!("{:x}", result);
    Ok(md5_str == expected_md5)
}

/// 下载并解压模型文件
/// # 参数
/// - `dest`: 下载后保存的目标路径
/// # 返回
/// - `Result<()>`: 如果下载和解压成功，返回`Ok(())`，否则返回错误信息
pub async fn download_and_extract_model<F>(progress_callback: F) -> Result<()>
where
    F: Fn(f32) -> (),
{
    let mut download_success = false;
    for url in MODEL_FILE_URLS.iter() {
        if let Ok(_) = download_model_file(url, &progress_callback).await {
            download_success = check_md5(MODEL_FILE_NAME, MODEL_FILE_MD5)?;
        }
        if download_success {
            break;
        }
    }
    if !download_success {
        return Err(anyhow!("下载模型文件失败"));
    }
    unzip(MODEL_FILE_NAME, MODEL_PATH)?;
    remove_file(MODEL_FILE_NAME)
        .await
        .expect("删除临时压缩文件失败");
    Ok(())
}
