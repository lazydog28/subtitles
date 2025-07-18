use env_logger::builder;
use std::fs;
use std::path::Path;
use std::time::SystemTime;
use log::LevelFilter::Debug;

/// 初始化日志文件
fn get_file_age_days(path: &Path) -> Option<u64> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let now = SystemTime::now();
    let duration = now.duration_since(modified).ok()?;
    Some(duration.as_secs() / (24 * 60 * 60))
}

fn clean_old_logs() {
    if let Ok(entries) = fs::read_dir("logs") {
        for entry in entries.flatten() {
            if let Some(age) = get_file_age_days(&entry.path()) {
                if age > 3 {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
}

pub fn init() {
    // 判断 logs 文件夹是否存在，如果不存在则创建
    if !fs::metadata("logs").is_ok() {
        fs::create_dir("logs").expect("Failed to create logs directory");
    }

    clean_old_logs();

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if let Ok(file) = fs::File::create(format!("logs/log_{}.txt", timestamp)) {
        builder()
            .target(
                env_logger::Target::Pipe(Box::new(file)))
            .filter_level(Debug)
            .init();
    }
}
