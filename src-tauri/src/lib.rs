mod command;
mod funasr;
mod global;
mod tray_icon;
mod utils;

use command::*;
use log::info;
use tauri_plugin_dialog;
use tray_icon::setup_tray_icon;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    utils::init_log();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            models_exists,
            download_models,
            init,
            start_speech_recognition,
            stop_speech_recognition
        ])
        .setup(|app| {
            setup_tray_icon(app);
            info!("启动成功");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
