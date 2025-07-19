use crate::funasr::{devices, hosts, Language, Recorder};
use crate::global::{get_device_by_name, CONFIG, RECORDER, SENSE_VOICE};
use cpal::{traits::DeviceTrait, Device};
use log::{debug, info};
use tauri::tray::MouseButton::Left;
use tauri::tray::TrayIconEvent;
use tauri::{
    menu::{CheckMenuItem, CheckMenuItemBuilder, Menu, MenuItem, Submenu, SubmenuBuilder},
    tray::TrayIconBuilder,
    App, AppHandle, Manager, Wry,
};

pub(crate) fn setup_tray_icon(app: &App) {
    let exit_menu = MenuItem::with_id(app, "exit", "退出", true, None::<&str>).unwrap();
    let device_menu = create_device_menu(app.handle());
    let language_menu = create_language_menu(app.handle());
    let menu = Menu::with_items(app, &[&device_menu, &language_menu, &exit_menu]).unwrap();

    TrayIconBuilder::new()
        .show_menu_on_left_click(false)
        .on_tray_icon_event(|tray_icon, event| match event {
            TrayIconEvent::DoubleClick {
                id: _,
                position: _,
                rect: _,
                button: Left,
            }
            | TrayIconEvent::Click {
                id: _,
                position: _,
                rect: _,
                button: Left,
                button_state: _,
            } => {
                let app_handle = tray_icon.app_handle();
                if let Some(window) = app_handle.get_webview_window("main") {
                    debug!("显示窗口");
                    window.unminimize().expect("unminimize window failed");
                };
            }
            _ => {}
        })
        .on_menu_event(move |app_handle, event| {
            let event_id = event.id.as_ref();
            match event_id {
                "exit" => {
                    info!("退出程序");
                    exit(app_handle);
                }
                _ => {
                    if event_id.starts_with("input_device_") {
                        change_select_device_name(
                            event_id.replace("input_device_", ""),
                            &device_menu,
                        );
                    }
                    if event_id.starts_with("language_") {
                        change_select_language(event_id.replace("language_", ""), &language_menu);
                    }
                }
            }
        })
        .menu(&menu)
        .icon(app.default_window_icon().unwrap().clone())
        .build(app)
        .unwrap();
}

/// 获取所有的输入设备
pub fn all_input_devices() -> Vec<Device> {
    let mut input_devices: Vec<Device> = Vec::new();
    for host in hosts() {
        if let Ok(host_devices) = devices(host) {
            for device in host_devices {
                input_devices.push(device);
            }
        }
    }
    input_devices
}

/// 创建一个输入设备菜单项
fn device_menu_item(app: &AppHandle, device: &Device) -> Option<CheckMenuItem<Wry>> {
    if let Err(_) = device.name() {
        return None; // 如果设备名称无法获取，则跳过该设备
    }

    let device_name = device.name().unwrap();
    let select_device_name = { CONFIG.lock().unwrap().select_device_name.clone().unwrap() };
    Some(
        CheckMenuItemBuilder::with_id(format!("input_device_{}", device_name), device_name.clone())
            .checked(device_name == select_device_name)
            .build(app)
            .unwrap(),
    )
}

/// 创建一个输入设备菜单
fn create_device_menu(app: &AppHandle) -> Submenu<Wry> {
    let input_devices = all_input_devices();
    let device_menu_items: Vec<CheckMenuItem<Wry>> = input_devices
        .iter()
        .filter_map(|device| device_menu_item(app, device))
        .collect();
    if device_menu_items.is_empty() {
        return SubmenuBuilder::with_id(app, "input_device", "无可用输入设备")
            .enabled(false)
            .build()
            .unwrap();
    }
    let mut menu = SubmenuBuilder::with_id(app, "input_device", "输入设备");
    for item in device_menu_items {
        menu = menu.item(&item);
    }
    menu.build().unwrap()
}
/// 退出程序
fn exit(app: &AppHandle) {
    // 关闭所有窗口
    for (_, window) in app.webview_windows() {
        window.close().unwrap()
    }

}
/// 修改选中的设备名称
fn change_select_device_name(device_name: String, device_menu: &Submenu<Wry>) {
    // 修改配置
    let mut config = CONFIG.lock().unwrap();
    config.select_device_name = Some(device_name.clone());
    // 修改 RECORDER
    let mut recorder = RECORDER.lock().unwrap();
    let device = get_device_by_name(device_name).unwrap();
    let recorder_ins = Recorder::new(device);
    *recorder = recorder_ins;
    
    if let Some(select_device_name) = config.select_device_name.clone() {
        for item in device_menu.items().unwrap() {
            if let Some(check_item) = item.as_check_menuitem() {
                if check_item.id().as_ref().starts_with("input_device_") {
                    let item_device_name = check_item.id().as_ref().replace("input_device_", "");
                    check_item
                        .set_checked(item_device_name == select_device_name)
                        .unwrap();
                }
            }
        }
    }
}
/// 语言 菜单项
fn language_menu_item(app: &AppHandle, language: &Language) -> CheckMenuItem<Wry> {
    let language_name = language.to_string();
    let select_language = { CONFIG.lock().unwrap().language.to_string() };
    CheckMenuItemBuilder::with_id(format!("language_{}", language_name), language_name.clone())
        .checked(language_name == select_language)
        .build(app)
        .unwrap()
}

/// 创建一个语言菜单
fn create_language_menu(app: &AppHandle) -> Submenu<Wry> {
    let languages = Language::all();
    let language_menu_items: Vec<CheckMenuItem<Wry>> = languages
        .iter()
        .map(|language| language_menu_item(app, language))
        .collect();
    let mut menu = SubmenuBuilder::with_id(app, "language", "语言");
    for item in language_menu_items {
        menu = menu.item(&item);
    }
    menu.build().unwrap()
}

/// 修改选中的语言
fn change_select_language(language_name: String, language_menu: &Submenu<Wry>) {
    let mut config = CONFIG.lock().unwrap();
    let mut sense_voice = SENSE_VOICE.lock().unwrap();
    if let Ok(language) = Language::from_str(&language_name) {
        config.language = language;
        sense_voice.language = language;
        println!("已切换语言: {}", language_name);
    } else {
        eprintln!("无法识别的语言: {}", language_name);
        return;
    }
    for item in language_menu.items().unwrap() {
        if let Some(check_item) = item.as_check_menuitem() {
            if check_item.id().as_ref().starts_with("language_") {
                let item_language_name = check_item.id().as_ref().replace("language_", "");
                check_item
                    .set_checked(item_language_name == config.language.to_string())
                    .unwrap();
            }
        }
    }
}
