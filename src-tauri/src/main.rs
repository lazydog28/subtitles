#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] //阻止出现额外的cmd窗口

fn main() {
    app_lib::run();
}
