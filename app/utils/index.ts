import {invoke, Channel} from "@tauri-apps/api/core";
import {message as dialog_message, type MessageDialogOptions} from '@tauri-apps/plugin-dialog';

/*
消息提示框
 */
export async function message(msg: string, options?: MessageDialogOptions) {
    return dialog_message(msg, options)
}


/*
检查模型文件是否存在
 */
export async function models_exists(): Promise<boolean> {
    return invoke<boolean>("models_exists")
}

/*
下载模型文件
 */
export async function download_models(callback: (progress: number) => void) {
    const onEvent = new Channel<number>();
    onEvent.onmessage = (progress) => {
        callback(progress)
    }
    await invoke("download_models", {
        "onEvent": onEvent
    })
}

/*
加载模型
 */
export async function load_models() {
    await invoke("init")
}

export enum SubtitlesType {
    Online = "Online",
    Offline = "Offline"
}

export type Subtitles = {
    type_: SubtitlesType,
    msg: string
}

/**
 * 实时语音识别
 */
export async function start_speech_recognition(callback: (subtitles: Subtitles) => void) {
    const onEvent = new Channel<Subtitles>();
    onEvent.onmessage = (subtitles) => {
        callback(subtitles)
    }
    await invoke("start_speech_recognition", {
        "onEvent": onEvent
    })
}

/**
 * 停止实时语音识别
 */
export async function stop_speech_recognition() {
    await invoke("stop_speech_recognition")
}