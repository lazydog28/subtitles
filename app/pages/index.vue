<template>
  <div id="title-bar" class="fixed right-1 top-1 flex space-x-1">
    <div class="title-bar-button" id="title-bar-minimize" @click="minimize">
      <img
          src="https://api.iconify.design/mdi:window-minimize.svg"
          alt="minimize"
          class="title-bar-icon"
      />
    </div>
    <div class="title-bar-button" id="title-bar-drag" @mousedown="drag">
      <img
          src="https://api.iconify.design/material-symbols:drag-pan-rounded.svg"
          alt="drag"
          class="title-bar-icon"
      />
    </div>
  </div>
  <div id="main" class="w-screen h-screen flex justify-center items-center bg-opacity-50">
    <div id="msg" class="btn" ref="msgElement">{{ msg }}</div>
  </div>
</template>

<script setup lang="ts">
import {getCurrentWindow} from '@tauri-apps/api/window';
import {message, models_exists, download_models, load_models, start_speech_recognition,SubtitlesType} from "~/utils"

const webWindow = getCurrentWindow();

async function minimize() {
  await webWindow.minimize();
}

function drag(e: MouseEvent) {
  if (e.buttons === 1) {
    webWindow.startDragging();
  }
}

let msg = ref("状态初始化...")

const msgElement = ref<HTMLElement | null>(null);

// 调整字体大小以防止溢出
function adjustFontSize() {
  if (!msgElement.value) return;

  const element = msgElement.value;
  const maxWidth = window.innerWidth * 0.9; // 最大宽度为屏幕宽度的90%
  const maxHeight = window.innerHeight * 0.9; // 最大高度为屏幕高度的90%

  // 从初始字体大小开始
  let fontSize = Math.min(window.innerHeight * 0.7, window.innerWidth * 0.1); // 初始字体大小
  element.style.fontSize = `${fontSize}px`;
  element.style.lineHeight = `${fontSize}px`;

  // 逐渐减小字体大小直到不溢出
  while ((element.scrollWidth > maxWidth || element.scrollHeight > maxHeight) && fontSize > 12) {
    fontSize -= 2;
    element.style.fontSize = `${fontSize}px`;
    element.style.lineHeight = `${fontSize}px`;
  }
}

// 监听 msg 变化并调整字体大小
watch(msg, () => {
  nextTick(() => {
    adjustFontSize();
  });
});

const lastSubtitlesType = ref<SubtitlesType>(SubtitlesType.Offline)

onMounted(async () => {
  adjustFontSize();
  // 当窗口尺寸改变时，重新调整字体大小
  window.addEventListener('resize', adjustFontSize);
  console.log(new Date())
  let exist = await models_exists();
  if (!exist) {
    await download_models((progress) => {
      console.log("接收到进度", progress)
      msg.value = `正在下载模型：${Math.round(progress * 100)}%`
    })
  }
  msg.value = "加载模型中..."
  await load_models();
  msg.value = "等待识别..."
  await start_speech_recognition(
      (subtitles) => {
        console.log("接收到字幕消息：",subtitles,"lastSubtitlesType:",lastSubtitlesType.value)
        if (subtitles.type_ == SubtitlesType.Online && lastSubtitlesType.value==SubtitlesType.Online) {
          msg.value = msg.value + subtitles.msg
          lastSubtitlesType.value=SubtitlesType.Online
        }else if (subtitles.type_ == SubtitlesType.Online){
          msg.value = subtitles.msg
          lastSubtitlesType.value=SubtitlesType.Online
        }
        else {
          let splitMsg=subtitles.msg.split(">")
          msg.value = splitMsg[splitMsg.length-1] as string
          lastSubtitlesType.value=SubtitlesType.Offline
        }
      }
  )


})


onUnmounted(() => {
  window.removeEventListener('resize', adjustFontSize);
});


</script>
<style scoped>
@reference "~/assets/app.css";

#main {
  background: rgba(85, 85, 85, 0.5);
}

.title-bar-button {
  @apply p-0.5 rounded-xl
}

.title-bar-button:hover {
  @apply bg-gray-500 ;
}

#msg {
  white-space: nowrap; /* 防止文字换行 */
  overflow: hidden; /* 隐藏溢出内容 */
  max-width: 90vw; /* 设置最大宽度 */
  max-height: 90vh; /* 设置最大高度 */
  display: flex;
  align-items: center;
  justify-content: center;

}


</style>