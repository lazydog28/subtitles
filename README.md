# 极简实时字幕

> 本地离线实时语音识别字幕软件
>
> 前端使用 [tauri](https://tauri.app/) 框架开发
>
> 语音识别模型是基于[FUNASR]("https://github.com/modelscope/FunASR")，但是在实际音频数据预处理方面是我基于原版代码cpp代码简化实现，导致识别效果应该没有原版理想。
>
> 在有能力的情况下，可以使用 [FUNASR]("https://github.com/modelscope/FunASR") 自行二次开发。
> 
> 有其他功能需求请发 issues

![60f7287930f06b9923e33fde7931f3b6.png](https://imagesbed28.caiyun.fun/60f7287930f06b9923e33fde7931f3b6.png)

## 使用教程

### 下载

> 当前仅编译了 windows x64 平台安装包，其他桌面平台请自行下载源码编译。

1. [夸克网盘](https://pan.quark.cn/s/d62edd9e8545)
2. [GITHUB](https://github.com/lazydog28/subtitles/releases)

### 安装

下载完成后双击安装包后开始安装，自行选择安装位置。

### 启动
> 第一次启动时如果没有任何反应，请尝试以管理员权限启动一次以后应该就能正常启动了
> 
软件启动后会进行检查本地模型文件，如果不存在则会下载模型文件，模型文件压缩包大小为 `469M` 左右，仅需下载一次模型，模型将保存在软件路径下的`models`文件夹。

等待软件加载模型，加载完成后桌面文字提示`等待识别...`，即可开始识别语音。

软件在任务栏没有图标，退出、切换输入设备、切换语音在右下角托盘内有软件图标，右键图标进行使用。

![739cc69f5332f1cb862c1e8790756a9a.png](https://imagesbed28.caiyun.fun/739cc69f5332f1cb862c1e8790756a9a.png)

![b5d106f604f27cbdb35ff492b088012e.png](https://imagesbed28.caiyun.fun/b5d106f604f27cbdb35ff492b088012e.png)