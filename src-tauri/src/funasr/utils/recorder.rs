use crate::funasr::utils::constant::SAMPLE_RATE;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Devices, DevicesError, HostId, InputDevices, SampleFormat};
use num_traits::{Bounded, FromPrimitive, NumCast};
use std::any::TypeId;
use std::sync::{Arc, Mutex};

const MAX_QUEUE_SIZE: usize = 16000 * 300; // 约5分钟的16kHz音频

/// 获取可用的音频主机列表
///
/// # 返回值
/// 返回一个包含所有可用 HostId 的向量
pub fn hosts() -> Vec<HostId> {
    cpal::available_hosts()
}

/// 获取指定主机的输入设备
///
/// # 参数
/// - host: 主机的唯一标识
///
/// # 返回值
/// 返回包含输入设备的 Result，如果获取失败则返回 DevicesError 错误
pub fn devices(host: HostId) -> Result<InputDevices<Devices>, DevicesError> {
    let host = cpal::host_from_id(host).expect("获取主机失败");
    host.input_devices()
}

/// 获取默认主机的默认输入设备
///
/// # 返回值
/// 如果存在默认输入设备，则返回 Some(Device)，否则返回 None
pub fn default_device() -> Option<Device> {
    cpal::default_host().default_input_device()
}

pub struct Recorder {
    pub samples_queue: Arc<Mutex<Vec<f32>>>, // 音频采样队列
    #[allow(dead_code)] // 保证音频流存活，不然不会读取设备输入
    stream: cpal::Stream,                    // 音频流
}


impl Recorder {
    /// 创建一个新的Recorder实例，使用指定采样率
    pub fn new(device: Device) -> Self {
        Self::new_with_max_duration(device, SAMPLE_RATE as u32) // 默认最大5分钟
    }
    /// 创建一个新的Recorder实例，指定最大录制时长
    pub fn new_with_max_duration(
        device: Device,
        sample_rate: u32,
    ) -> Self {
        // 注意不要删除，宏里面要使用的
        #[allow(unused)]
        let err_fn = |err| {
            eprintln!("An error occurred on the output stream: {}", err);
        };

        let config = device
            .default_input_config()
            .expect("Failed to get default input config");
        // 注意不要删除，宏里面要使用的
        #[allow(unused)]
        let source_sample_rate = config.sample_rate().0;
        let samples_queue = Arc::new(Mutex::new(Vec::new()));
        #[allow(unused)]
        let target_sample_rate = sample_rate;

        // 宏来减少重复代码
        macro_rules! build_stream {
            ($sample_type:ty) => {
                device
                    .build_input_stream(
                        &config.into(),
                        {
                            let samples_queue = Arc::clone(&samples_queue);
                            move |data: &[$sample_type], _: &_| {
                                process_samples(
                                    data,
                                    source_sample_rate,
                                    target_sample_rate,
                                    &samples_queue,
                                );
                            }
                        },
                        err_fn,
                        None,
                    )
                    .expect(concat!(
                        "Failed to build input stream for ",
                        stringify!($sample_type)
                    ))
            };
        }

        let stream:cpal::Stream = match config.sample_format() {
            SampleFormat::I8 => build_stream!(i8),
            SampleFormat::I16 => build_stream!(i16),
            SampleFormat::I32 | SampleFormat::I24 => build_stream!(i32),
            SampleFormat::I64 => build_stream!(i64),
            SampleFormat::U8 => build_stream!(u8),
            SampleFormat::U16 => build_stream!(u16),
            SampleFormat::U32 => build_stream!(u32),
            SampleFormat::U64 => build_stream!(u64),
            SampleFormat::F32 => build_stream!(f32),
            SampleFormat::F64 => build_stream!(f64),
            _ => panic!("Unsupported sample format"),
        };
        stream.play().expect("开始录制失败");

        Self {
            samples_queue,
            stream,
        }
    }

    pub fn pop_head_sample(&mut self, chunk_size: usize) -> Option<Vec<f32>> {
        let mut queue = self.samples_queue.lock().expect("获取锁失败");
        if queue.is_empty() || queue.len() < chunk_size {
            return None; // 如果队列为空，返回 None
        }

        // 确保不会超出队列长度
        let chunk_size = chunk_size.min(queue.len());
        let head_samples = queue.drain(0..chunk_size).collect::<Vec<f32>>();

        if head_samples.is_empty() {
            None // 如果没有样本被弹出，返回 None
        } else {
            Some(head_samples) // 返回弹出的样本
        }
    }
}

fn process_samples<T: NumCast + Bounded + FromPrimitive + Copy + 'static>(
    data: &[T],
    source_rate: u32,
    target_rate: u32,
    samples_queue: &Arc<Mutex<Vec<f32>>>,
) {
    // 早期返回空数据
    if data.is_empty() {
        return;
    }

    // 标准化
    let normal_data = normalization(data);

    // 重采样至目标采样率
    let resample_data = resample(normal_data, source_rate, target_rate);

    // 如果重采样后也是空的，直接返回
    if resample_data.is_empty() {
        return;
    }

    // 尽量减少锁持有时间
    {
        let mut queue = samples_queue.lock().expect("获取锁失败");

        // 简单的内存保护：如果队列太大，移除旧的样本
        let new_total = queue.len() + resample_data.len();

        if new_total > MAX_QUEUE_SIZE {
            let excess = new_total - MAX_QUEUE_SIZE;
            queue.drain(0..excess);
        }

        // 预留空间避免多次重新分配
        queue.reserve(resample_data.len());
        queue.extend(resample_data);
    } // 锁在这里释放
}

fn is_float<T: 'static>() -> bool {
    // 尝试为类型 T 实现 Float trait
    TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>()
}

/// 标准化
/// 将传入的音频数据映射到f32 -1~1 的范围
fn normalization<T: NumCast + Bounded + Copy + 'static>(data: &[T]) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    if is_float::<T>() {
        data.iter()
            .map(|&val| {
                T::to_f32(&val).unwrap_or_else(|| {
                    eprintln!("警告: 浮点数  无法转换为 f32，使用 0.0。");
                    0.0
                })
            })
            .collect::<Vec<f32>>()
    } else {
        // 如果是整数，映射到 [-1, 1]
        let min_val_t = T::min_value();
        let max_val_t = T::max_value();

        // 转换为 f64 进行计算以保持精度，最后转换为 f32
        let min_val_f64 = T::to_f64(&min_val_t).expect("无法将最小值转换为 f64");
        let max_val_f64 = T::to_f64(&max_val_t).expect("无法将最大值转换为 f64");

        let range = max_val_f64 - min_val_f64;

        data.iter()
            .map(|&val| {
                let val_f64 = T::to_f64(&val).expect("无法将数据值转换为 f64");
                // 映射到 [0, 1]
                let normalized_0_1 = (val_f64 - min_val_f64) / range;
                // 映射到 [-1, 1]
                let normalized_minus_1_1 = 2.0 * normalized_0_1 - 1.0;
                normalized_minus_1_1 as f32
            })
            .collect::<Vec<f32>>()
    }
}
/// 采用线性插值的方式进行重采样
pub fn resample(data: Vec<f32>, source_sample_rate: u32, target_sample_rate: u32) -> Vec<f32> {
    // 如果源采样率等于目标采样率或者数据为空，则直接返回原始数据
    if source_sample_rate == target_sample_rate || data.is_empty() {
        return data; // 采样率相同，无需重采样
    }

    let source_len = data.len();
    let ratio = source_sample_rate as f64 / target_sample_rate as f64;
    let target_len = ((source_len as f64) / ratio).ceil() as usize;

    // 预分配内存，避免动态扩容
    let mut resampled = Vec::with_capacity(target_len);

    // 如果是整数倍下采样，使用更简单的算法
    if ratio.fract() == 0.0 && ratio >= 1.0 {
        let step = ratio as usize;
        data.iter().step_by(step).for_each(|&x| resampled.push(x));
        return resampled;
    }

    for i in 0..target_len {
        // 计算目标点在原始数据中的对应位置（浮点数）
        let source_index_float =
            (i as f64) * (source_sample_rate as f64) / (target_sample_rate as f64);

        // 获取左右两个最近的原始数据索引
        let left_index = source_index_float.floor() as usize;
        let right_index = source_index_float.ceil() as usize;

        // 如果左索引超出了原始数据范围，直接使用最后一个点的值 (这种情况通常发生在最后一个点或接近最后一个点时)
        if left_index >= source_len - 1 {
            resampled[i] = data[source_len - 1];
            continue;
        }

        // 获取左右两个点的值
        let left_value = data[left_index];
        let right_value = data[right_index];

        // 计算插值系数
        let alpha = (source_index_float - (left_index as f64)) as f32;

        // 线性插值
        let interpolated_value = left_value * (1.0 - alpha) + right_value * alpha;
        resampled[i] = interpolated_value;
    }

    resampled
}
