# Intel RealSense D435 — 使用说明

这台机器上接了一个 D435（序列号 `143322073091`，固件 5.17.0.10），走 USB 3.2，
系统节点是 `/dev/video2`–`/dev/video7`。驱动和工具脚本已经装好，插上就能用。

- 工具脚本：`tools/realsense_capture.py`
- 依赖装在 **conda base 环境**（Python 3.13）：`pyrealsense2` 2.58.3 + `numpy` + `opencv-python`

不需要 sudo，也不需要装 udev 规则 —— 本地登录会话的 ACL 已经放开了设备节点。

## 快速开始

```bash
# 1. 确认摄像头在线，顺便看有哪些分辨率/帧率可选
python tools/realsense_capture.py --info

# 2. 抓一帧（彩色 + 深度）存到 outputs/
python tools/realsense_capture.py --snapshot outputs/shot

# 3. 实时预览（左彩色，右深度伪彩）
python tools/realsense_capture.py --live
```

`--live` 窗口里按 `q` 或 `ESC` 退出，按 `s` 存一张彩色图到 `outputs/`。
注意它需要图形界面；纯 SSH 登录的时候用 `--snapshot`。

## 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--info` | — | 打印设备信息和全部可用的 stream profile |
| `--snapshot PREFIX` | — | 抓一帧存成 `PREFIX_color.png` / `PREFIX_depth.png` / `PREFIX_depth_raw.npy` |
| `--live` | — | 打开实时预览窗口 |
| `--width` / `--height` | 640 / 480 | 分辨率，可选值见 `--info` |
| `--fps` | 30 | 帧率 |
| `--no-align` | 关 | 关闭深度到彩色的对齐 |

分辨率和帧率必须是设备支持的组合，随便填会在 `pipeline.start()` 报错。常用的几个：
彩色 `640x480@30`、`1280x720@30`、`848x480@60`；深度 `640x480@30`、`848x480@90`。

## 输出文件

`--snapshot outputs/shot` 会生成三个文件：

- `shot_color.png` — BGR 彩色图，`(H, W, 3)` uint8
- `shot_depth.png` — 深度伪彩图，**只是给人看的**，不要拿去算距离
- `shot_depth_raw.npy` — 原始深度，`(H, W)` uint16，单位是 *深度单位* 而非米

真实距离要乘 depth scale（这台机器是 `0.001 m/unit`，即 1 单位 = 1 毫米）：

```python
import numpy as np
depth = np.load("outputs/shot_depth_raw.npy")
meters = depth * 0.001
print(meters[240, 320])   # 画面中心离相机多远
```

`0` 表示该像素没测出深度（太近、太远、反光、遮挡）。算统计量之前先滤掉：

```python
valid = meters[meters > 0]
```

## 在自己的代码里读画面

脚本里的函数可以直接复用：

```python
from tools.realsense_capture import start_pipeline, grab

pipeline, align, depth_scale = start_pipeline(640, 480, 30, align_to_color=True)
try:
    for _ in range(100):
        color, depth = grab(pipeline, align)   # color: HxWx3 uint8, depth: HxW uint16
        meters = depth * depth_scale
        # ... 你的处理
finally:
    pipeline.stop()
```

或者直接用裸 SDK：

```python
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

try:
    frames = align.process(pipeline.wait_for_frames())
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data())
finally:
    pipeline.stop()
```

两个容易踩的点：

1. **开流后前几十帧别用。** 自动曝光要时间收敛，一上来的图偏暗偏绿。脚本里默认丢掉前
   30 帧，自己写的话也照做。
2. **深度和彩色默认不对齐。** 两个镜头物理位置不同，同一个 `(u, v)` 在两张图上不是同一个点。
   要按彩色图的像素坐标查深度，必须过一遍 `rs.align(rs.stream.color)`。

## 像素坐标 → 三维点

有内参就能把某个像素反投影成相机坐标系下的 3D 点：

```python
depth_frame = frames.get_depth_frame()
intr = depth_frame.profile.as_video_stream_profile().intrinsics
d = depth_frame.get_distance(u, v)                    # 米，已经乘过 scale
point = rs.rs2_deproject_pixel_to_point(intr, [u, v], d)   # [x, y, z] 米
```

整张图转点云用 `rs.pointcloud()` 更快，别写 Python 循环。

## 精度和量程

- 官方标称有效量程约 **0.3–3 m**，最远能到 10 m 但误差迅速变大。
- 深度误差大致随距离平方增长，1 m 处约 ±2 mm，3 m 处到厘米级。
- 实测远处会出现几十米的离群噪点，需要的话按距离裁一刀，或者加个滤波：

```python
thr = rs.threshold_filter(min_dist=0.2, max_dist=3.0)
depth_frame = thr.process(depth_frame)
```

- 白墙、玻璃、强反光表面出不了深度（红外散斑无纹理可匹配），这是原理限制，不是坏了。

## 在别的 conda 环境里用

依赖目前只装在 base。要在 `dexmachina` / `env_isaaclab` / `hora` 里用：

```bash
conda activate dexmachina
pip install pyrealsense2 opencv-python
```

这几个环境是 numpy 1.x，pyrealsense2 兼容，不会冲突。

## 排查

**`no RealSense device found`**

```bash
lsusb | grep 8086          # 应该看到 8086:0b07 RealSense D435
ls /dev/video*             # 应该有 6 个属于 D435 的节点
```

看不到设备就换根线换个口 —— D435 对线材挑剔，必须是 USB 3.0 数据线，用充电线会枚举不出来
或者只能跑低分辨率。

**`--info` 里 usb 显示 2.1 而不是 3.2**

插到 USB 2 口上了，或者线不行。USB 2 下高分辨率高帧率的组合会直接失败。

**`RuntimeError: Frame didn't arrive within 5000`**

带宽不够或者被别的进程占着。确认没有第二个程序（比如 `realsense-viewer`）在同时开流，
一个设备同一时间只能被一个进程独占。

**图像偏暗/偏绿** — 丢帧数不够，把预热的 30 帧加大。
