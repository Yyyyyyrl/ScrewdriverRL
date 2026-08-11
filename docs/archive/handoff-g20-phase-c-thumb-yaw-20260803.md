# 接力 prompt：G20 Phase C 拇指 CMC yaw 标定

> **COMPLETED (2026-08-03) -- do not re-run.** Phase C thumb yaw/roll calibration; physical calibration reached 21/21. The results are consolidated in [docs/hardware/g20-physical-joint-calibration-runbook.md](../hardware/g20-physical-joint-calibration-runbook.md). Kept as the per-session evidence trail.

> **2026-08-03 最终状态：Phase C yaw/roll 均已完成，物理标定达到 21/21。**
> `thumb_cmc_roll` 候选安全读回范围 raw4..248、34结点、`82.5024 deg`；
> 最终停在 raw58，20/20 只读快照 fault 全零。全量 candidate-only 文件为
> `linker_calib_phase_c_thumb_yaw_roll_candidate_20260803.json`。生产映射、
> Isaac limits、semantic limits 与 live deployment 尚未启用。

> **2026-08-03 状态更新：本交接的 yaw 工作已完成。** 正式 candidate-only 结果为 `linker_calib_phase_c_thumb_yaw_candidate_20260803_v2.json`，安全回读范围 raw17..251、33结点、`75.3377 deg`；raw2因80 deg防卷绕门拒绝，raw0未尝试。硬件已回到稳定raw251，fault全零。详见记录索引与 session 内 `THUMB_CMC_YAW_CALIBRATION_REPORT_V2.md`；Phase C 仅剩 `thumb_cmc_roll` slot5。本文件以下内容保留为执行前历史合同，不再代表当前进度。

把这份文件整份贴给接手的模型，工作目录 `/home/user/dex-forge`。

---

你要接手一台 LinkerHand G20 左手（序列号 `LHT20-010-415-L-B-1-D`）的物理关节标定。
唯一流程入口是 [`docs/g20-physical-joint-calibration-runbook.md`](g20-physical-joint-calibration-runbook.md)，
已测结果索引是
[`records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md`](../records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md)。
**先读这两份**，本文件只补充它们没写的现场状态和刚踩过的坑。

## 1. 当前进度

- Phase A（四指 PIP/DIP/pitch + 拇指 MCP/IP/CMC pitch）：`15/15` 完成。
- Phase B（四指 MCP roll）：`4/4` 完成。
- Phase C：`thumb_cmc_yaw`(slot10) **进行中**，`thumb_cmc_roll`(slot5) 未开始。
- 总状态 `19/21`。聚合候选 `linker_calib_phase_b_roll_candidate_20260803.json`，16 个关节中 14 个有实测物理 LUT，315 个结点。

## 2. 硬件此刻的状态（2026-08-03 收工时实测）

```text
state20 = [247, 247, 252, 246, 248, 251, 81, 130, 166, 187, 80, 0,0,0,0, 254, 254, 255, 254, 246]
           s0    s1   s2   s3   s4   s5  s6   s7   s8   s9  s10  reserved  s15  s16  s17  s18  s19
faults 全零
```

- **`thumb_cmc_yaw` slot10 停在回读 80**，静止参考位是 **raw≈251**。第一件事就是把它退回去。
- 拇指其余轴的隔离 hold：`roll(s5)=251`、`pitch(s0)=247`、`mcp(s15)=254`。**每次开工前重新读，不要用常数。**
- 四根长指被一块灰布盖住做遮挡隔离，布不能动。

## 3. 相机与标记（已验证，勿改）

D435 `143322073091`，`1280x720@30`，曝光 166 / 增益 32 / 白平衡 4600，自动曝光与自动白平衡关闭。

**四条蓝标，两组 L 形**：

| 组 | 标记 | 深度 | elongation | 用途 |
|---|---|---|---|---|
| 拇指 L | `thumb`（沿指长）+ `thumb_cross`（环向） | 0.283 / 0.292 m | 11.0 / 6.7 | yaw 主值 + 出平面探测 |
| 掌部 L | `palm`（长横条）+ `palm_cross`（短竖条） | 0.332 / 0.328 m | 14.8 / 5.9 | 固定参考 + 机位漂移探测 |

两组之间 **0.30–0.32 m 是一条几乎全空的深度谷（仅 139 px）**。检测器靠这条谷分层，不靠 ROI。

已验证：拇指 L 夹角 `86.9°`、掌部 L 夹角 `81.8°`；发一步 yaw 后拇指两条各转 `-3.6°`,**掌部只动 `0.017°`**——掌部确实固定在不随 CMC 运动的件上。

## 4. 立刻要做的事

1. 把 slot10 从回读 80 退回 raw≈251。用 `tools.run_g20_thumb_cmc_yaw_candidate_step`，**每步从新鲜读数递推、≤14 raw**（见坑 §5.1）。
2. 在 raw≈251 采**两组** 180 帧零参考（`tools.measure_g20_thumb_yaw`），重复差应 `<0.05°`。
3. 用 `tools.drive_g20_thumb_yaw_sweep` 向下扫到 raw0，再向上回程，再采最终零参考。
4. 写 `tools/summarize_g20_thumb_yaw.py`（照抄 `summarize_g20_mcp_roll.py` 的结构，含**中线 LUT**，见坑 §5.6），产出 summary/points.csv/lut.json/REPORT.md。
5. 合并进新的候选 JSON，更新 runbook checklist 与 record index。
6. 然后才开始 `thumb_cmc_roll`。它大概率要换机位并重新贴标——**不要假设 yaw 的机位和标记能直接用于 roll**。

## 5. 已经踩过的坑（重新发现的代价很高，务必先读）

### 5.1 执行器欠冲 + 死区
每次命令都停在目标短侧，量因关节而异（index roll 2、middle 3、ring 4–5、拇指 yaw 0）。
**小于约 8 raw 的补正步会完全不动**（实测 3 raw 命令零位移、无 fault）。因此：
- 所有点按**稳定回读**编号，不按命令值。
- 下一步的起点必须现读，不能用上一步的 settled 值（pitch 轴退出后还会松回 3 raw）。
- `--settle-tolerance-raw` 按关节显式设定并在报告里写明理由，**不要为了让工具通过而默默放宽**。

### 5.2 无向直线的 ±90° 卷绕
`_normalize_axis_heading` 归一到 `[-90,90)`。拇指-掌部夹角原始值在 +76° 附近，距边界只有 14°，
一扫就翻号。**必须在归一化之前减掉参考偏置**（现在 `--reference-offset-deg 118.0`，让静止位读约 -40°、
行程末端 +40°，两侧各留 50° 余量）。Phase B 掌部竖直胶带也踩过同一个坑。

### 5.3 ROI 裁切能伪装成物理现象 ★
最贵的一次教训：拇指扫出 ROI 后标记被裁，**角度偏了 9.5°,而失败率、块稳定性、palm 漂移全部照常通过**。
当时误判为「转出像面」。防御：
- ROI 必须覆盖整个行程，不是静止姿态。
- 每帧记录标记**面积**和**主轴投影长度**，两者异常收缩就停。
- 真正可靠的出平面判据是 **L 夹角**——它不依赖面积，裁切伪造不了。

### 5.4 指间接触会留下 SDK 看不见的永久位移 ★
四指全伸展时 index 内收 `+7.5°` 顶上中指。相机看到 middle `+0.145°`、ring `-0.278°`，
**而 ring 的回读全程是 116 没变过，事后也不回弹**。位移在编码器下游。结论：
- 隔离要把非目标手指 **pitch 弯开**（raw96 够），只卷 PIP 腾不出空间。
- 不要靠放松力矩隔离——被顶的手指偏得更多且位置不受控。

### 5.5 环境光与姿态变更都会移动"零"
- 光照变化单独就能造成 **0.30°** 的假位移（手完全没动）。房间灯中途熄灭过一次，那一点作废。
- 弯三根手指会把整只手在图像里推移 **42 px**。
- 所以：**每根关节、每个隔离姿态，零参考都要现采**，不能跨姿态复用。掌部标记是唯一能发现这类漂移的东西。

### 5.6 中线 LUT 不只是回差处理 ★
roll/yaw 是双向关节，零位在行程中段、两端是 raw 命令的 0/255 边界而非机械限位。
**单一方向的曲线覆盖不了两端**（内收端只有下行到得了，外展端只有上行到得了）。
§8.3 的中线（上行与下行平均）同时解决回差和覆盖两个问题。
注意 `summarize_g20_mcp_roll.py` 里有个已修的 bug：算回差**必须用未合成的正向曲线**
（`forward_pass`），拿中线去比回程会把回差凭空除以二。

### 5.7 部署映射器的 overlay schema 是白名单
`screwdriver_rl/deploy/linker_sdk_map.py` 的 `_TOP_KEYS` 只允许 `version/note/joints`，
关节内也只允许固定几个键。这是防止手写 overlay 串槽的安全设计，**不要为了塞元数据去放宽它**——
把元数据写进 `note` 字符串。

### 5.8 相机独占
`tools/realsense_capture.py --live` 经常在后台占着 D435，采集前要先结束它。
**永远不要同时开两个 RealSense 进程**，也不要在 CAN 进程运行时开相机（反之亦然）。

## 6. 工具清单

拇指 yaw：
- `tools/measure_g20_thumb_yaw.py` — 深度分层的四标记检测器，输出 yaw + 两个 L 夹角 + 面积/长度诊断
- `tools/run_g20_thumb_cmc_yaw_candidate_step.py` — 单步 fail-closed，0x41 帧元素 1
- `tools/drive_g20_thumb_yaw_sweep.py` — 序列驱动，含卷绕邻近门和 L 收缩门
- `tests/test_g20_thumb_cmc_yaw.py` — 24 个离线测试

Phase B roll（可作模板）：
- `tools/measure_g20_mcp_roll.py` / `run_g20_mcp_roll_candidate_step.py` /
  `drive_g20_mcp_roll_sweep.py` / `setup_g20_roll_isolation.py` / `summarize_g20_mcp_roll.py`

回归：`/home/user/miniconda3/envs/env_isaaclab/bin/python -m pytest tests/ -q -k "thumb or roll or g20 or linker or deploy or calib"`
→ 当前基线 **248 passed, 1 skipped**。

Python 解释器：相机用 `/home/user/miniconda3/bin/python3`，SDK/测试用
`/home/user/miniconda3/envs/env_isaaclab/bin/python`。

## 7. 不可违反的红线

- 发运动前必须过 serial / raw20 / finger-frame / reserved / fault 全部预检；任一不过就写 JSON 记录并停止。
- 单步 ≤17 raw，一次只发一帧，不自动继续、不自动返回、不调 `finger_move`。
- 不覆盖任何已有 JSON、报告或失败记录；`motion_sent=false` 的拒绝记录同样要留。
- 机械端点、新的运动范围由**现场人员批准**，不要自己往极限推。
- 发现异常先停，把证据落盘，再判断——不要为了让门禁通过而调门槛。

## 8. 一段有价值的待验证线索

旧机位、单条胶带时测得的一条 yaw 曲线保留在
`records/.../20260803T_thumb_cmc_yaw_sweep/wide_raw*/`（`回读 251→83`,`0..+54.19°`）。
它有已知的出平面低估。新方案测完后拿同样 raw 位置对比：
**如果新读数系统性偏大，就定量确认了旧方案的低估幅度**，这个结论对判断 2D 方法的适用边界有用，
值得写进报告。
