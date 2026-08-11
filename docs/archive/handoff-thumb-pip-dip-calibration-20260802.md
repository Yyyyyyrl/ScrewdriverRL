# 新 Session 接力 Prompt：G20 拇指 PIP/DIP 物理标定

> **COMPLETED (2026-08-02) -- do not re-run.** Thumb PIP/DIP physical calibration. Results are consolidated in [docs/hardware/g20-physical-joint-calibration-runbook.md](../hardware/g20-physical-joint-calibration-runbook.md).

状态：**已于2026-08-02完成并归档，不要再次执行。**  
完成结果见：
`records/g20_physical_joint_calibration_20260801/sessions/20260802T202449Z_thumb_mcp_ip_pitch_thin_tape_full_raw_sweep/THUMB_MCP_IP_PITCH_FIXED_EXPOSURE_CALIBRATION_REPORT.md`

把下面整段复制到新的 Codex session。工作目录必须是 `/home/user/dex-forge`。

---

你正在接力一项已经进行到实机物理标定阶段的 LinkerHand G20 工作。不要从头重做，也不要先跑 policy。请先读现有结果和 SOP，然后直接继续完成拇指 PIP/DIP 等价关节的物理标定。

## 当前唯一目标

完成并留证：

- `thumb_mcp`：SDK slot15，主动屈曲关节；在本标定计划中承担“thumb PIP”等价角色。
- `thumb_ip`：被动机械 mimic，无独立 SDK slot；承担“thumb DIP”等价角色。

本轮不要把 `thumb_cmc_pitch`、`thumb_cmc_yaw` 或 `thumb_cmc_roll` 标成已完成。yaw/roll 在本轮只允许用于把拇指侧面转向相机，正式 yaw/roll 标定仍属于最后的 Phase C。

## 必须先读

1. `/home/user/dex-forge/docs/g20-physical-joint-calibration-runbook.md`
2. `/home/user/dex-forge/records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md`
3. `/home/user/dex-forge/docs/realsense-d435.md`
4. 已完成的食指 PIP/DIP 报告：
   `/home/user/dex-forge/records/g20_physical_joint_calibration_20260801/sessions/20260802T060815Z_index_pip_dip_thin_tape_full_raw_sweep/INDEX_PIP_DIP_FIXED_EXPOSURE_VALIDATION_REPORT.md`
5. 当前映射：
   `/home/user/dex-forge/screwdriver_rl/deploy/linker_sdk_map.py`
6. 当前候选 overlay：
   `/home/user/dex-forge/linker_calib_index_pip_pitch_lut_candidate_20260802.json`
7. 旧的拇指姿态修正 overlay，仅作历史/摆姿参考：
   `/home/user/dex-forge/linker_calib_thumbfit.json`

必须保护 dirty worktree和全部现有记录，不覆盖旧session，不删除失败、拒绝发送或低质量相机点。

## 当前现场状态

- 实机：左手 G20。
- hand serial：`LHT20-010-415-L-B-1-D`。
- CAN：`can0`。
- SDK：`/home/user/linkerhand-ros-sdk`，G20 left。
- 当前为空手，不带螺丝刀。
- 用户报告：其他四根手指已经用黑布遮住。
- 拇指侧面已经贴蓝色胶带。
- D435仍通过USB连接，固定曝光基线：
  - serial `143322073091`
  - `1280x720@30`
  - exposure `166`
  - gain `32`
  - white balance `4600`
  - auto exposure/white balance off
- 用户提供的摆姿参考：`thumb_cmc_roll = 1.22 rad`，`thumb_cmc_yaw = 0.86 rad`，目的是让贴蓝胶带的拇指侧面正对相机。

重要：`1.22/0.86 rad` 是软件姿态参考，不是已经验证的物理角或SDK机械端点。当前 `linker_calib_thumbfit.json` 中 yaw/roll 的 mapping hi 是 `1.6673/1.4788`，所以不得声称 roll `1.22 rad` 就等于 raw0或机械最大。正式移动前必须明确选用哪个overlay，计算并打印 semantic rad→slot raw，保存计算输入、overlay路径和digest。

## 立即执行顺序

### 1. 只读预检，暂不运动

- 检查相机在线、USB 3.2、没有第二个RealSense进程。
- 创建新session：
  `records/g20_physical_joint_calibration_20260801/sessions/<UTC>_thumb_mcp_ip_thin_tape_full_raw_sweep/`
- 保存固定曝光 RGB、aligned depth、raw depth、camera info。
- 保存20帧SDK raw20/state/fault/temperature快照。
- 核对serial、side、G20、reserved slots11..14、fresh thumb frame与raw20槽位。
- 明确记录当前 slot0/5/10/15 的命令与稳定回读。
- 先把当前相机图给用户检查；只有蓝标身份和曝光可用才继续。

### 2. 建立拇指专用的fail-closed工具

现有食指单步工具只能作设计参考，不能直接改参数运行：

- `tools/run_g20_index_pip_candidate_step.py`
- `tests/test_run_g20_index_pip_first_step.py`

新增并测试至少两个专用工具：

1. `tools/run_g20_thumb_view_pose_step.py`
   - 只允许slot5 roll或slot10 yaw一次改变一个轴。
   - slot0 pitch和slot15 MCP保持实际起始值。
   - 不同时改变yaw和roll。
   - 必须做expected-start、fresh thumb frame、fault、temperature、reserved slot和非目标槽门禁。
   - 低速、低扭矩、相邻小步。

2. `tools/run_g20_thumb_mcp_candidate_step.py`
   - 只允许slot15改变。
   - slot0/5/10全部保持已经验证的相机摆姿。
   - 不向`thumb_ip`发送命令，因为它没有独立slot。
   - 每步发送前和发送后都写JSON。

先写离线单元测试，证明错误slot、过大步长、起点不匹配、fault、reserved slot异常和非目标槽变化都会拒绝发送。测试通过后才能带 `--execute`。

### 3. 把拇指侧面摆正

- 先检查当前姿态，也许已经无需移动。
- 如需移动，先yaw、后roll，每次只动一个轴。
- 以用户给的 yaw `0.86 rad`、roll `1.22 rad` 为目标参考，但先通过明确选定的overlay换算为raw。
- 不允许一次性大跨度；以不超过约8–12 raw的相邻步接近。
- 每步保存状态、fault、temperature；任何异常立即停止。
- 每个轴到位后关闭CAN进程，再采固定曝光画面。
- 让用户人眼确认拇指贴胶带侧面确实正对相机；确认后把slot5/10作为本session的“view hold raw”，但不要写成yaw/roll物理标定结果。

### 4. Camera-only识别验证

- 不照抄食指ROI、深度门限或angle priors。
- 为拇指建立自己的ROI、深度分布和标记身份。
- 需要从掌部/拇指proximal/拇指distal标记恢复：
  - `thumb_mcp`物理屈曲角；
  - `thumb_ip`相对`thumb_mcp`的被动mimic角。
- 胶带安装偏角必须通过伸展零参考相减消除。
- 先在手不动时采两组180帧；中位数差不超过`0.5°`。
- 保存逐帧CSV、RGB、aligned depth、annotated PNG、失败率和60帧块稳定性。
- 如果黑布侵入HSV候选不应有问题，但仍需用ROI+深度+几何身份证明蓝标属于拇指；身份不确定时停止。

### 5. `thumb_mcp`/`thumb_ip`正式 sweep

- slot15必须单独测实机范围，不能采用当前Isaac `1.05 rad`或旧overlay `1.29 rad`作为答案。
- raw255是否是物理零位、raw0是否安全都必须实测，不得默认。
- 先覆盖policy需要的工作范围，再探索full physical range。
- 参考点序列可以从：
  `255, 240, 224, 208, ...`
  开始，但每个新端点区域都必须独立放慢。
- 实际相邻变化不超过17 raw；靠近未验证端点改用更小步长。
- 每个点：
  1. 发送前preflight；
  2. 只发送一个slot15目标；
  3. 至少记录20组state/fault/temperature；
  4. 确认slot0/5/10保持在view hold附近；
  5. 关闭CAN；
  6. 相机采180帧；
  7. 汇总target raw、stable readback raw、MCP/IP物理角与质量；
  8. 门禁通过才继续。
- 用户此前说明硬件关节有约1–2 raw松动，可记录为正常量化/松旷，但不能放宽故障和明显串扰门禁。
- 完成正向后必须相邻步反向回程，至少测中点回差和最终零漂。
- 任一fault、异常声音、碰撞、卡滞、明显温升、角度反向、相机身份错误立即停止。

### 6. 结果和软件产物

必须生成：

- session manifest；
- 每步原始运动JSON；
- SDK快照、fault、temperature；
- 每个正式相机点的逐帧CSV/RGB/depth/annotated；
- calibration points CSV；
- 机器可读summary；
- `REPORT.md`；
- `thumb_mcp`单调 physical LUT；
- `thumb_ip = f(thumb_mcp)` 的mimic ratio/offset或非线性关系；
- held-out误差、正反回差、零漂、物理full range、policy valid range；
- 只含已有index物理LUT加新thumb MCP物理LUT的 `CANDIDATE ONLY` overlay。

不得覆盖：

- `linker_calib_thumbfit.json`
- `linker_calib_index_pip_pitch_lut_candidate_20260802.json`
- 任何已有report/session。

如果测得的thumb IP mimic与当前Isaac multiplier `1.1619`不同，先在报告中提出候选，不要在其余拇指关节未完成前静默修改生产配置。

### 7. 完成后的状态更新

只有上述证据全部齐全且安全回位后：

- 在 `docs/g20-physical-joint-calibration-runbook.md` 将：
  - `thumb_mcp`标为`[x]`
  - `thumb_ip`标为`[x]`
- Phase A由`3/15`更新为`5/15`。
- 总状态由`3/21`更新为`5/21`。
- `thumb_cmc_pitch`仍保持未完成。
- Phase B/C全部保持未完成。
- 更新：
  `/home/user/dex-forge/records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md`

完成后向用户报告完整结果，不要每个正常中间点都重复询问。只有初始画面/摆姿需要人眼确认，或出现异常、需要批准未知机械端点时才停下来询问。

不要启动policy，不要装螺丝刀，不要切换相机视角。本轮结束目标仅是可靠完成thumb MCP/IP，即本计划中的拇指PIP/DIP。

---
