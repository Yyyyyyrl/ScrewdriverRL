# LinkerHand G20 全手物理关节标定运行手册

版本：2026-08-02 v1  
对象：左手 G20，序列号 `LHT20-010-415-L-B-1-D`  
目的：把 Isaac rad、SDK raw0..255 和相机测得的真实物理角建立为可验证、可追溯的合同。

本手册是后续标定的唯一流程入口。已测结果索引见：

[`../records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md`](../records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md)

## 1. 总体顺序

严格按以下顺序执行，不交叉：

1. 用侧视相机完成所有手指的 PIP/DIP/pitch，包括拇指对应关节。
2. 第一阶段全部完成后，才切换到 roll 视角，标定 index/middle/ring/pinky 的 MCP roll。
3. 四根长指 MCP roll 全部完成后，最后标定拇指 CMC yaw 和 CMC roll。

相机机位合同（适用于全部三个阶段）：

- 硬约束只有一条：**同一关节的零参考与它的全部采点必须在同一机位下完成**，中途相机、手掌和胶带都不能动。
- 允许在关节与关节之间换位，包括阶段内部。换位不需要保留可还原的外参或支架几何。
- 换位后，受影响关节的零参考、ROI、深度门限和 angle priors 全部作废，必须重建；跨机位的绝对角不得混用，只有各自机位内的差值角可比。
- 换位事实必须写进该关节的 session 记录，说明哪些曲线属于哪个机位。Phase A 中指就是这样处理的：PIP 用旧机位双零点，pitch 与 PIP 滞回用新机位双零点。

阶段一中，拇指没有解剖学上的 PIP/DIP，使用机构等价项：

- “thumb PIP”角色：`thumb_mcp`，SDK slot15，主动屈曲关节。
- “thumb DIP”角色：`thumb_ip`，被动 mimic，无独立 SDK slot。
- “thumb pitch”角色：`thumb_cmc_pitch`，SDK slot0。

## 2. 全关节 checklist

完成标记规则：

- `[x]` 只表示已经完成真实运动、相机外部测量、正/反向回差、SDK故障/温度记录、LUT/报告和安全回位。
- 只做槽位确认、只读快照、手工目测或复用别的手指数据，均不能标 `[x]`。
- 不再要求逐项勾选清单。判据是该关节 session 内的原始证据齐全且未被覆盖，任何结论随时可以从原始记录重新验证；证据留存规则见 §4 和 §6.1。

### Phase A — 所有手指 PIP/DIP/pitch，侧视相机

#### Index

- [x] `index_pip`，slot16，主动关节。
- [x] `index_dip` mimic，无独立slot；已测随 PIP 的物理关系。
- [x] `index_mcp_pitch`，slot1，主动关节。

#### Middle

- [x] `middle_pip`，slot17。
- [x] `middle_dip` mimic。
- [x] `middle_mcp_pitch`，slot2。

#### Ring

- [x] `ring_pip`，slot18。
- [x] `ring_dip` mimic。
- [x] `ring_mcp_pitch`，slot3。

#### Pinky

- [x] `pinky_pip`，slot19。
- [x] `pinky_dip` mimic。
- [x] `pinky_mcp_pitch`，slot4。

#### Thumb

- [x] `thumb_mcp`，slot15；在本阶段承担 thumb PIP 等价角色。
- [x] `thumb_ip` mimic；承担 thumb DIP 等价角色。
- [x] `thumb_cmc_pitch`，slot0。

Phase A 状态：`15/15` 完成。

Pinky完成说明：

- PIP实测安全物理端点为`1.7332971062 rad`，被动DIP为`1.3204194204 rad`。
- MCP pitch最低保持隔离的命令是raw6，对应`1.2128883287 rad`；命令raw0时side轴从保持目标raw153掉到raw140，因此raw0被拒绝且不进入LUT。
- 最终零参考漂移事后补算：PIP `+0.135°`、DIP `-0.115°` 在门内；`pinky_mcp_pitch` `-1.023°` 超过§9的`0.5°`门。已归因为raw255伸展止点的机械复位不重复性（掌部标记只动`0.079°`，相机与夹具稳定），接受为零锚点偏置而非重测，该端点须带`±0.0178546 rad`不确定度使用。见 `PINKY_FINAL_ZERO_DRIFT_ADDENDUM.md`。
- 四根长指PIP候选`hi`均以各自实测安全物理端点为准。Isaac limits和部署门禁中的旧`1.08 rad`假设延期到Phase A之后统一修改，不能反向覆盖实测结果。
- Phase A聚合候选：`linker_calib_index_thumb_middle_ring_pinky_mcp_pitch_lut_candidate_20260802.json`。

### Phase B — 切换相机后，四根长指 MCP roll

- [x] `index_mcp_roll`，slot6。
- [x] `middle_mcp_roll`，slot7。
- [x] `ring_mcp_roll`，slot8。
- [x] `pinky_mcp_roll`，slot9。

Phase B 状态：`4/4` 完成。

Phase B 完成说明：

- roll 是双向关节，零位是行程**中段的参考位**而非机械止点，两端都是 raw 命令的 0/255 边界而不是机械限位。§8.1「低速回到预计伸展端」和 §10「raw255 对应伸展零位」都不适用。
- 四指实测全行程 `25.5..28.3°`，SDK 表的 `±0.17 rad`（19.5°）低估约 30–45%，且机械行程比 raw 范围还宽。
- LUT 取 §8.3 的**中线**（上行与下行的平均）。这不只是回差处理，也是让 LUT 覆盖两端的唯一办法——内收端只有下行到得了，外展端只有上行到得了。
- 执行器每次命令都欠冲，且欠冲量因指而异（index 2、middle 3、ring 4–5、pinky 3）。低于约 8 raw 的补正步会落进死区完全不动，所以所有点按**稳定回读**而非命令编号。
- 隔离方式：非目标手指的 MCP pitch 弯到 raw96，只要求目标指的 proximal 蓝标可见。代价是失去了非目标标记这个接触探测器（见下）。
- Phase B 聚合候选：`linker_calib_phase_b_roll_candidate_20260803.json`。

### Phase C — 最后标定拇指 yaw/roll

- [x] `thumb_cmc_yaw`，slot10。
- [x] `thumb_cmc_roll`，slot5。

Phase C 状态：`2/2` 完成。

Phase C yaw 完成说明：

- `thumb_cmc_yaw` 使用 slot10，稳定回读 raw251 为相对零参考；安全接收范围为 raw17..251，物理范围 `75.3377 deg / 1.3148913 rad`。raw2 重复测得 `+80.0168 deg`，触发无向直线 80 deg 防卷绕门，被拒绝；raw0 未尝试。
- 33 结点 LUT 取正反向中线，回读域最大回差 `0.8263 deg`，held-out 分段线性最大误差 `0.0078569 rad`。初始双零重复差 `0.010841 deg`，最终零漂 `+0.0478 deg`。
- 掌部参考使用刚性 L 双标：两条相对漂移在 0.5 deg 内时平均；端点处掌纵标被拇指遮挡、两者分歧时自动回退到未遮挡的掌横标。所有接收相机点 180/180，53 份已发送运动全 fault-free，最高温度 56 C。
- yaw-only candidate 聚合：`linker_calib_phase_c_thumb_yaw_candidate_20260803_v2.json`，当时16个关节中15个有实测物理LUT，共348个结点；该历史候选现已被下述yaw+roll候选取代。

Phase C roll 完成说明：

- `thumb_cmc_roll` 使用 slot5；yaw/pitch/MCP 分别固定 raw125/247/254。稳定回读 raw58 是相机相对测量锚点，不是语义零位；安全接收范围 raw4..248，raw248 对应非负 URDF 语义 `0 rad`，raw4 对应 `82.5024 deg / 1.4399387 rad`。raw0/255 未声称可达：端点剩余距离已低于实测 8 raw 死区。
- 主值为两枚拇指蓝标之间 3D 向量的方位角，新白色水平掌标只作刚性漂移见证。旧水平蓝标松动造成零重复差约 `0.457 deg`，其全部数据保留为 rejected evidence，不进入 LUT。
- 34 结点 LUT 取稳定回读域正反向中线；最大回差 `0.9648 deg`，held-out 分段线性最大误差 `0.0047725 rad / 0.2734 deg`。初始 raw58 锚点重复差 `0.045008 deg`，最终锚点重复差 `0.005196 deg`，完整往返锚点漂移 `+0.4430 deg`。
- settle 容差显式为4 raw：两次 fault-free 且 hold 稳定的欠冲分别达到3和4 raw。最终20次只读快照均为raw58、全 fault-free。
- candidate-only 聚合：`linker_calib_phase_c_thumb_yaw_roll_candidate_20260803.json`，16 个主动关节均有实测物理 LUT，共 382 个结点。生产映射、Isaac limits、semantic limits 与 live deployment 均未启用。

总状态：`21/21` 关节或 mimic 合同完成。

## 3. 人与软件的责任

现场人员必须：

- 固定手掌，清空运动空间，保持断电/急停可达。
- 确认当前为空手，不带螺丝刀或其他负载。
- 观察碰撞、异常声音、线缆拉扯、胶带松动和明显发热。
- 负责贴标、遮挡非目标手指以及批准新的机械端点。
- 相机换位前确认当前关节的全部采点和最终零参考复核都已完成；跨阶段换位还要确认上一阶段已全部完成。

软件执行者可以：

- 只读核对 serial、SDK、raw20、fault、temperature、camera。
- 生成并测试单关节 fail-closed 控制工具。
- 每次只发送一个已批准相邻 raw 目标。
- 保存全部 SDK、相机和拟合记录。
- 发现任何异常后停止，不自动跨过异常点。

## 4. 不可违反的原则

- 每根手指、每个主动关节单独测；不复制食指 LUT。
- 每个机械端点单独验证；食指 PIP raw0 fault64 不代表别的关节也fault，食指 pitch raw0安全也不代表别的关节安全。
- DIP/IP 是被动 mimic，只从相机测物理关系，不向其发送独立 raw。
- 不用当前 Isaac/SDK表的 `1.08`、`1.4` 或 `1.57` 当作物理先验答案。
- 不用 mapper 正算再反算的零误差证明物理正确。
- 不同时运行两个 RealSense 进程；D435只能被一个进程占用。
- 不在校准流程末尾自动启动 policy。
- 不覆盖旧 JSON、URDF、报告或失败记录。

## 5. 固定设备与环境

```bash
cd /home/user/dex-forge
```

SDK/CAN：

```text
SDK root: /home/user/linkerhand-ros-sdk
side: left
hand_joint: G20
CAN: can0
serial: LHT20-010-415-L-B-1-D
SDK version: 3.1.0
embedded: 1.0.7
```

相机：

```text
serial: 143322073091
profile: 1280x720@30
exposure: 166
gain: 32
white balance: 4600
auto exposure: off
auto white balance: off
depth scale: about 0.001 m/raw
```

Python：

- 相机：`/home/user/miniconda3/bin/python3`
- SDK/项目测试：`/home/user/miniconda3/envs/env_isaaclab/bin/python`

## 6. 每个新关节开始前

### 6.1 建立不可覆盖 session

目录命名：

```text
records/g20_physical_joint_calibration_20260801/sessions/
  <UTC timestamp>_<joint>_<marker/view>_<sweep>/
```

建议结构：

```text
initial_readonly/
camera_detector_validation/
raw255_zero/
flex_rawXXX/
return_to_raw255/
final_readonly/
session_manifest.json
calibration_points.csv
summary.json
REPORT.md
```

任何失败点和 `motion_sent=false` 记录也必须保留。

### 6.2 贴标与视觉身份

长指 PIP/DIP/pitch：

- 掌部或对应 metacarpal：一条蓝标。
- proximal：一条蓝标。
- middle：一条蓝标。
- distal：一条蓝标。
- 胶带可以不是完美细长矩形，但必须牢固、不能跨关节。

如果其他手指也贴同色胶带：

- 优先用ROI、深度门限和掌骨端点几何隔离目标指。
- 仍有身份歧义时，遮挡非目标手指。
- 不允许在身份不确定时“选最像的一个”继续。

拇指：

- Phase A 的 MCP/IP/CMC pitch 至少需要掌部、拇指 proximal、distal 标记。
- Phase C 的 yaw/roll 需要能恢复平面方向的 L 形双标记或刚性方形标记；单条胶带不足以完成三维姿态合同。

### 6.3 相机只读快照

```bash
/home/user/miniconda3/bin/python3 -m tools.realsense_capture \
  --snapshot SESSION/initial_readonly/fixed_rgb_snapshot \
  --width 1280 --height 720 --fps 30 \
  --rgb-exposure 166 --rgb-gain 32 --rgb-white-balance 4600 \
  --control-warmup-frames 45
```

现场人员先检查：

- 不过曝。
- 蓝胶带边界可见。
- 黑布/背景没有被误分成标记。
- 目标关节完整位于预计运动ROI。

### 6.4 SDK只读预检

```bash
/home/user/miniconda3/envs/env_isaaclab/bin/python \
  -m tools.record_g20_sdk_snapshot \
  --sdk-root /home/user/linkerhand-ros-sdk \
  --side left --hand-joint G20 --can can0 \
  --calib CALIBRATION_CANDIDATE.json \
  --samples 20 --hz 20 \
  --out SESSION/initial_readonly/sdk_snapshot.json
```

必须确认：

- serial完全匹配。
- raw20长度20，reserved slots11..14全0。
- 所有 finger fault全0。
- 目标slot稳定。
- 新鲜finger六元素frame与raw20可见槽一致。

### 6.5 先写工具、后动手

每个新主动关节必须有专用单步工具和离线测试。工具必须：

- 固定 serial、side、G20、CAN frame。
- 只允许目标slot和明确的隔离hold slot改变。
- 校验 expected-start 与实际raw20。
- 校验 fresh finger frame 和raw20的side/root/tip映射。
- 校验reserved槽和全部fault。
- 只设置目标finger的speed/torque。
- 只发送一次position frame。
- 不自动继续，不自动返回，不调用全手 `finger_move`。
- 发送前先写JSON，发送后持续记录state/fault/temperature。

已验证参考实现：

- `tools/run_g20_index_pip_candidate_step.py`
- `tools/run_g20_index_mcp_pitch_candidate_step.py`
- `tests/test_run_g20_index_pip_first_step.py`
- `tests/test_run_g20_index_mcp_pitch_candidate_step.py`

其他手指不能直接改命令行参数复用食指脚本；先写对应slot/finger-frame版本并测试。

## 7. Phase A 的相机测角

### 7.1 PIP/DIP

PIP：

```text
physical PIP = extension-reference tape angle - current tape angle
```

DIP：

```text
physical DIP = extension-reference DIP tape angle - current DIP tape angle
```

符号应以屈曲增加为正。若相机几何导致公式符号相反，在汇总脚本中显式记录，不能靠 `abs()` 隐藏方向错误。

已验证食指检测器：

```bash
/home/user/miniconda3/bin/python3 -m tools.measure_g20_tape_angles \
  --frames 180 --warmup-frames 45 \
  --rgb-exposure 166 --rgb-gain 32 --rgb-white-balance 4600 \
  --roi 550,185,1045,550 \
  --hsv-lower 75 50 20 --hsv-upper 165 255 255 \
  --min-area 15 \
  --marker-association chain-angle-prior \
  --morph-open-kernel 0 \
  --unwrap-pip-near-deg EXPECTED_PIP_TAPE_DEG \
  --unwrap-dip-near-deg EXPECTED_DIP_TAPE_DEG \
  --proximal-heading-prior-deg EXPECTED_PROXIMAL_HEADING_DEG \
  --segmentation-depth-min-m 0.350 \
  --segmentation-depth-max-m 0.370 \
  --max-failure-fraction 0.10 \
  --out-prefix SESSION/POINT/camera_fixed_exp_180f
```

注意：

- 上述ROI、深度和angle priors只属于已完成食指视角。
- 每根新手指先camera-only搜索自己的ROI、深度分布和priors。
- PIP/DIP检测低质量点可以保留作diagnostic，但不能自动进入LUT。

### 7.2 MCP pitch

MCP pitch只依赖掌骨与proximal两条标记，避免middle/distal碎片影响身份。

食指已验证命令：

```bash
/home/user/miniconda3/bin/python3 \
  -m tools.measure_g20_index_mcp_pitch \
  --frames 180 \
  --out-prefix SESSION/POINT/camera_fixed_exp_180f
```

食指当前默认检测参数：

```text
ROI: 500,185,1100,500
HSV: H75..165, S40..255, V20..255
depth: 0.330..0.390 m
component min area: 15 px
palm min area: 1000 px
proximal min area: 100 px
```

物理pitch：

```text
physical pitch = raw255 reference tape angle - current tape angle
```

最终拟合只使用2D投影。D435深度PCA的3D轴保留为诊断，因为窄胶带深度边缘会产生明显离群值。

遮挡硬门：

- pitch 主值必须来自真实可见的 palm 与 proximal 标记；middle/distal 只辅助确认解剖身份。
- 如果 proximal 被其他手指遮住，必须移动遮挡手指；禁止用后两条胶带推算根部角，也禁止把3个连通块拆成4个来“补” proximal。
- 深弯与伸展端可以采用不同的其他手指避让姿态，但被标定手指和相机都不得移动；每个姿态都必须先保存四个真实连通块的可见性证据。
- 正式深弯复采建议使用 `--disable-merged-split`；任一帧少于四个真实标记就记失败，不能自动进入 LUT。

其他手指应基于相同算法建立自己的工具/ROI，不要把食指的掌骨端点几何常数直接复制。

## 8. 单关节 sweep 标准流程

### 8.1 建立零位

1. 确认隔离关节已经位于批准姿态。
2. 低速回到预计伸展端。
3. 记录20组 SDK state/fault/temperature。
4. 独立启动相机，采两组180帧零参考。
5. 两组中位数差应不超过 `0.5°`。

固定胶带安装偏角只通过零参考相减消除，不能把单帧胶带夹角当作关节绝对角。

### 8.2 向屈曲端逐点

参考候选序列：

```text
255, 240, 224, 208, 192, 176, 160, 144, 128,
112, 96, 80, 64, 48, 32, 20, 12, 6, 0
```

这只是步长模板，不是端点批准：

- 每次实际相邻变化不超过17 raw。
- 先验证policy需要的工作范围，再探索full physical range。
- raw20/raw12/raw6/raw0分别视作新的机械端区域。
- 每根关节都必须独立决定是否允许到raw0。

每个点：

1. 专用工具做发送前preflight。
2. 只发送一个相邻目标。
3. 至少记录20组 state/fault/temperature。
4. 确认目标稳定、非目标活动槽变化不超过2 raw。
5. 关闭CAN进程后再启动相机180帧采集。
6. 汇总命令raw、稳定回读raw、物理角、检测质量。
7. 通过门禁后才进入下一点。

食指pitch单步命令示例：

```bash
/home/user/miniconda3/envs/env_isaaclab/bin/python \
  -m tools.run_g20_index_mcp_pitch_candidate_step \
  --sdk-root /home/user/linkerhand-ros-sdk \
  --calib CALIBRATION_CANDIDATE.json \
  --can can0 \
  --expected-start-raw START \
  --expected-start-tolerance-raw 2 \
  --pip-tolerance-raw 1 \
  --target-raw TARGET \
  --speed 5 --samples 20 --hz 5 \
  --out SESSION/POINT/motion_START_to_TARGET.json \
  --execute
```

这条命令只用于解释记录格式。开始新关节时必须换成已经测试过的对应关节工具。

### 8.3 反向回程

- 从实际安全停止点按相邻步反向回到伸展端。
- 每步仍记录fault/temperature，不做一次性大跨度返回。
- 至少在曲线中点做一次相机回程测量。
- 回到零位后再做一组180帧相机复核和20帧SDK快照。
- 最终复核必须与初始零参考**相减成一个落盘数值**并对照§9的`0.5°`门给出通过/不通过。只采到复核帧而不差值化，等于该项未评估，不能视为通过。超门时按标记朝向分解出掌部与手指各自的贡献，再决定重测还是记为零锚点偏置。
- 正反同raw物理角差即机械回差。

回差处理：

- `<=3°`：可使用中线LUT，并把回差写入不确定度。
- `3–6°`：考虑方向感知映射。
- `>6°`：阻断部署，先检查机械、SDK和控制。

## 9. 自动停止条件

任一条件发生即停止：

- serial、side、SDK或calibration digest不匹配。
- 任一fault非0，或fault telemetry不可用。
- reserved slots11..14非0。
- fresh finger frame和raw20目标映射不一致。
- 非目标活动槽稳定变化超过2 raw。
- 单个sweep目标关节温升达到8°C，或温度持续快速上升。
- 相机检测失败率超过10%。
- 60帧块中位数范围超过0.5°。
- raw向屈曲方向移动但物理角反向超过1°。
- 前后零参考漂移超过0.5°。
- 胶带脱落、相机/手掌移动。
- 现场人员看到碰撞、异常声音、卡滞或明显发热。

发送前拒绝同样要写JSON并保留。不得为了“让工具通过”而临时放宽非目标槽、fault或身份门禁。

## 10. 拟合和候选 calibration

每个主动关节至少生成：

- command raw knots。
- stable readback raw统计。
- physical rad knots。
- forward/reverse回差。
- full physical range。
- policy valid range。
- 量化误差和held-out误差。
- serial、SDK、camera、tool、输入文件哈希。

LUT要求：

- raw严格递增。
- physical rad严格递减。
- raw255通常对应伸展零位，但必须实测。
- command inversion与SDK readback使用同一份明确的物理LUT。
- 命令锚点取实际发送的target raw；稳定回读单独记录，不能用目录名代替记录内容。

每个关节先生成 `CANDIDATE ONLY` overlay。只有以下全部满足后才能合并生产artifact：

- 全关节checklist完成。
- 所有held-out点通过。
- sim limits/mimic已经更新。
- 旧checkpoint的目标范围已经审计。
- live deploy serial/digest绑定。
- 全手静态姿态和低幅动态轨迹通过。

## 11. Phase B：相机换位与四指 MCP roll

只有 Phase A `15/15` 后才能执行。

换位前：

- 确认 Phase A 最后一个关节的全部采点和最终零参考复核都已完成并落盘。Phase A 机位不需要留档，也不要求可还原；一旦有关节的采点没做完就换位，那个关节必须整体重测。
- 关闭所有相机进程。

换位后必须重新做：

- camera info、固定曝光目视确认。
- 掌部参考平面。
- 每根手指roll中性位。
- 每根手指独立ROI/depth/marker identity。
- 新的零参考；Phase A的像素、ROI、深度和priors全部失效。

MCP roll测量：

- 非目标手指的 MCP pitch 必须弯开（实测 raw96 足够）。只卷 PIP 腾不出空间：四指全伸展时 index 内收约 `7.5°` 就会顶上中指。
- **接触会在被顶的手指上留下 SDK 完全看不见的永久位移**。2026-08-03 实测：ring 全程回读 116 未变，却被顶出 `-0.38°` 且不回弹。因此宁可把非目标指弯开，也不要靠放松力矩。
- 目标指自身的 pitch/PIP 固定在已校准隔离姿态。
- 先小幅验证raw增减对应左右方向，并同时确认标记身份（命令某个 roll 槽时只应有对应那条标记动）。
- 主值使用 proximal 相对掌部参考的**2D 面内夹角**；掌部胶带在此机位只有 3–7% 有效深度，重建不出掌部平面，3D 轴只作诊断。
- 掌部胶带近竖直，朝向落在无向直线约定 `[-90,90)` 的卷绕边界上，必须先旋转 90° 再归一化，否则参考会逐帧翻号。
- 单指模式下身份不可用「排除法」推定：要求 ROI 内恰好一个标记，且其行位置落在该指的预期行带内。行带按实测最近邻间距取，不能用固定值——相邻指仅约 74 px，固定 ±70 px 会把邻指边缘卷进来。
- index/middle/ring/pinky分别完成正向、回程、回差和LUT。

## 12. Phase C：拇指 CMC yaw/roll

只有 Phase B `4/4` 后才能执行。

- `thumb_cmc_yaw`：slot10。
- `thumb_cmc_roll`：slot5。
- 已完成的thumb CMC pitch固定在物理参考位。
- 每次只改变一个轴，另一个轴和thumb MCP固定。
- 使用掌部刚性参考与拇指L形双标记/方形标记恢复相对3D姿态。
- 先做小幅slot/sign/crosstalk验证，再做正反向范围。
- 分别生成yaw和roll LUT，不能用一个二维角吸收两轴误差。

## 13. 全部标完后的统一验收

全部 `21/21` 后才进入：

1. 生成序列号绑定的全手物理 calibration artifact。
2. 更新四指 DIP 和 thumb IP mimic。
3. 更新所有实测 Isaac joint limits。
4. 重新训练受物理范围变化影响的policy。
5. 逐关节held-out sim/real静态姿态对照。
6. 全手无policy低幅阶梯/正弦轨迹。
7. policy no-send。
8. startup-only。
9. 1/3/10/20 tick空手短运行。
10. 空手全部通过后，才另立带螺丝刀commissioning计划。

当前状态不允许跳到第7步。
