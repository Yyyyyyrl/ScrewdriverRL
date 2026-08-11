# LinkerHand G20 全关节物理/软件校准计划（历史 Review Draft）

版本：2026-08-01 v1  
对象：左手 G20，序列号 `LHT20-010-415-L-B-1-D`  
相机：RealSense D435，序列号 `143322073091`  
状态：**历史设计草案，不再作为现场执行入口**

当前唯一运行手册：
[`g20-physical-joint-calibration-runbook.md`](g20-physical-joint-calibration-runbook.md)

已经完成的结果与原始证据：
[`../records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md`](../records/g20_physical_joint_calibration_20260801/CALIBRATION_RECORD_INDEX.md)

## 1. 目标

建立可独立验证、可追溯的三层关节合同：

1. Isaac/训练坐标：16 个主动关节目标 `q_sim`，单位 rad。
2. G20 SDK：20 槽 command/state raw，范围 0--255，其中 11--14 保留。
3. 实机物理坐标：外部相机测得的真实关节角 `q_physical`，单位 rad。

最终需要两条彼此独立的映射，不能继续用同一张表正算再反算证明自己正确：

- command encoder：`q_sim target -> raw command -> q_physical`
- state decoder：`raw state -> q_physical -> policy observation q`

完成后还要验证：槽位、符号、物理范围、回差、串扰、mimic 比例、静态误差、动态跟踪、Isaac 对照和空手 policy 部署。

## 2. 不允许的做法

- 不把当前 URDF 的 PIP `1.08 rad` 当作实机真值。
- 不把旧 semantic schema 的 PIP `1.57 rad` 当作实机真值。
- 不以 `command -> 当前 mapper -> inverse` 的零误差作为物理校准证据。
- 不把任意贴歪的胶带单帧夹角直接叫作绝对关节角。
- 不在没有序列号、故障、温度、相机检测和人工急停门禁时连续自动扫 raw。
- 不在校准完成后自动进入 policy；每个阶段单独 review 和放行。
- 不清除或覆盖现有 calibration、URDF、SDK patch、policy bundle；先新增版本化 artifact，并保留回滚路径。

## 3. 当前已经完成的事实基线

以下工作已完成，均未发送新的手部运动：

- D435 在线：序列号 `143322073091`、固件 `5.17.0.10`、USB 3.2。
- 1280x720@30 彩色/对齐深度可稳定采集。
- 食指四段蓝胶带可识别为 metacarpal/proximal/middle/distal。
- 180/180 帧检测成功；PIP 三个 60 帧窗口中位数范围约 `0.299 deg`。
- 食指 PIP 的 proximal/middle 深度约 0.319/0.318 m，当前视角适合 2D 角测量。
- G20 只读预检：序列号正确、SDK 3.1.0、固件 1.0.7、故障为 0。
- 食指 PIP slot16 连续 10 次 state 都为 raw255，跨度 0。
- raw255 伸展参考下，胶带 PIP 夹角中位数为 `5.4778066523 deg`；该值作为胶带安装偏角，不作为物理 q。
- 当前 SDK `get_current()` 固定返回 `-1`，没有真实电流反馈；`get_torque()` 是最大扭矩设定，不是负载测量。

已完成记录：

- `records/g20_physical_joint_calibration_20260801/camera/CAMERA_PIXEL_MEASUREMENT_REPORT.md`
- `records/g20_physical_joint_calibration_20260801/index_pip_extension_reference_001.json`
- `records/g20_physical_joint_calibration_20260801/preflight/sdk_readonly_after_camera_001.json`
- `records/g20_physical_joint_calibration_20260801/planning/freeze_manifest.json`

## 4. 总责任划分

| 工作 | 我可以独立完成 | 需要你操作/确认 | 说明 |
|---|---|---|---|
| 源码、URDF、bundle、SDK patch 冻结与哈希 | 是 | 否 | 只读检查与生成记录 |
| D435 采集、胶带识别、像素/3D 角计算 | 是 | 初次标记身份需确认 | 相机不控制手 |
| SDK 序列号/state/fault/temperature 只读预检 | 是 | 否 | 不发运动指令 |
| 生成下一条 raw 指令与安全检查 | 是 | 否 | 指令先展示，不立即发送 |
| 手掌固定、运动空间清空、急停/断电准备 | 否 | **必须** | 物理安全只能由现场人员保证 |
| 胶带/AprilTag 贴附、重新贴标、改变相机视角 | 否 | **必须** | 我可以给具体位置和检查结果 |
| 第一根手指首次运动 | 否 | **每步确认** | 你观察碰撞、声音、发热、机械异常 |
| 通过门禁后的普通中间 raw 点 | 可自动 | 你必须在场 | 是否改为批量自动需单独批准 |
| raw255/raw6/raw0 或新机械端附近 | 否 | **每点单独确认** | 端点永不默认安全 |
| 采集 SDK/camera 数据、保存视频/CSV/JSON | 是 | 否 | 每个姿态自动完成 |
| 手工复核少量角度 | 否 | 需要 | 只作为独立 sanity check，不作主测量 |
| 拟合 command/state 两套映射 | 是 | review 结果 | 自动计算，结果需共同确认 |
| 修改 URDF/schema/calibration/deploy 代码 | 是 | review diff | 不直接覆盖旧版本 |
| Isaac 渲染、离线 replay、自动测试 | 是 | 人眼对照最终确认 | 我生成并解释差异 |
| 真机 policy 空手试运行 | 我执行软件 | **你在场并放行** | 校准全通过后才进入 |

## 5. 关节范围

### 5.1 主动关节（16）

| 组 | 关节 | SDK slot | 主测量方式 |
|---|---|---:|---|
| 食指 | mcp_roll / mcp_pitch / pip | 6 / 1 / 16 | D435 3D 轴；pitch/PIP 侧视 2D 作为主值 |
| 中指 | mcp_roll / mcp_pitch / pip | 7 / 2 / 17 | 同上，需先确认无遮挡 |
| 无名指 | mcp_roll / mcp_pitch / pip | 8 / 3 / 18 | 同上，需先确认无遮挡 |
| 小指 | mcp_roll / mcp_pitch / pip | 9 / 4 / 19 | 同上，需先确认无遮挡 |
| 拇指 | cmc_yaw / cmc_roll / cmc_pitch / mcp | 10 / 5 / 0 / 15 | CMC 最终需完整 3D 姿态；MCP 可侧视角 |

### 5.2 机械联动关节（5）

- index/middle/ring/pinky DIP：分别跟随对应 PIP；当前 URDF 候选倍率 `0.8917`。
- thumb IP：跟随 thumb MCP；当前 URDF 候选倍率 `1.1619`。

这些倍率也必须实测。主动关节 offset 和 mimic 比例不能混在一个参数中吸收误差。

## 6. 测量坐标定义

### 6.1 PIP/DIP

当前蓝胶带可以使用，不要求细长笔直。每块胶带只需固定在对应刚性连杆上：

- `proximal` 与 `middle` 的 60 帧中位长轴夹角用于 PIP。
- `middle` 与 `distal` 的 60 帧中位长轴夹角用于 DIP。
- 胶带固定偏角通过伸展参考消除：

`q_physical_deg(raw) = median_tape_angle_deg(raw) - median_tape_angle_deg(extension_reference)`

食指当前 extension reference 为 slot16 raw255、胶带角 `5.4778066523 deg`。每个完整 sweep 前后都要重测 raw255；如果参考漂移超过 0.5 deg，视为胶带/相机移动，本 sweep 作废。

### 6.2 MCP roll/pitch

长指 MCP 是串联双轴，不能从任意单张 2D 图同时读出 roll 和 pitch。

- 校准 roll：固定同指 pitch/PIP 在隔离姿态，使用 D435 3D proximal 轴相对掌部参考平面的横向变化。
- 校准 pitch：固定 roll=物理中性位，使用侧视 2D 与 3D proximal 轴的俯仰变化。
- roll=0 必须由手指位于掌骨中性平面的外部几何确认，不由旧 mapper 给出。
- 如果一个蓝色长轴无法稳定区分串联轴，则需要你在该连杆上增加第二条不平行胶带，形成 L 形方向基准。

### 6.3 Thumb CMC

拇指 CMC yaw/roll/pitch 是串联三轴，单条胶带或单张 2D 图只能做粗筛，不能完成绝对 3D 校准。

推荐最终方案：

- 掌部一个固定 AprilTag/方形刚性标记。
- 拇指目标连杆一个固定 AprilTag，或两条不平行胶带形成可恢复平面方向的 L 形标记。
- D435 记录相对 SE(3)，每次只改变一个 thumb 轴，其余轴固定在已校准参考。

需要你完成标记的打印/贴附；我负责生成标记、识别、姿态拟合与误差分析。若不加 3D 标记，thumb CMC 只能标为“粗校准”，不能进入最终完美映射声明。

## 7. 分阶段执行计划

### Phase 0 — Plan review 与冻结（当前阶段）

我完成：

- 冻结 policy bundle、当前 URDF、当前 provisional overlay、SDK patch、D435 和工具哈希。
- 生成全关节 joint contract、PIP raw 候选表、相机零参考和本计划。

你完成：

- 审核本计划末尾的 review 决策。
- 明确运动批准方式和 thumb 最终标记方式。

门禁：你明确批准前，不发送新运动。

### Phase 1 — 食指 PIP pilot

目的：用风险最低、当前相机最清楚的关节跑通完整闭环。

初筛条件：

- 空手、手掌固定、你在现场、急停可用。
- speed `[20,20,20,20,20]`。
- torque limit `[40,40,40,40,40]`。
- 只改变 slot16；其余 19 槽保持冻结参考。
- 每次 raw 变化通过 2--5 秒插值 ramp，不瞬间跳变。

候选下降序列：

`255, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 20, 12, 6, 0`

返回序列根据安全停止点生成；计划候选为：

`6, 12, 20, 32, 48, 64, 96, 128, 160, 192, 224, 240, 255`

每点自动采集：

- 实际 raw command20。
- 20 次 state20，目标槽 median/min/max 和非目标槽变化。
- fault20；温度 before/after。
- D435 至少 60 个有效帧：PIP 2D/3D、DIP 2D/3D、MAD、失败率、深度差。
- 原始彩色、深度、叠加图和连续视频时间戳。

你每点检查：

- 无自碰撞、无异常声音、无松动、无明显发热。
- 胶带仍固定，相机/手掌未动。
- 决定继续、保持或安全返回。

特别规则：

- raw255、raw6、raw0 必须单独放行。
- 不要求一定到 raw0。
- SDK 没有实际电流反馈，不能用“电流到阈值”证明机械止挡。
- 角度进入平台、状态跳变、故障、温升异常、非目标关节变化或你看到风险，立即停止。

Pilot 输出：

- raw command -> physical PIP/DIP 曲线。
- raw state -> physical PIP/DIP 曲线。
- 正/反向回差。
- 可重复伸展零位、可重复屈曲上限。
- 物理 PIP 上限及不确定度；不引用 1.08/1.57 作为先验答案。

Review Gate 1：我们共同看 pilot 曲线和视频，决定是否修改 raw 步长、速度/扭矩或停止条件。

### Phase 2 — 食指 PIP operating-condition 复核

仅在 Phase 1 无异常后执行。

- 只选择伸展参考、曲线中部、平台前、平台点和一个返回点。
- 使用此前已经实机运行过的 speed40/torque80。
- 不重新扫所有点，避免无意义机械负担。
- 比较 torque40 与 torque80 下的物理上限、回差和零位回归。

物理上限必须写明扭矩条件。若上限随 torque 明显变化，则 URDF 使用保守可重复范围，calibration artifact 保存 torque-dependent 证据。

Review Gate 2：确定食指 PIP 物理上限、command/state 映射和 DIP mimic 比例。

### Phase 3 — 其余三个 PIP

在每根手指开始前先做 camera-only visibility check。

我完成：

- 自动检测目标指胶带、生成 ROI 和零参考。
- 确认目标 PIP slot17/18/19，非目标槽保持。
- 按食指 pilot 验证后的自适应 raw 序列执行并记录。

你可能需要：

- 给目标指 proximal/middle/distal 贴胶带。
- 如果被食指遮挡，临时将目标指放到可见姿态，或重新固定相机；相机一旦移动，本指要重新建立 reference。
- 确认软件没有把中指/无名指/小指身份认错。

Review Gate 3：四个 PIP 各自得到独立上限。不能只测一根指再复制给其余三根。

### Phase 4 — 长指 MCP pitch

- 每根指 roll 固定在外部确认的中性位。
- PIP 固定在已校准的低屈曲隔离点。
- 先覆盖 policy deployment range，再谨慎探索 full physical range。
- 每个 target 记录 2D/3D proximal 方向、raw command/state、回差和非目标关节。

你负责确认每根指的中性平面和相机可见性；我负责其余采集、拟合和记录。

### Phase 5 — 长指 MCP roll

- pitch 固定在已校准隔离角，PIP 固定。
- 使用 D435 3D link direction；2D 侧视只作辅助。
- 首先验证 raw 增减与左右方向符号，再测范围。
- 如果单条胶带的 3D 方向不稳定，你增加第二条不平行标记。

Review Gate 4：12 个长指主动关节和4个 DIP mimic 全部完成。

### Phase 6 — Thumb MCP/IP

- thumb CMC 三轴固定在参考姿态。
- 侧视测 thumb proximal/distal 角。
- 分别拟合 thumb MCP command/state，并测 thumb IP mimic 比例与 offset。

你负责确认 thumb 没有碰到食指/掌部，并在需要时补标记。

### Phase 7 — Thumb CMC yaw/roll/pitch

- 先做小幅 slot/符号/crosstalk 验证。
- 每次只动一个 CMC 轴，其余两轴保持已校准参考。
- 最终采用 AprilTag 或两条不平行标记恢复完整 3D 相对姿态。
- 每个轴独立做 forward/reverse 和 held-out 点。

Review Gate 5：16 个主动关节、5 个 mimic 关系均有外部物理证据。

### Phase 8 — 拟合与 calibration artifact

每个主动关节独立生成：

- 物理可重复范围与不确定度。
- command encoder 的 monotonic piecewise-linear knots。
- state decoder 的 monotonic piecewise-linear knots。
- forward/reverse 分支和回差。
- policy deployment valid range。
- full physical range（若安全完成）。
- torque/speed/serial/SDK hash/camera hash/URDF hash/bundle hash。

拟合点：20%、50%、80%。  
保留验证点：35%、65%，不得参与拟合。  
预处理：从10%方向接近拟合点，从90%方向接近 held-out 点。

映射策略：

- 回差小于等于3 deg：使用中线映射并保存不确定度。
- 回差大于3 deg但小于等于6 deg：运行时使用方向感知分支。
- 回差大于6 deg：阻断 policy，先查机械/SDK/控制问题。

新 artifact 建议：

`assets/calibrations/linker_g20_left_lht20_010_415_physical_v2.json`

它必须序列号绑定、digest 绑定，并分别包含 `command_encoder` 与 `state_decoder`；不能再用一套 `lo/hi/flip` 同时承担两条物理链。

### Phase 9 — 软件接入

我完成以下代码工作并给你 review diff：

1. 修正/重建 semantic schema，使其与实测范围和当前训练坐标的用途明确分离。
2. 不再让 live deploy 依赖 module-global `apply_calibration()`。
3. 每个控制 session 加载一个不可变、序列号/digest 绑定的 mapper。
4. command 和 state 使用不同的校准段。
5. 部署日志同时保存：`q_target_sim`、`raw_cmd`、`raw_state`、`q_state_physical`、饱和/方向分支。
6. 保留旧 calibration 文件，增加显式 `--calib`/version 选择和 rollback。

必须区分：

- 当前 checkpoint 的训练坐标合同：保持 bundle 里的 q 定义，校准只保证 `q_sim -> 同角度 q_physical`。
- 未来训练 URDF：实测范围写入新 URDF/schema 后需要重新训练，不能假装旧 checkpoint 自动获得新物理分布。

如果实测物理上限小于当前 checkpoint 会命令的 q 上限，当前 policy 直接阻断，不能靠静默 clamp 继续部署。

### Phase 10 — 自动测试与离线 replay

我可以独立完成：

- joint order/slot/reserved-slot 测试。
- 每段 mapping 的单调、边界、digest、serial 测试。
- 录制数据上的 fit/held-out 误差回放。
- command/state 分离测试，防止重新退化为同表自证。
- dirty worktree 与旧 artifact 不被覆盖检查。
- policy bundle 的所有目标都位于实测 valid range 检查。

### Phase 11 — Isaac/实机姿态对照

我完成：

- 对每个 held-out q 在 Isaac 中设置精确关节 rad 并渲染。
- 使用同一关节顺序、mimic 关系和 checkpoint task contract。
- 生成 real camera / Isaac 并排图、关节数值表和差值视频。

你完成：

- 人眼确认关节身份、弯曲方向、姿态是否符合机械直觉。
- 对任何视觉不一致提出复测，而不是只接受数值表。

### Phase 12 — 真机静态合同测试

不运行 policy，只发送一组 held-out 静态 q：

- 检查 `q_target_sim -> raw_cmd -> q_physical`。
- 检查 `raw_state -> q_state_physical`。
- 每点 forward/reverse 各一次。
- 所有点通过后才允许动态测试。

### Phase 13 — 真机空手动态轨迹与 policy

先运行无 policy 的低幅正弦/阶梯轨迹，验证10 Hz下的延迟与跟踪误差；之后才运行用户指定的冻结 policy：

1. no-send 只读计算。
2. startup-only。
3. 1 个 policy tick。
4. 3 个 policy ticks。
5. 10--20 ticks 短运行。

每一级都单独 review。空手通过不代表带螺丝刀通过；带物体属于后续独立 commissioning。

## 8. 自动判停条件

任何一项发生即停止，不继续下一 raw：

- 序列号/SDK hash/calibration digest 不匹配。
- 任一 fault 非0或 fault telemetry 不可用。
- state 连续无效、目标 state 与 command 长时间明显不收敛。
- 非目标主动槽变化超过2 raw count，或相机发现非目标关节明显运动。
- D435 标记失败率超过10%。
- 60帧 PIP中位数分组范围超过0.5 deg。
- PIP 两标记中位深度差超过15 mm。
- q 随 raw 出现超过1 deg 的反向非单调变化。
- 胶带 reference 回归漂移超过0.5 deg。
- 你观察到碰撞、异常声音、松动或发热。

温度数值门槛需要结合本手基线和厂商安全范围最终确认。在此之前采用保守临时门禁：单个 joint sweep 内温升达到8 degC 或出现快速持续上升即停止；该值需你在 review 中确认/修改。

## 9. 验收标准（建议值，等待 review）

### 测量系统

- 每姿态至少60个有效帧。
- 检测成功率不低于90%。
- 60帧窗口中位数范围不高于0.5 deg。
- 2D/3D 的角度变化差异不高于1 deg；超过则该点不用于拟合。

### 槽位与机械

- 16/16 主动关节槽位、符号、单调性正确。
- 保留槽11--14始终为0。
- 非目标关节视觉变化不高于1 deg，state变化不高于2 raw count。
- 每个 mimic 比例都有单独外部测量与不确定度。

### 静态 command encoder

- held-out MAE 不高于1.5 deg。
- held-out 最大误差不高于3 deg。
- return-to-zero 不高于1.5 deg。

### 静态 state decoder

- held-out MAE 不高于1.5 deg。
- held-out 最大误差不高于3 deg。
- forward/reverse 分支选择正确。

### 动态

- 先报告10 Hz延迟、settling time、稳态误差，不用静态阈值掩盖动态滞后。
- 稳定后 q tracking 中位误差建议不高于2 deg、最大不高于4 deg。
- 任一 fault、持续 rail saturation 或映射 valid-range 外目标阻断 policy。

## 10. 记录目录合同

每次运动 session 单独建立不可覆盖目录：

`records/g20_physical_joint_calibration_20260801/sessions/<timestamp>_<joint>_<phase>/`

至少包含：

- `session_manifest.json`：所有输入路径、SHA256、serial、speed/torque、操作者确认。
- `command_steps.csv`：计划/实际 raw20、ramp、时间戳。
- `sdk_samples.csv/json`：state/fault/temperature。
- `camera_samples.csv`：逐帧角度、深度、检测质量。
- `camera_color/depth/annotated` 和视频或时间戳索引。
- `summary.json`：零位、上限、fit/holdout、回差、判停原因。
- `REPORT.md`：结论、异常、是否放行下一阶段。

失败和中止 session 同样保留，不覆盖、不删除。

## 11. 用户 review 决策

请逐项批准或修改：

- [ ] 范围：完整16主动关节 + 5 mimic，而不只是 PIP。
- [ ] 当前先执行食指 PIP pilot；第一遍每个 raw 点由你单独确认。
- [ ] 初筛 speed20/torque40；仅平台附近用 speed40/torque80 复核。
- [ ] raw6/raw0 永远是条件候选，不要求必须到达。
- [ ] 当前 D435 + 蓝胶带作为 PIP/DIP 主测量，60帧中位数。
- [ ] Thumb CMC 最终使用 AprilTag 或两条不平行标记；否则不声明最终精标完成。
- [ ] command encoder 与 state decoder 使用独立 piecewise mapping。
- [ ] 接受第9节建议误差门槛，或给出修改值。
- [ ] 接受临时温升8 degC门禁，或提供厂商/实验室规定值。
- [ ] 每个 Review Gate 停止并先出报告，不自动进入下一阶段。

## 12. 计划批准后的第一步

不是直接跑完整 sweep，而是只执行一个最小运动验证：

1. 再做一次 serial/fault/temperature/state/camera reference 只读预检。
2. 展示完整 raw20 before/after diff，确认只有 slot16 变化。
3. speed20/torque40，slot16 从255缓慢 ramp到240。
4. 采集60帧相机与20次SDK state/fault/temperature。
5. 返回结果和叠加图，等待你 review；不自动进入224。

这个单步通过后，我们才决定是否继续食指 PIP pilot。
