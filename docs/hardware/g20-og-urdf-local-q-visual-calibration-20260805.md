# G20 全 21 关节：physical raw → OG URDF local q 视觉配准

日期：2026-08-04 至 2026-08-05  
对象：LinkerHand G20 左手，`LHT20-010-415-L-B-1-D`  
状态：**21/21 人工视觉配准、静态整手回归和三段动态 A/B 人工复核通过；16-active LUT 与训练安全资产已晋升生产**

## 1. 结论

这次工作的最终标定合同是：

```text
真实手稳定 SDK readback raw [0,255]
        ↕  同视角人工视觉匹配
linker_hand_l20_OG 的 URDF local revolute joint q [rad]
```

它不是“相机画面里胶带投影角度 → rad”，也不是“把旧 visual rad 直接塞进 Isaac”。
所有 21 个关节都使用正式实拍样本和同视角 Isaac 网格图重新人工匹配，结果均满足：

- 每个关节的人工匹配 LUT 无方向单调性违规；
- 每个关节在开始标定前都通过相机视角 A/B gate（用户明确允许共用的视角除外）；
- 每个关节完成代表性五点 real / Isaac / overlay 人工复核；
- 所有最终 archive 状态均为 `visual_review_pass_candidate_only_not_promoted`；
- 21 个匹配结果都没有卡在诊断 q 网格边缘；
- 生产 OG URDF 全程未改，SHA256 始终为
  `697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f`。

独立候选产物：

- 16 个主动关节 overlay：
  [`assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805.json`](../assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805.json)
- 21 关节完整溯源、量化平台合并和 follower 拟合：
  [`assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805_provenance.json`](../assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805_provenance.json)
- 全部原始视觉配准记录：
  [`records/g20_urdf_local_q_visual_registration_20260804`](../records/g20_urdf_local_q_visual_registration_20260804)

候选生成时不会被默认加载；其后的独立静态和动态 gate 通过后，已按第 14.4–14.5 节
晋升到生产 `linker_calib_deploy.json`、签名 mapper 合同和训练资产。原始 candidate/provenance
保持不变，仍是晋升输入证据而不是可变生产文件。

## 2. 为什么旧的 physical angle 标定不能直接作为 Isaac q

最初的 21 关节物理标定可靠地建立了 SDK raw、相机可见运动和机械重复性，
但其中的 visual rad 不一定与 URDF local q 共享同一个零点和旋转轴：

1. 单目画面中的 2D 胶带角是关节运动在相机平面的投影；只要关节轴不垂直于画面，
   投影角就不等于 3D revolute q。
2. D435 深度重建能辅助诊断，但胶带边缘、遮挡和窄标记会导致深度离群；
   直接拟合 3D 轴仍会混入手掌摆放和相机外参误差。
3. URDF q 的零点由 `<origin>`、`<axis>` 和 link 坐标系定义，不是“照片里看起来伸直”或
   某个 raw 端点。
4. `thumb_cmc_roll` 曾暴露最明显的错误：历史做法把 raw248 平移为 semantic zero，
   但真实照片与 OG URDF `q=0` 在正确侧视图中不重合。该中间结果被明确否决。
5. MCP roll 的符号也说明不能假设所有关节都满足“raw 增大、q 减小”：
   四个 MCP roll 的 OG-local q 实测随 raw 增大而增大。

因此，旧的 visual measurement 只保留为来源与诊断信息；新的 matcher 中刻意隐藏旧角度，
防止人工选择被旧数值锚定。

## 3. 硬件、相机与软件基线

真实手：

- 型号：LinkerHand G20 left；
- 序列号：`LHT20-010-415-L-B-1-D`；
- 主动 SDK slots：16 个；
- 机械 follower：`index_dip`、`middle_dip`、`ring_dip`、`pinky_dip`、`thumb_ip`；
- reserved slots：11–14，任何运动脚本都不得写入。

正式相机：

- Intel RealSense D435，serial `143322073091`；
- `1280x720 @ 30 fps`；
- exposure `166`，gain `32`，white balance `4600`；
- auto exposure 和 auto white balance 均关闭；
- 相机 intrinsics 随各标定数据集固化并带 SHA256。

解释器：

- RealSense / OpenCV：`/home/user/miniconda3/bin/python3`；
- Isaac Lab：`/home/user/miniconda3/envs/env_isaaclab/bin/python`。

生产坐标模型：

- [`assets/linker_hand_l20_OG/linkerhand_l20_left.urdf`](../assets/linker_hand_l20_OG/linkerhand_l20_left.urdf)
- 标定期间它只作为只读 reference；扩展 limit、移除 mimic 等操作只发生在各 session
  的 `diagnostic_asset/linkerhand_l20_left.urdf` 副本中。

## 4. 每个关节的标准流程

### 4.1 选择正式实拍样本

沿用前一轮安全物理 sweep 留下的固定 D435 照片。大多数关节约每 raw16 一个样本；
LUT 的 raw 使用稳定 readback，不使用“曾经发送的 command raw”。正反向相同 stable raw
作为重复性和迟滞证据，不被当成两个独立坐标点。

### 4.2 相机 A/B gate

每个新关节开始前，先用一张正式照片和 OG URDF 基准姿态生成 real / Isaac A/B 图，
由用户人工确认视角、透视、手掌位置和画面旋转。未通过时只调整相机；不得开始 q 匹配。

关键实践：

- 相机必须位于实拍中四指朝向或指定侧面的真实方向，不能仅凭“手大致在画面中”判断；
- 使用 D435 的真实 `fx/fy/ppx/ppy`，不能用默认 Isaac 透视；
- 允许在 image plane 做已记录的平移/旋转来复现实拍构图；
- 背景改为较深灰色，避免白色手模型与背景融合；
- 匹配单个手指时可隐藏不相关手指的 visual body，但掌部和目标手指必须保留；
- 隐藏 visual 不得改变 joint state、碰撞、URDF joint origin 或 axis。

### 4.3 生成带余量的 q 网格

对 production OG URDF 做 session-local 副本，只扩展目标关节诊断 limit，并在 production
limit 两端留余量。网格通常为 0.02 rad；MCP roll 和后期拇指为 0.01 rad。

扩展 limit 的目的只是避免把人工选择截断在旧边界上。它不表示生产 limit 已获准改变。

### 4.4 人工匹配

交互 matcher 对每张正式照片显示同视角 Isaac q 网格。人工选择视觉上最重合的
OG-URDF local q，并可标记 uncertain。若大量样本落到网格边缘，则扩大网格重渲染；
不得把边缘值直接当成测量终点。

### 4.5 数值审计

每个关节生成：

- stable raw → selected `q_urdf_local_rad` LUT；
- 相同 raw 的合并结果；
- q 网格半步量化不确定度；
- 单调性违规列表；
- 仅用于诊断的 affine fit、RMSE 和最大残差；
- 已观察 q 范围与 production OG limit 的对比；
- 原始照片、render、相机、URDF 和 summary 的 SHA256。

Affine fit 不是最终映射。最终候选使用人工匹配的 piecewise-linear LUT。

### 4.6 五点人工复核与归档

从 LUT 中选择约五个覆盖全范围的姿态，重新渲染 real / sim / overlay panel。
人工通过后用 `tools/archive_g20_manual_match_review.py` 生成不可混淆的 archive manifest，
状态仍保持 candidate-only。

## 5. follower / mimic 关节的特殊处理

五个 follower 没有独立 SDK slot，不能写入 16-joint overlay：

```text
index_dip  <- index_pip
middle_dip <- middle_pip
ring_dip   <- ring_pip
pinky_dip  <- pinky_pip
thumb_ip   <- thumb_mcp
```

对 DIP/IP 的人工匹配采用以下方法：

1. 在隔离的 diagnostic URDF 中只移除目标 follower 的 `<mimic>`；
2. 每张照片先按同一 raw 从已独立匹配的 parent LUT 取得 parent q；
3. 固定 parent q，只扫描 follower q；
4. 人工选择 follower link 与实拍最重合的 local q；
5. 将 follower q 对 parent q 做 affine 和 through-origin 两种诊断拟合。

对于 `thumb_mcp` 主动关节的扫描，保留 OG `thumb_ip` mimic `1.1619`；当 parent q
诊断范围超过旧 limit 时，parent 和 follower 的 diagnostic limit 同时扩展，防止 follower
被 clamp 后造成“拇指乱飘”或错误匹配。

本次重新拟合结果如下。它们只进入 provenance，尚未写入生产 URDF 或 semantic schema：

| follower | parent | affine multiplier | offset rad | affine RMSE rad | through-origin multiplier | RMSE rad |
|---|---|---:|---:|---:|---:|---:|
| index_dip | index_pip | 0.780692 | -0.000973 | 0.04751 | 0.779956 | 0.04751 |
| middle_dip | middle_pip | 0.777635 | +0.012545 | 0.06458 | 0.787110 | 0.06485 |
| ring_dip | ring_pip | 0.864940 | -0.066561 | 0.04418 | 0.813453 | 0.05449 |
| pinky_dip | pinky_pip | 0.784565 | +0.051065 | 0.05876 | 0.825638 | 0.06469 |
| thumb_ip | thumb_mcp | 1.079962 | +0.016655 | 0.05321 | 1.099367 | 0.05396 |

这些值表明 OG URDF 的四指 `0.8917` 和拇指 `1.1619` 不应在没有视觉回归的情况下
继续被视为已验证真值。是否更新 mimic 必须作为独立 sim-real gate 处理。

## 6. 21 关节最终结果

`observed q` 是人工匹配到的 OG URDF local q 范围；`grid` 是完整诊断候选范围和步长；
`outside` 是落在原 OG limit 之外的 unique raw 数；RMSE 仅是 affine 诊断误差。

| joint | unique raw | observed q rad | diagnostic grid rad / step | original OG limit rad | outside | affine RMSE | monotonic violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| index_dip | 15 | [0.000, 1.480] | [-0.400, 2.100] / .020 | [0.000, 1.400] | 1 | .0515 | 0 |
| index_mcp_pitch | 19 | [0.000, 1.260] | [-0.400, 1.900] / .020 | [0.000, 1.400] | 0 | .0094 | 0 |
| index_mcp_roll | 18 | [-0.270, 0.180] | [-0.500, 0.500] / .010 | [-0.170, 0.170] | 5 | .0069 | 0 |
| index_pip | 15 | [0.040, 1.800] | [-0.400, 2.100] / .020 | [0.000, 1.570] | 4 | .0300 | 0 |
| middle_dip | 19 | [0.000, 1.440] | [-0.400, 2.200] / .020 | [0.000, 1.400] | 2 | .0558 | 0 |
| middle_mcp_roll | 18 | [-0.290, 0.180] | [-0.500, 0.500] / .010 | [-0.170, 0.170] | 5 | .0055 | 0 |
| middle_mcp_pitch | 19 | [0.000, 1.240] | [-0.400, 1.900] / .020 | [0.000, 1.400] | 0 | .0175 | 0 |
| middle_pip | 19 | [0.060, 1.760] | [-0.400, 2.200] / .020 | [0.000, 1.570] | 5 | .0354 | 0 |
| ring_dip | 19 | [0.020, 1.480] | [-0.400, 2.200] / .020 | [0.000, 1.400] | 2 | .0187 | 0 |
| ring_mcp_roll | 20 | [-0.250, 0.200] | [-0.500, 0.500] / .010 | [-0.170, 0.170] | 6 | .0092 | 0 |
| ring_mcp_pitch | 19 | [0.000, 1.240] | [-0.400, 1.900] / .020 | [0.000, 1.400] | 0 | .0082 | 0 |
| ring_pip | 19 | [0.020, 1.760] | [-0.400, 2.200] / .020 | [0.000, 1.570] | 4 | .0348 | 0 |
| pinky_dip | 19 | [0.020, 1.360] | [-0.400, 2.200] / .020 | [0.000, 1.400] | 0 | .0367 | 0 |
| pinky_mcp_roll | 18 | [-0.230, 0.240] | [-0.500, 0.500] / .010 | [-0.170, 0.170] | 6 | .0101 | 0 |
| pinky_mcp_pitch | 18 | [0.000, 1.140] | [-0.400, 1.900] / .020 | [0.000, 1.400] | 0 | .0250 | 0 |
| pinky_pip | 19 | [0.030, 1.720] | [-0.400, 2.200] / .020 | [0.000, 1.570] | 4 | .0369 | 0 |
| thumb_cmc_yaw | 33 | [-0.120, 1.120] | [-0.400, 1.900] / .010 | [0.000, 1.400] | 5 | .0124 | 0 |
| thumb_cmc_roll | 34 | [0.420, 1.780] | [0.000, 1.900] / .020 | [0.000, 1.220] | 12 | .0172 | 0 |
| thumb_cmc_pitch | 19 | [0.000, 0.840] | [-0.250, 1.100] / .010 | [0.000, 0.790] | 2 | .0176 | 0 |
| thumb_mcp | 19 | [0.000, 1.210] | [-0.250, 1.550] / .010 | [0.000, 1.050] | 4 | .0262 | 0 |
| thumb_ip | 19 | [0.000, 1.250] | [-0.250, 1.550] / .010 | [0.000, 1.220] | 2 | .0353 | 0 |

“outside” 不等于自动批准扩大生产 limit。它只说明物理手在安全 sweep 中出现了与旧 OG
limit 外 local q 最匹配的姿态；下一阶段必须在相同坐标合同下做完整 sim-real 动作验证。

## 7. 最终相机视角与人工 gate 决策

每个 joint archive manifest 都固化了最终 camera path 和 SHA256；完整逐关节索引见 provenance
的 `joint_archives`。重要的共享或人工调整视角如下：

- 四指 MCP roll 共用：
  `four_finger_mcp_roll_shared_camera/camera_archive_roll_shared_palm_image_cw5deg_candidate.json`，
  画面顺时针 5°，深灰背景；用户明确批准 ring/pinky 沿用同一视角。
- `thumb_cmc_yaw`：
  `thumb_cmc_yaw/archive_formal_palm_l_v1/camera_projective_rigid_palm/camera_projective_image_up150px_left25px_candidate.json`。
- `thumb_cmc_roll`：
  `thumb_cmc_roll/manual_view_gate_user_adjusted_v4/camera.json`；修正了早期把白色掌标平面解释错误而得到的顶视角。
- `thumb_cmc_pitch`、`thumb_mcp`、`thumb_ip` 共用用户精调后的食指侧视角：
  `thumb_cmc_pitch/camera_user_adjusted_rotate180_image_right160px.json`；画面旋转 180°，
  再向右平移 160 px。pitch/mcp/ip 配准时固定 `thumb_cmc_yaw≈0.66`、
  `thumb_cmc_roll≈1.22`，以复现实拍中的拇指展开姿态。
- pinky PIP/DIP 使用用户 live 调整后的精确透视：
  `pinky_pip/interactive_camera_view_20260804_cw90_refine_exact_perspective/camera_user_refined_local_y_plus5mm_image_right_13p5mm.json`。

## 8. 独立候选 config 的生成规则

生成器：

```bash
python3 tools/build_g20_og_local_q_candidate_config.py
```

它执行以下硬检查：

1. 必须找到且只找到 21 个 `archive/five_point_spotcheck_manifest.json`；
2. joint 名集合必须正好等于 16 active + 5 follower；
3. 每个 archive 必须是人工通过、candidate-only；
4. summary SHA256 和 production OG URDF SHA256 必须匹配；
5. 每个 summary 必须是 0 monotonic violation；
6. active LUT 必须可被部署 loader 解析；
7. follower 只写 sidecar，不伪造 SDK slot。

旧 loader 错误地要求 LUT q 随 raw **严格递减**。本次数据证明 MCP roll 是递增，因此
`linker_sdk_map.py` 已做向后兼容扩展：q 可以严格递增或严格递减，forward inverse
根据 LUT 自身方向选择。默认 `DEFAULT_JOINTS` 不变，未加载候选时行为完全不变。

人工网格的相邻样本偶尔会选择相同 q。逆映射需要一一单调，所以生成器仅将连续同 q
平台合并为一个 mean-raw knot，并在 provenance 中保存原始 knots 与合并详情。
本次发生在：

- `index_pip`：q1.80，raw4/6 → raw5；
- `ring_mcp_roll`：q-0.05，raw119/122 → raw120.5；
- `thumb_cmc_yaw`：q0.42，raw151/154 → raw152.5；
- `thumb_cmc_roll`：q1.50 raw58/59；q1.12 raw119/129/130；q0.52 raw228/234。

除此之外不做 affine 替代、不平滑人工 q、不把 q 截到旧 URDF limit。

## 9. 常用复现工具

相机和渲染：

- `tools/interactive_g20_camera_viewer.py`
- `tools/render_g20_hand_pose.py`
- `tools/compose_g20_focus_view_gate.py`
- `tools/make_g20_diagnostic_joint_limit_urdf.py`
- `tools/make_g20_diagnostic_follower_urdf.py`

人工匹配和汇总：

- `tools/prepare_g20_manual_joint_q_match.py`
- `tools/prepare_g20_manual_follower_q_match.py`
- `tools/manual_match_g20_joint_q.py`
- `tools/summarize_g20_manual_joint_q_matches.py`
- `tools/compose_g20_manual_match_spotcheck.py`
- `tools/archive_g20_manual_match_review.py`

候选生成：

- `tools/build_g20_og_local_q_candidate_config.py`

运行具体工具前应先读取 `--help` 和目标 joint archive 的 manifest；不同关节的 base pose、
camera、ROI、hidden bodies 与 q grid 不应凭记忆重建。

## 10. 安全与数据完整性规则

- RealSense 与 CAN 运动进程不能并发占用资源；正式照片和稳定 raw 采集分阶段执行。
- 任何上电动作前核对 hand serial、side、20-slot 长度、reserved slots、fault、temperature、
  当前 raw 和候选 digest。
- 运动使用逐步限速、稳定回读和 fault gate；不要直接跳到随机全幅姿态。
- 相机或手掌移动、胶带脱落、曝光改变时，当前视角 gate 失效；必须重新 gate。
- diagnostic URDF 只能存在于记录目录或明确的 candidate 目录，不能覆盖 production OG。
- 旧 visual rad、2D tape angle、3D axis diagnostic 和 URDF local q 必须使用不同字段名；
  不允许都叫 `rad` 后凭上下文猜测。
- 所有 config promotion 都必须显式记录旧/新文件 digest 和回退路径。

## 11. 候选静态验证结果

验证器：

```bash
python3 tools/validate_g20_og_local_q_candidate.py
```

机器可读报告：
[`records/g20_og_local_q_candidate_validation_20260805/static_validation.json`](../records/g20_og_local_q_candidate_validation_20260805/static_validation.json)

2026-08-05 结果：

- `tests/test_linker_sdk_map.py`：`30 passed, 1 xfailed`；xfail 是已有的 production-asset
  endpoint 合同项，不是本次新增失败；
- 16 个 active joints、16 个唯一非 reserved slots；
- 候选通过 `build_joint_table()`，且只构建 table 不会改变 module-active mapping；
- 所有 candidate joints 都有 piecewise LUT；
- 每个 LUT 整数有效 raw 域逐点验证，`raw→q→raw` 最大误差为 `0 raw`；
- 每关节 1001 个均匀 q 点验证，经过 uint8 command 后 `q→raw→q` 全局最大误差
  `0.010000000000000009 rad`，最差关节为 thumb yaw/roll，等于人工 0.01/0.02 rad
  网格和 raw 量化共同决定的预期级别；
- 四个 MCP roll 被验证为 increasing-q LUT，其余 12 个 active joints 为 decreasing-q LUT；
- 验证结束调用 `reset_calibration()`，默认 mapping 成功恢复；
- production OG URDF SHA256 仍为
  `697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f`。

candidate affine follower 相对当前 OG mimic 的 RMSE 降幅：index DIP `64.4%`、middle DIP
`53.3%`、ring DIP `57.6%`、pinky DIP `38.2%`、thumb IP `24.4%`。这支持继续做
candidate-mimic 视觉 A/B，但还不足以直接修改 URDF。

## 12. 下一阶段验证门

候选生成不等于标定已可用于训练。应按顺序完成：

1. schema、digest、raw/q 单调性、16-joint slot 唯一性；
2. 全关节 raw→q→raw 和 q→raw→q 数值回环，报告量化误差；
3. 检查 active q 范围与实际用于验证的 sim URDF limit 的交集；
4. 对五个 follower 比较 OG mimic、candidate affine mimic 与独立人工 q；
5. 使用已匹配的真实 D435 视角，在多个低风险离散姿态做 Isaac / 真机左右 A/B；
6. 通过后做短连续动作视频 A/B，重点观察方向错误、拇指 self-collision、follower 耦合和迟滞；
7. 输出独立 validation report；
8. 只有报告和人工复核都通过，才讨论修改 runtime config、semantic schema、URDF 或训练资产。

## 13. 不应再重复的错误

- 不要把相机画面的方向差当成关节 q 差；先确认相机在正确的物理侧面。
- 不要只看脚本轮廓分数；最终 gate 是人工观察 link、关节轴和整根手指的重合。
- 不要把 raw command 当 stable readback。
- 不要把每个关节的物理角都假设为 raw 增大时减小。
- 不要为了让 config 通过旧 schema 而翻转 URDF q 符号。
- 不要把 follower 塞入不存在的 SDK slot。
- 不要用 production limit 截断诊断网格；也不要因为诊断网格更大就自动扩大 production limit。
- 不要在拇指 parent 超范围时忘记 follower mimic 的 clamp；这会产生看似 self-collision 的错误姿态。
- 不要在相机视角未人工通过时批量生成后续 joint renders。

这篇文档、21 份 archive manifest 和候选 provenance 三者共同构成可复现记录：
文档解释方法和决策，archive 保存逐关节视觉证据，provenance 提供机器可读的完整索引与 digest。

## 14. 2026-08-05 sim→SDK→真机独立回归结果

最终技术报告：

- 自包含 HTML：
  [`records/g20_og_local_q_candidate_validation_20260805/report/report.html`](../records/g20_og_local_q_candidate_validation_20260805/report/report.html)
- canonical artifact：
  [`records/g20_og_local_q_candidate_validation_20260805/report/artifact.json`](../records/g20_og_local_q_candidate_validation_20260805/report/artifact.json)
- presentation A/B 关键图：
  [`records/g20_og_local_q_candidate_validation_20260805/report/key_visual_real_candidate_og.jpg`](../records/g20_og_local_q_candidate_validation_20260805/report/key_visual_real_candidate_og.jpg)
- 15 姿态完整 A/B：
  [`records/g20_og_local_q_candidate_validation_20260805/comparisons/exp300_15poses_settled_candidate/matched_pose_contact_sheet.png`](../records/g20_og_local_q_candidate_validation_20260805/comparisons/exp300_15poses_settled_candidate/matched_pose_contact_sheet.png)
- follower candidate/OG 消融：
  [`records/g20_og_local_q_candidate_validation_20260805/comparisons/mimic_affine_vs_og_15poses/calibration_ablation_contact_sheet.png`](../records/g20_og_local_q_candidate_validation_20260805/comparisons/mimic_affine_vs_og_15poses/calibration_ablation_contact_sheet.png)

### 14.1 固定条件

- 当前相机：RealSense D435，serial `814412070035`，1280×720；
- 固定 RGB：exposure `300`、gain `32`、white balance `4600`，auto exposure/WB 关闭；
- Isaac 使用用户人工通过的同视角 camera pose，intrinsics 使用当前 D435 实测值；
- 3 个诊断姿态 + seed `20260805` 的 12 个随机姿态，共 15 个；
- 真机动作使用 speed `20`、40-step ramp、20 Hz、settle 1.5 s、每姿态 7 个稳定样本；
- Isaac 使用 settled readback q 做 kinematic render，不进行 physics settle，因此 joint drift 应为 0；
- 验证结束将手安全返回初始候选姿态。

### 14.2 16 个 active SDK joints：通过 candidate gate

- 15/15 姿态完成，240 个 active-joint settled observations；
- `|q_readback-q_target|` mean `0.004485 rad`、median `0.003682 rad`、
  P95 `0.011010 rad`、max `0.032949 rad`；
- 最大项是 `random_05` 的 thumb CMC roll，约 3 raw counts；
- 15/15 post-pose fault vectors 全零；
- 最高温度 `52 °C`；
- 15 个 Isaac kinematic renders 的最大写入/读回漂移 `0 rad`；
- 返回初始姿态最坏残差 `0.020 rad`，无 fault；
- 用户人工通过完整 15 姿态 OG-mimic A/B contact sheet；直接视觉检查显示 palm/base 对齐稳定，
  四指主动链方向和幅值总体一致，没有 thumb self-collision 乱飘、链条翻转或错误耦合。

因此，16-joint piecewise LUT overlay 可以判定为
`hardware_visual_candidate_pass`。这表示它可继续作为**独立候选**使用，不等于默认 mapping 已切换。

### 14.3 5 个 mechanical followers：证据混合，不晋升

独立 candidate-mimic URDF 只修改 index/middle/ring/pinky DIP 与 thumb IP 的 mimic
multiplier/offset，不修改 limits、active joints 或 production OG。生产 OG SHA256 始终是：

```text
697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f
```

逐关节 archive fit 中，candidate affine 相对 OG mimic 的 RMSE 下降 `24.4%–64.4%`；
但同一 15 姿态的独立整手回归没有稳定胜出：

- whole-hand contour mean：OG `19.497 px`，candidate `19.581 px`；
- candidate 在 15 个姿态中 8 个局部变化带指标更好；
- follower-focused mean：candidate 比 OG 高 `0.058 px`，约 `0.6%`；
- 改变量远小于 real/URDF mesh、深度分割和残余 camera translation 带来的系统误差。

所以结论不是“candidate follower 更差”，而是
`inconclusive_not_promoted`：保留人工 q 和 affine fit 作为 provenance，production 继续用 OG mimic，
直到有 follower-specific side view、fiducial 或更可靠的 link-level 测量。

### 14.4 当前 promotion 决策

| 层 | 结论 | 当前动作 |
|---|---|---|
| 16 active raw↔OG local q LUT | 静态、数值、15-pose 和三段动态真机/视觉 gate 通过 | 已晋升 `linker_calib_deploy.json` |
| 5 follower mimic | archive fit 有信号，独立 A/B 不足以稳定区分 | 不修改 production OG URDF |
| OG joint limits | 本轮没有独立验证 limit 扩张 | reference OG 不修改；不晋升诊断扩张 |
| runtime mapping | 16-active piecewise LUT 已人工通过 | 生产 overlay 与 immutable mapper 均绑定同一 LUT/digest |
| training asset | 动态 gate 通过；limit 扩张仍独立 | 已晋升为“reviewed active range ∩ OG limits”，保留 OG mimic |

### 14.5 动态 gate 与 2026-08-05 生产晋升

完成并由用户人工判定通过的同步连续动作 A/B：

- [`01_flex_extend_AB.mp4`](../records/g20_og_local_q_candidate_dynamic_validation_20260805/videos/01_flex_extend_AB.mp4)：190 帧，20 Hz；
- [`02_mcp_roll_fan_AB.mp4`](../records/g20_og_local_q_candidate_dynamic_validation_20260805/videos/02_mcp_roll_fan_AB.mp4)：214 帧，20 Hz；
- [`03_thumb_opposition_AB.mp4`](../records/g20_og_local_q_candidate_dynamic_validation_20260805/videos/03_thumb_opposition_AB.mp4)：214 帧，20 Hz。

三段均用 SDK 命令墙钟时间对齐 D435 和 Isaac 目标帧；轨迹执行期间 fault 全零、最高温度
`53 °C`，每段均回到执行前姿态。用户明确回复“校验通过，可以进行训练/部署资产晋升”。

生产资产：

- deployment overlay：`linker_calib_deploy.json`，SHA256
  `b728cd58f15408193080a0bd35dc6ae281adff9a96d6115ded012498b6f7ecb7`；
- training URDF：`assets/linker_hand_l20/linkerhand_l20_left.urdf`，SHA256
  `1fe1f1db92347f2f5ef7b05458f95d68ae6f30f1dc5fe2ec581c0c3725a6bddc`；
- signed schema digest：`1bb09533ebf945a693d2726f394b5004cbf0dc2132f045e7d45d3c106a5181ad`；
- signed calibration digest：`354d057686e2d47e562b0cf3e40993c677e39bd651302b14b1b88495c28a62dd`；
- 完整晋升与回滚清单：
  [`records/g20_og_local_q_asset_promotion_20260805/promotion_manifest.json`](../records/g20_og_local_q_asset_promotion_20260805/promotion_manifest.json)。

训练关节边界只采用 `reviewed LUT range ∩ unchanged OG geometric limit`；不扩大 OG limit。
五个 follower 继续使用 OG mimic（四指 `0.8917`、拇指 `1.1619`），未把证据不足的 affine fit
混入生产。晋升前文件的 byte-for-byte 副本位于
`records/g20_og_local_q_asset_promotion_20260805/rollback/`。

### 14.6 训练兼容性边界

本次“训练资产晋升”只表示训练使用的 URDF、semantic schema 和 16-active joint limits
已经切换到实物视觉验证后的坐标合同；它**不表示旧 checkpoint 或旧 task home posture
自动兼容**。

扩展回归发现旧 top-down contact target 的 `thumb_cmc_yaw=1.2406667465 rad`，超过本次
验证后的上限 `1.12 rad`。该任务还要求 target 至少保留 `0.105 rad` joint-limit margin，
所以合规 target 应不高于 `1.015 rad`。旧策略包则明确绑定旧的 `1.38–1.40 rad` yaw rail，
不能只改 manifest 后继续部署。

在新 URDF 上进行了不修改生产 task 的重建试验：

- 多起点离线 fit 得到了全部关节均在新范围内的候选；
- dense mesh 与 contact-role 精炼后，五个 distal contacts 和 self-collision gate 通过；
- 但 `pinky_middle` 到物体的 non-distal clearance 只有 `1.068 mm`，低于严格门槛
  `1.5 mm`；
- fingerwise 全局优化可以恢复该间隙，但会失去 middle/ring 的有效接触。

因此没有把任何试验候选写入
`screwdriver_rl/tasks/linker_l20/screwdriver_rotation_topdown_posture.py`。旧 checkpoint、
旧 candidate 117 和旧 immutable policy package 保留为历史资产，状态为
`incompatible_with_promoted_calibration`。新训练应先在晋升后的 URDF 上重新建立
top-down wrist/grasp baseline，通过 static mesh、Isaac contact persistence 和 DR gate，
再从该 baseline 训练并导出新的 digest-bound policy package。

机器可读兼容性报告：
[`records/g20_og_local_q_asset_promotion_20260805/training_compatibility.json`](../records/g20_og_local_q_asset_promotion_20260805/training_compatibility.json)。

### 14.7 旧 checkpoint 的 deterministic coordinate adapter

“旧 checkpoint 原生不兼容”不等于权重完全不能复用。本轮新增了一个与 learned Stage-2
proprio adapter 分离的 deterministic coordinate adapter。旧 policy 内部继续运行在训练时的
虚拟 joint-q 坐标；进入 SDK 前转换到新硬件合同，真实 readback 在送入 actor/history 前执行
逆变换。

逐轴审计发现旧 action rails 中只有 `thumb_cmc_yaw` 超出新验证范围。因此没有对 16 轴做
range-fraction 重映射：15 轴严格 identity；thumb yaw 使用
`hardware_q = legacy_policy_q - 0.26 rad`，逆向 readback 加 `0.26 rad`。这样旧 action span、
`0.05 rad` delta scale、虚拟 target integration 和 proprio history 都保持不变；旧 home yaw
从虚拟 `1.2406667` 映射到硬件 `0.9806667 rad`，旧 reset yaw 从 `0.7977368` 映射到
`0.5377368 rad`，旧 action upper `1.38` 映射到新验证 upper `1.12 rad`。

adapter 同时绑定旧 checkpoint SHA256、新 production overlay SHA256、calibration artifact digest
和 semantic schema digest；任一文件改变、joint order/limit 不一致或 adapter digest 被修改，
部署入口均 fail closed。离线测试已证明 actor action、虚拟 history 与 target integration 等价，
但 adapter 当前状态仍是 `offline_validated_live_gate_required`。允许 dry-run、no-send、
startup-only，以及显式 `--candidate-adapter-gate` 的最多 100 ticks 录制式 CAN gate；普通无限制
live policy 会被拒绝，直到 gate 通过并将状态晋升为 `live_promoted`。

完整设计、命令、证据与放行流程见：
[`docs/g20-legacy-topdown-policy-coordinate-adapter-20260805.md`](g20-legacy-topdown-policy-coordinate-adapter-20260805.md)。
