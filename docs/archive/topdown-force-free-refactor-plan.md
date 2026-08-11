# Topdown 任务去力重构计划（Force-Free Reward Refactor）

> **SUPERSEDED (2026-08-08).** The force-free reward refactor for the mounted top-down screwdriver task. The project moved to the HORA formulation instead; the accepted policy is documented in [TRAINING_REPORT_20260807.md](../../TRAINING_REPORT_20260807.md). Kept for the failure analysis in the 2026-07-29..07-31 entries, which is still the best record of why force-based grip rewards were abandoned.

> 任务：`Isaac-LinkerL20-Screwdriver-Rotation-Topdown`（注册配置
> `LinkerL20ScrewdriverRotationTopdownDiameterRandEnvCfg`，60/64/68 mm 手柄）
> 日期：2026-07-27　|　分支：`rand`
> 最终目标：**完整去力重构 + Stage 1/2 全量重训 + 部署级 policy 性能验收**

---

## 1. 背景与动机

真机部署效果与 Isaac 仿真差距大，且对螺丝刀放置精度极其敏感。对照两份参考实现
（hora / allegro_inhand_rotation：持续旋转；MFR_benchmark：定量 90° 转螺丝刀）
逐行核对后确认，本仓库是三者中**唯一把接触力的模拟数值当作优化目标**的实现：

| | dex-forge topdown | hora | MFR_benchmark |
|---|---|---|---|
| 力参与 reward | 6 项使用力幅值/阈值 | 完全不用（tensor acquire 后从未引用） | 不用力，只用指尖-手柄轴**距离**二值 gate |
| 力参与 obs | privileged obs 含每指力 5 维 | 无（仅 pos/scale/mass/friction/COM） | 无 |
| 抓握维持 | 每步"接触收入"（grip/authority/idle） | 隐式：掉落终止 + pose_diff 惩罚 | 距离 gate + 掉落终止 |
| 防捏碎 | 力 > 8 N 惩罚 | torque² + (torque·vel)² 做功惩罚 | 无 |

接触力 = 接触刚度 × 穿透深度，是仿真里最不可迁移的量：PhysX 接触参数
（本任务 contact offset 特意压到 0.25 mm）决定了"关节目标越过接触面多深 → 多少牛"，
而部署路径是纯位置控制 + proprio-only 观测，策略在真机上只能复现**关节目标轨迹**，
无法复现力。真机手柄刚度远高于仿真软接触，同样的目标穿透在真机上要么堵转要么悬空。

---

## 2. 现状问题清单（按严重度排序）

### P0-1　力窗口 shaping 把行为拟合到"仿真牛顿值"

涉及代码（`screwdriver_rl/tasks/linker_l20/screwdriver_rotation_env.py`，
`_get_rewards()` L368–628；窗口定义 `screwdriver_rl/core/rewards.py:95` `force_window`；
阈值 `screwdriver_rotation_env_cfg.py:222-225`，注释自证 "FIRST-CUT values" 从未标定）：

| 项 | 行号 | 力的用法 |
|---|---|---|
| `contact_authority_reward` | env.py:447 | `combined_gate`（sustained ≥3 指 F>0.1 N × soft_count(force_window) × upright） |
| `grip_reward` | env.py:475 | 5 指 `force_window(F_total)` 均值，最优 = 每指 0.5–4 N |
| `drive_reward` | env.py:457-473 | `force_window` × 切向速度因子 |
| `idle_cost` | env.py:513 | **每根** F<0.1 N 的手指罚款（= 强制 5 指全贴） |
| `excess_cost` | env.py:511 | `relu(F − 8 N)` |
| `wrong_surface_cost` | env.py:512 | 非指尖链接的 net force **幅值** |
| milestone / 进度指标 gate | env.py:481-486 | `sustained_gate`（力二值）限定 `eval_net_turns` 等 |

机制：P0 阶段（cfg L287-304）`w_contact_authority=8.0 + w_grip=1.5 + w_idle`
规避 ≈ 每步 ~11 的"压力收入"，无需任何旋转技能即可获得；早期 turn reward
（120 × fwd_vel）只有十几。策略最先学到的技能是"找一组 joint target 偏移量使
仿真中每指稳定压出 1–4 N"，这个力设定点行为在真机上不可复现。

注意：主 turn reward **已经不被力 gate**（topdown 配置
`reward_physical_progress_outside_contact=True`，
`screwdriver_rotation_topdown_env_cfg.py:70`），所以去力改造只动外围项，主干不变。

### P0-2　Privileged obs 含 5 维力 → 污染部署 latent

`_compute_privileged_obs()`（env.py:634-653）：priv obs 21 维
（euler3 + angvel3 + rel_pos3 + quat4 + friction1 + **F_total5** + geom2）。
actor 是 latent-conditioned，**euler/tilt 信息只能通过该 latent 获得**
（raw euler 不是独立 actor 输入，cfg L170-187）。Stage 2 adapter 从 30×32 proprio
历史回归 priv obs；仿真里"跟踪误差 ↔ 力"的相关性由仿真刚度定义，真机映射不同
→ 力 5 维系统性回归偏差 → **连带污染 latent 里的姿态估计**。这是独立于行为
塑形的第二条 deploy-gap 通道。

### P0-3　相对位姿零随机化（与力正交，但同等致命）

`tasks/base/screwdriver_rotation_env.py` `_reset_idx`（L640-716）：螺丝刀为固定
安装 3-DOF euler joint；手根位姿 per-bucket 固定表；reset 仅随机绕轴角
（`randomize_obj_start`）与固定 tilt 表。**训练从未见过 ≥1 mm 的放置误差**，
"部署时不 100% 对齐就崩"首先源于此，P0-1 的力设定点行为将其放大。

### P1-4　DR 覆盖面缺口（实际配置核对版）

实际生效的 DR（`DomainRandCfg`，base cfg L144-231；per-episode，按相位
`dynamics_randomization_scale` 0.25/0.60/1.0 向中心收缩）：
rotation damping ×(0.5,2.0)、screwdriver mass ×(0.5,2.0)、
load torque ×(0.5,1.5)（base 0.045 N·m → 0.023–0.068）、
finger stiffness/damping ×(0.8,1.2)、几何 60/64/68 mm、obs 白噪声 σ=0.01/步。

缺口（按危害排序）：
1. **接触摩擦默认关、固定 1.5**（`randomize_contact_friction=False`）——真机
   指垫-手柄摩擦不可控，hora 随机绝对值 0.3–3.0；
2. **负载力矩范围未对照真机标定**——若真螺丝 breakaway 力矩 >0.07 N·m 即在
   训练分布外，可直接解释"真机转不动"；且 stick-slip（静>动）未建模；
3. **关节零点偏置缺失**——白噪声可被策略滤掉，per-episode 常量偏置（真机
   零点标定误差的形态）完全没有；
4. 相对位姿零随机化（即 P0-3）；tilt 轴承阻尼 DR 默认关；无延迟 DR。
5. 决定"穿透→力"映射的接触刚度/柔度参数完全固定（去力后危害降级，仍影响
   接触动力学迁移）。

### P2-5　次要问题

- `idle_cost` 的 per-finger 全罚强制 5 指全贴：topdown 姿态下 pinky/ring 可能顶歪
  螺丝刀，真机上未必是好策略。
- checkpoint 选择指标（`eval_total_turns`/`eval_net_turns`）被力二值 sustained gate
  限定（env.py:481-483），阈值 0.1 N 的仿真力噪声直接影响选型。

---

## 3. 重构设计

原则：**行为塑形与状态估计中不出现任何力幅值**。力传感器读数仅保留为
diagnostic extras（日志/audit 用，`_read_contact_forces` 不删）。每项替代物都
必须是运动学量或真机同样可测的量。

### 3.1 接触判定：力二值 → 指尖-手柄轴距离（MFR 同款）

- 复用 `rewards.point_segment_distance`（core/rewards.py:222，已有单测基础设施）。
  轴段 = handle base origin → cap origin（`_compute_fingertip_tangential_speed`
  已在取这两个 body，env.py:326-332）。
- 新 primitive（加入 `core/rewards.py`，纯 torch + 单测）：

```python
def distance_window(dist, d_contact, d_far) -> Tensor:
    """1 when dist <= d_contact, linear ramp to 0 at d_far. [0,1]."""

def contact_present_dist(dist, d_contact) -> Tensor:
    """Binary kinematic contact predicate: dist <= d_contact."""
```

- 新 cfg 字段（`LinkerL20ScrewdriverRotationEnvCfg`）：
  - `contact_d_margin`：指尖 pad 有效厚度余量（初值 0.008 m，见 3.7 标定）。
  - `contact_d_far_margin`：ramp 外沿（初值 0.02 m）。
  - `d_contact = handle_radius × geom_scale + contact_d_margin`（geometry DR 下
    per-env，复用 `scale_drive_speed_with_geometry` 同款 `_env_geom_scale` 机制）。
- 替换点：
  - `drive_count` / `valid_contact`（env.py:410-413）→ `contact_present_dist`；
    `sustained_binary_gate` 与 `turn_contact_hold_steps=3` 机制不变。
  - `contact_quality` 的 `soft_count_gate` 输入 → `distance_window` 得分。
  - milestone gate 与 `qualified_delta_z` 进度指标同步换距离 gate。

### 3.2 grip / drive / idle 的力得分 → 距离得分

- `grip_reward = w_grip × distance_window(dist).mean(-1)`。
- `drive_reward = w_drive × (distance_window(dist) × tang_factor).mean(-1) × upright_gate`
  （切向速度因子为运动学量，原样保留，含 per-bucket 半径缩放）。
- `idle_cost`：从 per-finger 全罚改为**接触数不足罚**：
  `w_idle × relu(min_contact_fingers − contact_count_dist)`。
  不再强制 5 指全贴；3 指以上即无罚。

### 3.3 防捏碎：力超限 → 目标穿透代理（hora 做功惩罚的位置控制等价物）

```python
def target_penetration(cur_targets, q, deadband) -> Tensor:
    """sum_j relu(|target_j - q_j| - deadband)^2 — '命令位置越过实际位置多深'，
    即挤压意图。自由空间跟踪误差落在 deadband 内不罚。"""
```

- `excess_cost = w_excess × target_penetration(cur_targets, finger_q, pen_deadband)`。
- 真机上同样可测（同一对量：命令目标 vs 编码器位置），物理含义一致，是本次
  重构中最关键的替代项。
- `pen_deadband` 标定：从现有训练日志取自由运动阶段 |target−q| 的 P95（预估
  0.03–0.06 rad，见 3.7）。注意 mimic/coupled 从动关节不参与（只取 16 个独立 DOF）。
- 权重量纲从 N 变为 rad²，`w_excess` 需重标定：初值取"P0 阶段一个中等挤压
  （超 deadband 0.1 rad × 5 指）≈ 原 excess_cost 在 5 N 超限时的量级"。

### 3.3b Co-motion 授权 gate 与零张力起步（2026-07-28 事故后追加，见 §7）

首次去力重训收敛到"冻结手指 + 求解器蠕变白拿 turn reward"的 exploit
（§7）。两项结构性修正：

- **Co-motion 授权**（`rewards.surface_co_motion` + cfg `turn_motion_authorized`）：
  每指得分 = 指尖速度在"手柄表面该点刚体速度"（由 handle body 世界系
  twist 计算，倾斜/进动下无符号歧义）上的投影比 ∈ [0,1]；
  `motion_auth = soft_count(co_motion × distance_present, ≥2 指)`。
  turn reward、milestone、qualified 进度指标（checkpoint 选拔）全部乘以它：
  手柄转而手指不动 → 一分不给。手柄近静止（表面速度 ≤ 0.002 m/s）时 gate
  常开，不阻碍起转。topdown 的 `reward_physical_progress_outside_contact`
  语义保留，但"物理进度"现在必须是**指尖共动**的进度。
- **零张力起步**（`reset_zero_tension_targets`，base env settle 末尾 snap）：
  settle 结束后把 `cur_targets` 对齐到实测关节位置（耦合从动关节与 RMA
  历史帧同步重写）。消灭常驻目标穿透 —— 它既是蠕变伪影的能量来源，也让
  "维持抓握"永久越过 `pen_deadband` 被罚。起步后握持免费，挤压是策略决策。

### 3.4 wrong-surface：幅值 → 二值存在性（保留传感器，去幅值）

`wrong_surface_cost = w_wrong × (wrong_force > 1e-3).float()`。
二值存在性对刚度不敏感，保留防"手掌倚靠"exploit 的作用；hora/MFR 均无此项，
若后续验证 home_dev + 姿态约束已足够，可在 A/B 中删除。

### 3.5 Privileged obs：力 5 维 → 距离接触得分 5 维（维度契约不变）

- `F_total(5)` 替换为 `distance_window(dist)(5)`（[0,1]，运动学量，adapter 可从
  proprio 历史回归且真机语义一致）。
- 力 5 维原位替换后 priv 仍 21 维；叠加 §3.6b 的摩擦双通道 +1 维后，**最终
  契约 priv 22 维 / obs 54 维**（`history_obs_dim 32 + privileged 22`）。语义与
  维度均变化，与旧 checkpoint 不兼容（本就要求全量重训），需要在 policy
  package 元数据里 bump obs 语义版本
  （`tools/export_policy_package.py` / `tests/test_topdown_deploy_metadata.py`）。

### 3.6 对齐鲁棒性：reset 相对位姿 DR（新增，非去力但必须同批重训）

**范围的确定原则**：训练范围 ≥ 真机误差预算的 1.5–2 倍。真机误差 =
手臂/手基座标定 + 螺丝刀夹具定位 + 每次放置重复性 的叠加，各层典型 1–3 mm，
RSS 合计 2–4 mm、最坏 5–7 mm。**±3 mm 只在标定良好的场景够用；目标最终相
±5 mm**，实际取值以两个实测为准：
- （M6 前置）实测 rig 放置重复性：同一流程放置 ≥20 次量分布，要求实测误差
  ≤ 训练范围的一半；
- （M3 门槛）settle 吸收上限 sweep：±2/±4/±5/±8 mm 各测 settle 后 ≥3 指
  距离贴合率，取"≥95% 的最大幅度"为最终范围。

**运动学余量核算**（±5 mm 可行性）：`home_deviation_deadband=0.1 rad` ≈ 指尖
~6 mm 自由行程（~60 mm 有效杆长），±5 mm 所需姿态偏离 ~0.08 rad 仍在免罚区；
`joint_motion_range=0.35 rad` ≈ 21 mm 行程，±5 mm 消耗 <25%。若上限要推到
±8 mm（~0.13 rad），需同步将 deadband 放宽至 ~0.15 rad。

**维度与初值**（敏感度排序：手根 roll/pitch > xy > 轴向 z > yaw——手柄圆柱
对称，yaw 仅重排手指方位角）：
- `reset_root_pos_noise_m = 0.005`（xy 均匀 ±5 mm；z ±2 mm）
- `reset_root_tilt_noise_rad = 0.025`（手根 roll/pitch ±1.5°，最敏感维度，起步保守）
- `reset_root_yaw_noise_rad = 0.09`（±5°）
- `reset_screwdriver_tilt_noise_rad = 0.03`（绕既有 per-bucket tilt bias 表叠加
  ±1.7°——真机螺孔不垂直；tilt 本就是仿真自由 DOF，零实现成本）
- 全部噪声乘以相位 `dynamics_randomization_scale`（P0 ±1.25 mm → P1 ±3 mm →
  P2 ±5 mm），与现有 DR 收缩模式一致，避免破坏 P0 抓握学习。
- 实现点：base env `_reset_idx` 写 `root[:, :3]` / `root[:, 3:7]` 处
  （L651-659）叠加 per-env 每次 reset 重采样的噪声（**per-episode 常量**，
  模拟准静态标定误差，不是逐步噪声）；现有 32-step compliant settle +
  `reset_target_ramp` 机制原样吸收误差，手指自行贴合。
- 该噪声同时供给 Stage 2：adapter 由此学会从 proprio 推断实际接触几何——
  这正是部署真正需要的能力（`rel_pos` 已在 priv obs 中，此前恒为常数、无信息量）。

### 3.6b 既有 DR 数值逐项修订

| 项 | 现值 | 评估与修订 | 优先级 |
|---|---|---|---|
| 接触摩擦 | 关（固定 1.5） | **开启**，绝对值范围放宽至 (0.6, 2.5)；见下方 priv obs 注 | P0 |
| 负载力矩 | ×(0.5,1.5)，base 0.045 N·m | **先实测真螺丝启动/滑动力矩**，base/range 覆盖实测 ±50%；范围下限勿抬高（低阻尼防 free-spin 探针仍需覆盖） | P0 |
| 关节零点偏置 | 无 | 新增 per-episode per-joint 常量 ±0.015 rad（真机零点标定误差形态；白噪声不可替代） | P0 |
| 手指刚度/阻尼 | ×(0.8,1.2) | 去力后行为不再追力设定点，±20% 基本够；P2 相 A/B 放宽到 (0.7,1.3)（cfg 注释警告过宽毁抓握，不一步到位） | P1 |
| tilt 轴承阻尼 | 关 | 开启 ×(0.5,2.0)，零成本 | P1 |
| obs 白噪声 | σ=0.01/步 | 量级合理，保留 | — |
| 旋转阻尼 | ×(0.5,2.0) | 合理，不动 | — |
| 螺丝刀质量 | ×(0.5,2.0) | 安装式，影响有限，不动 | — |
| 延迟 | 无 | 10 Hz 下真机 SDK 延迟 <1 步；可选 0–1 步 action delay | P2 |
| 接触刚度/柔度 | 固定 | 可选实验：`RigidBodyMaterialCfg` compliant contact 随机化 | P2 |

**priv obs 注**：开启接触摩擦 DR 后，friction 槽位会从负载力矩代理**切换**为
接触摩擦（env.py:641-646 的 if/else 分支），两个量都需暴露给 adapter——priv obs
增 1 维同时保留两者（本次重构已 bump obs 语义、必须重训，成本仅为契约
21→22、obs 53→54，与 §3.5 一并落实并更新 deploy 元数据）。

### 3.7 标定与预检（改代码之前做）

1. 用现有 checkpoint 跑 `eval.py --num_envs 256`，从 extras 导出：
   - 每指 tip-axis 距离分布（接触帧 vs 非接触帧）→ 定 `contact_d_margin`/`d_far`；
   - |target−q| 分布（自由 vs 接触）→ 定 `pen_deadband`；
   - 现有 per-finger force 分布 → 留作重构前后对照基线。
2. 复用 `tools/audit_linker_l20_topdown_functional_contact_gate.py` 交叉验证：
   距离二值 vs 力二值在现有 policy 下的一致率（预期 >90%；不一致帧人工抽查）。

### 3.8 Curriculum：结构不动，语义替换

三相结构、step 门槛（0/40 M/90 M）、turn weight 递增、DR scale 递增全部保留
（对照 MFR 四相设计，结构本身不是问题）。逐相权重语义映射：

| 字段 | 旧语义 | 新语义 | P0/P1/P2 初值 |
|---|---|---|---|
| `w_contact_authority` | 力 gate 收入 | 距离 gate 收入 | 8.0 / 5.0 / 3.0（不变） |
| `w_grip` | force_window 均值 | distance_window 均值 | 1.5 / 0.6 / 0.3（不变） |
| `w_drive` | 力×切速 | 距离×切速 | 0.5 / 1.0 / 1.2（不变） |
| `w_idle` | per-finger 缺力罚 | 接触数不足罚 | 0.3 / 0.5 / 0.6（不变） |
| `w_excess` | N 超限 | 穿透代理 rad² | 按 3.3 重标定后等比 0.5/1.0/1.5 |
| `w_wrong` | 力幅值 | 二值存在 | 1.0 / 2.0 / 3.0 × 重标定系数 |

---

## 4. 实施里程碑与验收标准

### M1　Reward primitives（core/rewards.py + 单测）

改动：新增 `distance_window` / `contact_present_dist` / `target_penetration`；
`force_window` 等旧 primitive 保留（audit/diagnostic 引用）。

验收：
- [ ] `python -m pytest tests/test_rewards.py -q` 全绿；新 primitive 各有边界
  单测（deadband 内为零、ramp 端点、per-bucket 半径广播），风格对齐现有用例。
- [ ] 不依赖 Isaac Sim（纯 CPU torch）。

### M2　Env/cfg 重构（reward + priv obs + 元数据）

改动：`screwdriver_rotation_env.py` `_get_rewards`/`_compute_privileged_obs`
按 §3.1–3.5 替换；cfg 新字段 + curriculum 权重表；deploy 元数据 bump。

验收：
- [ ] `grep -n "force_window\|F_total\|F_body\|F_cap" _get_rewards` 范围内无
  力幅值参与任何 reward 项（wrong-surface 仅二值）；`_compute_privileged_obs`
  无力幅值。
- [ ] obs shape 53 不变；`tests/test_linker_topdown_cfg.py`、
  `test_linker_topdown_geometry_dr.py`、`test_topdown_deploy_metadata.py`
  更新后全绿。
- [ ] `tools/audit_linker_l20_topdown_functional_contact_gate.py` 扩展后报告：
  现有 checkpoint 下距离 gate 与力 gate 一致率 ≥90%（记录不一致案例）。
- [ ] 单 env smoke run（`train.py --num_envs 32 --max_env_steps ~50k`）reward
  各项量级与 §3.8 表预期一致（tensorboard 逐项曲线人工检查，无 NaN、无单项
  吞噬总和）。
- [x] **零动作探针（§7 事故后新增，硬性前置）**：`eval.py --zero_action`
  下 authorized net turns ≈ 0 且 coasting `fwd_velocity_without_contact`
  < 0.03 rad/s——确认无可被 reward 兑现的"免费前向旋转"伪影；同时静置
  `Excess` ≈ 0——确认零张力起步生效、握持不被穿透代理收税。
  （零动作下 60% episode 因无支撑而倾倒、伴随少量回转漂移属预期物理；
  回转不产生任何收入。2026-07-28 实测：authorized 0.0002 圈、
  without_contact 0.025 rad/s、Excess 0.001、零动作每步总 reward −9.8。）

### M3　Reset 位姿 DR + 既有 DR 修订

改动：base env `_reset_idx` 噪声注入 + cfg 字段（§3.6）；开启接触摩擦 DR、
tilt 阻尼 DR、关节零点偏置（§3.6b）；priv obs 摩擦双通道（22 维）。

验收：
- [ ] `tools/audit_linker_l20_screwdriver_topdown_domain_rand.py` 扩展：采样
  1k 次 reset，位姿噪声分布符合配置。
- [ ] **吸收上限 sweep**：xy 噪声 ±2/±4/±5/±8 mm 各档测 settle 后 ≥3 指
  距离贴合率，取"≥95% 的最大档"为最终训练范围（预期落点 ±5 mm；若 ±4 mm
  已不达标则收窄并在 M6 相应收紧 rig 标定要求）。
- [ ] 噪声=0 时逐位复现旧 reset 状态（回归测试）。
- [ ] （真机前置，可与 M4 并行）实测 rig 放置重复性 ≥20 次 + 真螺丝
  启动/滑动力矩：确认 训练位姿范围 ≥ 2× 实测位姿误差、负载力矩范围覆盖
  实测值 ±50%——不满足则先改 cfg 再进 M4。

### M4　Stage 1 全量重训

命令（budget ≥130 M env steps，覆盖 P2 起点 90 M 之后 ≥40 M）：

```bash
python train.py --task Isaac-LinkerL20-Screwdriver-Rotation-Topdown \
  --stage 1 --num_envs 16384 --headless
```

验收（对照基线 `docs/m0-baseline-2026-07-16.md` 与重构前最优 run）：
- [ ] P0 末（~40 M）：距离接触数 ≥3 的步占比 >90%；无 reward-suicide 迹象
  （fall 率不随训练上升——沿用 hold-first curriculum 的既有判据）。
- [ ] 收敛后（final phase，DR on）`eval.py --num_envs 256`：
  - median `eval_net_turns` ≥ 重构前基线的 90%（允许小幅回撤，换鲁棒性）；
  - fall/终止率 ≤ 基线；`eval_osc_ratio` ≤ 基线；
  - `--rot_damping_scale 4.0` 探针不塌（防 free-spin hacking 回归）。
- [ ] **去力验证**：diagnostic 力 extras 的每指力分布不再向 [0.5, 4] N 窗口
  集中（对照 §3.7 基线直方图）；均值压力显著下降或至少不再是双峰贴窗。
- [ ] **反蠕变验证（§7 事故后新增）**：oracle eval 的 coasting 指标
  `fwd_velocity_without_contact` < 0.03 rad/s（0.25 即为蠕变在驱动）；
  旋转期间 `eval_motion_auth` 中位数 > 0.7（手指真的在共动驱动）；
  `eval.py --zero_action` 复测保持 ≈ 0。
- [ ] **鲁棒性验证（新增 eval 模式）**：eval 时对手根位姿施加固定偏置 sweep
  （0 / ±2 / ±4 / ±5 / ±8 mm，含 ±1.5° 手根倾斜档）：训练范围内（≤±5 mm）
  median net_turns ≥ 无偏置值的 80%；±8 mm（分布外）允许性能下降但不得
  系统性掉落/堵转（考察退化是否平滑）。

### M5　Stage 2 重训 + 部署差距测量

```bash
python train.py --task Isaac-LinkerL20-Screwdriver-Rotation-Topdown \
  --stage 2 --headless --checkpoint runs/<...>/nn/<best>.pth
```

验收：
- [ ] Adaptation MSE（held-out）< 0.01（docs/2-stage-training.md 既有判据）；
  分通道检查：接触得分 5 维与 rel_pos 3 维的回归误差单独报告（这是本次重构
  改善的通道，必须显著低于旧力通道的回归误差）。
- [ ] **Deploy-gap probe**：为 topdown 增加 oracle-vs-adapter 对比（复用
  `tools/probe_inhand_resets.py --adapter_checkpoint` 的实现模式，该工具目前
  硬编码 inhand 任务；可提参数化或另立 `probe_topdown_deploy_gap.py`）：
  adapter-latent 模式 median net_turns ≥ oracle-latent 模式的 90%。
- [ ] 位姿偏置 sweep 在 **adapter-latent 模式**下复测：≤±5 mm 保持 ≥80%。

### M6　最终 policy 性能评测（真机）

前置：`tools/export_policy_package.py` 导出、
`tests/test_deploy_rate_contract.py` / `test_deploy_policy_bundle.py` 全绿。

验收协议（每项 ≥10 次试验，记录视频与电流日志）：
- [ ] 标称放置：连续正向旋转 ≥2 整圈不掉落、不堵转（电机电流不触限）。
- [ ] 放置偏差容忍：人为 ±3 mm 与 ±5 mm / ±2° 偏置下成功率分别 ≥80% / ≥60%
  （对照重构前真机表现；±5 mm 档若训练范围经 M3 收窄则同步下调）。
- [ ] 前置数据核验：M3 实测的 rig 重复性与螺丝力矩仍在训练分布内（若 rig
  或螺丝更换需复测）。
- [ ] 转速达到仿真 eval median fwd_vel 的 ≥50%（sim2real 转速折损经验上限）。
- [ ] 失败模式记录：悬空/堵转/顶歪三类占比对照重构前——**堵转与悬空类失败
  应显著减少**（这两类正是力设定点行为的真机表征）。

---

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 距离阈值标定失准（0.25 mm contact offset 下距离≈半径处梯度陡） | §3.7 先用现网 checkpoint 标定 + 一致率审计门槛 90% 再动训练 |
| 穿透代理 deadband 过小 → 自由运动被罚、过大 → 防捏碎失效 | 日志 P95 标定 + M2 smoke run 逐项曲线检查 |
| `w_excess`/`w_wrong` 重标定引发训练不稳 | 逐项 A/B：先只替换接触判定与 priv obs（保留旧 excess）训一版对照，再全量 |
| 位姿噪声破坏 pregrasp 贴合 | M3 吸收上限 sweep（±2/±4/±5/±8 mm）定范围；95% 贴合门槛为硬约束 |
| 真机负载力矩在训练分布外 | M3 前置实测真螺丝力矩，range 覆盖实测 ±50% 后才进 M4 |
| 新语义 net_turns 与旧指标不可比 | `eval_raw_net_turns` 不 gate、语义不变，作为跨版本可比指标 |

## 6. 明确不做的事

- 主干 reward 项（turn/reverse/upright/fall/tilt_vel/home_dev/action*）不引入
  力因素；turn 的 co-motion 授权（§3.3b）是纯运动学 gate，不违背此条。
- 不动三相 curriculum 结构与 step 门槛。
- 不删 ContactSensor 与 `_read_contact_forces`（降级为 diagnostic-only）。
- 不做 Stage 3 actor 微调（沿用 docs/2-stage-training.md 的既有决策，M5 数据
  不达标时再评估）。

---

## 7. 训练事故记录

### 2026-07-28　冻结手指 / 蠕变旋转 exploit（run `force_free_m1_m5_proxy_20260728_v4`）

**现象**：Stage 1 收敛后确定性策略动作全零（play `ActionCost 0.000`），手指
静止在 pregrasp 姿态，螺丝刀以 ~0.28 rad/s 缓慢自转；`TurnRew ~46/step`
照常发放，`best` checkpoint 的 `authorized_net_turns` 达 2.06 圈/episode。

**证据链**：
- `Drive 0.204` → 指尖切向速度 ~0.005 m/s < 手柄表面线速度 0.009 m/s：手柄
  在静止手指下打滑，手指没有驱动。
- oracle eval coasting：`fwd_velocity_without_contact 0.249` vs
  `in_contact 0.313`——脱离接触距离后转速几乎不降；0.045 N·m 负载
  （`omega_eps=0.05`，该转速下全额生效）+ 手柄惯量 ~1e-4 kg·m² 下自由旋转
  应毫秒级停转 → 驱动源为物理伪影而非手指。
- `MaxJointDev 0.193 rad`：pregrasp 目标常驻压入，持续向接触求解器注energy
  （PhysX 深穿透位置目标蠕变），tilt bias 下接触法向不对称将其整流为绕轴转。

**根因**（三者叠加，缺一不成立）：
1. turn reward 只被 upright gate 门控（`reward_physical_progress_outside_
   contact=True`），蠕变旋转照付；
2. `target_penetration_scale=100` + `pen_deadband=0.06` <常驻压入 0.19：
   维持抓握每步被罚 ~5，任何探索性挤压即刻加罚 → 通往主动旋转的梯度被税
   封死，冻结成为局部最优；
3. 距离接触语义被静止姿态满足（DriveCnt=4 常开）→ 接触收入与 qualified
   进度指标全部被动可得，checkpoint 选拔选中"最会蠕变"的权重。

**修复**（本文档 §3.3/§3.3b 已并入设计）：
- F1 co-motion 授权 gate（turn/milestone/qualified 指标 × motion_auth）；
- F2 `target_penetration_scale` 100 → 10（cfg 注释含定价推导）；
- F3 `reset_zero_tension_targets`：settle 末尾把目标 snap 到实测关节位置；
- F4 `eval.py --zero_action` 物理伪影探针 + `tests/test_rewards.py` 中
  `test_creep_exploit_regression_motion_auth_gate_closes` 回归单测；
  M2/M4 验收新增零动作与 coasting 门槛。

**教训**：任何"物理进度"奖励在给钱前必须验证进度确实由策略的作用产生
（零动作探针本应是 M2 的第一道关）；穿透代理的 deadband 必须以"验证抓握
静息态免费"为标定基准，而不是自由运动。

**修复验证（2026-07-28，`eval.py --zero_action`，v4 phase3 checkpoint，
104 episodes）**：

| 指标 | 事故 run | 修复后零动作 | 判定 |
|---|---|---|---|
| authorized net turns / ep | 2.06 圈（蠕变计入选拔） | **0.0002 圈** | ✓ gate 生效 |
| coasting `fwd_vel_without_contact` | 0.249 rad/s | **0.025 rad/s** | ✓ 蠕变消除 |
| 静置 `Excess` /step | ~5.0（握持被收税） | **0.001** | ✓ 零张力生效 |
| 零动作每步总 reward | **+5.5**（冻结有利可图） | **−9.8** + 60% 倾倒罚 | ✓ exploit 经济性死亡 |

零动作下 60% episode 倾倒还恢复了 hora 式的隐式压力：不主动扶持就损失
upright/fall 项——"什么都不做"从最优策略变成最差策略。

### 2026-07-28　v8 遗留红灯排查（final ep488 checkpoint）

**① 64 mm DR fall 38.8%（红）→ 根因：reset 静默窗捕获失败，已定位到两个 cfg 数字。**
fall-timing 直方图（64 mm 专项，311 次倒）：p10=14 / p50=16 / p90=21 步，
86% 的倒发生在 ≤20 步——恰好是 hold(5)+ramp(10)=15 步动作静默窗的边缘。
零张力起步（F3）移除了旧版靠常驻压入实现的被动夹持，而静默窗让策略在手柄
开始倾倒时无行动权；68 mm 手柄粗、指笼几何自锁所以幸免（fall 2.9%）。
被 reject 的 64 mm 视频逐帧证实：1.6 s 内手柄从未咬合的指间滑出倒下。
**同 checkpoint A/B（64 mm + DR，256 envs）**：

| | hold5/ramp10（默认） | hold0/ramp3 |
|---|---|---|
| fall | 46.7% | **11.7%** |
| success | 25.8% | **42.7%** |
| authorized turns | 1.55 | **2.61** |

策略本来就会"接住"，只是被静默窗挡住。修复：topdown 覆盖
`reset_action_hold_steps=0`、`reset_action_ramp_steps=3`（hold 的原始目的
"保持验证抓握"在零张力下等价于什么都不做，已无意义），按此重训/微调后
剩余 11.7% 预期继续下降；可选叠加 light-touch snap（snap 时给屈曲关节
+0.02 rad 合拢偏置，仍在 pen_deadband 内免费）。eval.py 已加
`--reset_action_hold_steps/--reset_action_ramp_steps` 覆盖参数供 A/B。

**② wrong-surface max 2020 N（红）→ 判定：reset 释放瞬间的单步求解器冲击，
诊断噪声，非行为问题。** `eval.py --wrong_surface_trace_n 20` 追踪：最大
事件发生在 episode_age=1、tilt 0.07（直立）；全部事件 near_termination
占比 **0.0**（与倒下零相关）；中期 20–40 N 轻微擦碰集中在 68 mm
（variant 直方图 25/34，最粗手柄离掌部最近）而 68 mm fall 率最低。
力幅值本就不进 reward（wrong-surface 只按二值计费）。处置：在报告口径中
把 wrong-surface 统计改为剔除 reset 后前 2 步（或 warmup 后统计），
p99 4.5 N 才是真实行为水平；不需要 reward/物理改动。

**③ damping×4 retention 79.1%（黄）→ 判定：边界差 1%，优雅降速而非失效
（fall 0%、success 100%），非阻塞。** 处置：下轮训练把
`rotation_damping_range` 上限 (0.5,2.0)→(0.5,3.0) 缩小探针的分布外距离；
零动作 no-contact 0.0315 vs 0.03 同理为 reset 瞬态边界值，warmup 后复测即可。

### 2026-07-29　ep645 残余 fall 归因（假设被数据否定）

续训（hold0/ramp3 + 宽 DR）把 overall fall 24.5%→8.6%，但 60/64 mm 仍
>5% 门线。用 `eval.py` 新增的 fall×DR-corner 交叉表（570 ep，10.7%）排查
"极端 DR 角落（尤其 13.1× 负载力矩）制造 fall"这一假设——**否定**：

| 分档 | low | mid | high |
|---|---|---|---|
| 负载力矩× | 11.1% | 12.6% | **8.4%**（≥9.3×） |
| 位移误差 mm | 18.4% | 5.2% | 8.5%（>7 mm） |

高负载∩高位移角落 fall **8.6% < 其余 11.0%**。结论：残余 fall 不是 DR 角落
伪影，**收窄负载力矩 DR 不会改善门线**，那是治不存在的病。falls 是全分布
的控制质量缺口，集中在最细的 60 mm（12.1%）。

时间分布：p50=19 步、62% 在 ≤20 步内（ep488 默认是 86%）、p90=59 步。
hold0/ramp3 确实把 reset 边缘 fall 占比压下来了，残余约 60% 落在
grasp→turn 过渡（step 15–20，动作权在 step 3 放开后手柄起转失稳）、
约 40% 是真正的 episode 中段失稳。

**plateau 信号**：ep716（多训 ~30M）fixed-64 稳定性与 authorized turns 均
不如 ep645 → 同配方续训在 8–10% 附近已收益递减，**达到 5% 需换杠杆而非加步数**。

次要疑点（单 seed，勿过度解读）：低位移误差档 fall 反而更高，提示可能是
位移**方向**而非幅值敏感——交叉表目前只用幅值。留待复核。

### 2026-07-29　方案A deploy-gap 验证:管线正确,但闭环协变量漂移致命

Stage 2 从 ep645 训 proprio adapter（HORA latent，iter100 早停，部署通道已收敛：
rel_pos MSE 3.8e-05、distance-contact 0.012；euler_z/load 高但策略不依赖）。
`--deploy_eval` 量 oracle-vs-adapter 行为差距，结果：

| 条件 | fall | net_turns | adapter latent_mae | bias峰值 |
|---|---|---|---|---|
| oracle（真 priv latent） | 7.0% | 2.31 | — | — |
| **闭环 deploy（adapter 驱动）** | **63.9%** | **0.007** | 0.555 | 0.806 |
| **开环探针（oracle 驱动，adapter 只记录）** | 5.7% | 2.35 | **0.087** | 0.028 |
| no-DR 闭环 | 62.8% | -0.009 | 0.679 | 1.539 |

**诊断（决定性）**：deploy 包 adapter 权重与验证 checkpoint 逐位相同（max diff 0）、
输入表示一致、oracle 路径正常 → 接线/打包无 bug。开环探针（轨迹保持分布内）下
adapter latent_mae 0.087、bias≈0、fall≈oracle，与训练验证吻合 → **adapter 分布内
准确**。崩溃仅在 adapter 闭环驱动时发生：初始小误差复合 → 状态 OOD → 误差放大。
纯 RMA 协变量漂移，非 wiring。no-DR 也崩，因为轨迹分布漂移与 DR 无关。

根因是两半的复合环：① adapter 纯 off-policy 训练（只见 oracle 轨迹）；②
latent-conditioned actor 只在真 latent 上训过，对 latent 误差脆弱。任一半都不足以
崩，合起来快速崩。

**方案A 的产出（达成目的）**：去力策略的部署管线在真机前就被证伪——off-policy
Stage 2 不可部署。工具沉淀：`eval.py --deploy_openloop`（开环 adapter 探针，
分离协变量漂移 vs wiring）。

**修复杠杆（两个，通常需并用）**：
1. **on-policy adapter 精化**（`--adapt_onpolicy`，从 iter100 好初值 resume）：让
   adapter 在自身诱导的轨迹上训练。仓库 help 已警告在直立任务上 destabilise
   （rollout 崩 + AdaptLoss 升），需 gentle schedule；从已收敛的 off-policy 初值
   resume 比从零更稳。
2. **latent 抗噪 actor**（Stage-1 注入 latent 噪声 / Stage-3 微调）：直接治 actor
   脆弱这一半，使其容忍 ~0.1 的 latent 误差。

**与既有结论的合流**：ep645 本非终版（fall ~10%、负载力矩 DR 待收窄）。最经济的
路径可能是一次合并的"定稿 Stage-1"：右调负载力矩 + 针对 60mm/过渡降 fall +
注入 latent 噪声，然后 Stage-2 带 on-policy。待定。

### 2026-07-29　latent 语义诊断:env_mlp 编码的一半是快变状态(违反 RMA 前提)

为找一条绕开 Stage-2 的快速上机路径，测试"冻结常数 latent"（台架是单一已知
螺丝刀+固定孔位，环境参数已知，理论上不需在线 sysID；常数无反馈环，构造上
不可能协变量漂移）。新增 `eval.py --latent_calib_output / --frozen_latent`。

**结果：证伪。** no-DR（动力学固定）下标定 latent：

| 量 | 值 |
|---|---|
| 逐通道 std | [0.32,0.20,0.64,0.54,0.38,0.02,0.69,0.47]，均值 **0.407** |
| 时间内 std（episode 内） | 均值 **0.254** |
| 环境间 std（跨 env） | 均值 **0.259** |
| 冻结 @no-DR | fall 7.3%、net_turns 0.447（oracle 0.0% / 1.408） |
| 冻结 @全DR | fall 63.2%（崩） |

**诊断**：latent 是 tanh 限幅 [-1,1]，在**动力学完全固定**时仍有 0.25 的时间内
波动 → `env_mlp` 编码的**不是**环境参数，约一半是快变状态（自转相位/角速度/
接触）。这偏离 HORA：HORA 的 priv 是慢变 extrinsics（质量/摩擦/COM），才可能从
本体历史推断；此处 priv 含物体状态（euler/angvel/relpos/contact），使 latent 变成
**状态观测器**。actor 只吃 [当前 proprio 帧(32), latent(8)]，latent 是唯一的时序/
状态通道，故 actor 重度依赖它。

**这统一解释了所有观测**：开环误差小（轨迹一致时历史与状态强相关）→ 闭环必崩
（latent 需跟踪状态，动作偏 → 状态偏 → latent 更偏，正反馈）→ 常数不可行
（状态无法用常数表示）。

**量化死刑**：最优每-episode 常数的残余误差 = 时间内 std 0.254，是 adapter 分布
内误差 0.087 的 **3 倍**；实测 0.497 误差已使 net_turns 降至 1/3。故**按几何桶
标定常数同样不可行**，无需再试。

**剩余杠杆**（按代价）：① on-policy adapter（进行中，直接治复合误差）；
② Stage-3 actor 与 adapter 联合微调（治 actor 对 latent 误差的脆弱）；
③ 重训 Stage-1 并把 `env_mlp` 输入限制为真正的慢变 extrinsics（最正统，最慢）；
④ 台架加装传感（相机/编码器）直接供给物体状态 → 近 oracle 部署，无需重训。

### 2026-07-29　台架条件复测:零训练部署方案全部证伪

新增 `eval.py --bench_dr`（保留摆放噪声的训练展布——rig 是人手上料，这部分展布
是真的；把动力学收到打印件+孔位实际范围：负载 0.5–4.0×、摩擦 0.8–1.6、阻尼
0.8–1.25），几何钉 64mm（部署包 `deployment_geometry_scale [1.0,1.0]`）。

| 配置 | fall | net_turns |
|---|---|---|
| ORACLE @标称(无摆放噪声) | 0.0% | 1.408 |
| **ORACLE @台架** | **9.1%** | 1.078 |
| ORACLE @全DR | 7.0% | 2.310 |
| FROZEN 常数 @标称 | 7.3% | 0.447 |
| **FROZEN 常数 @台架** | **89.4%** | 0.010 |

**两个结论，都是否定**：
1. **常数 latent 不可部署**。@标称的 7.3% 是无摆放噪声的假象；加入真实人手摆放
   误差后 latent 误差 0.497→0.743，策略崩到 89.4%。零训练三旋钮（常数 /
   blend / EMA）至此全部证伪 —— blend 单调恶化（λ=0.2→16.9%，λ=0.6→86.2%，
   且 λ≥0.4 的低 latent_mae 是"掉太快来不及累积误差"的选择偏差）；EMA 0.95
   仅 48.5%，因误差是系统性 bias 而非噪声。
2. **teacher 在台架条件下也不达标**（9.1% > 5%）。此前"台架≈标称 ⇒ teacher 已
   0% ⇒ 严格路线可省一半"的假设被否定：0%→9.1% 的跳变几乎全部来自摆放噪声，
   而该噪声真实、不可收窄。

**故不存在零训练上机路径。** 最低限度（定性对比）= 修部署路径，真机预期
9–12% 掉落（受 teacher 天花板限制）；严格 ≤5% = 修部署路径 + teacher 定向重训。

### 2026-07-29　根因（对照 HORA / MFR 源码）：privileged/obs 划分实现错误

用户质疑"是不是这套上机方案本身有问题"。对照两个参考实现（本地 repo）：

**HORA**（`hora/tasks/allegro_hand_hora.py:49`）—— `priv_info` = 9 维，且几乎全是
**每-episode 恒定的慢变物理参数**：
`obj_scale, obj_mass, obj_friction, obj_com`（6 维）+ `obj_position`（3 维，reset
置零且常被 mask）。env_mlp 编码的 extrinsics 是质量/摩擦/COM —— adapter 从本体
历史推断"每-episode 常数"是良定义问题，故部署稳定。策略对对象姿态基本是"盲"的。

**MFR**（`CLAUDE.md:57`）—— 螺丝刀任务 obs = `[12 手指关节 + 3 螺丝刀 euler] = 15`。
对象朝向**直接放进 obs** 喂给 actor，不经过 latent。（wrapper 里 obs≈priv_info。）

**我们**（`privileged_obs` 22 维）—— **16/22 维是快变对象状态**：euler(含自转)、
角速度、四元数、相对位置、5 维距离接触；只有 friction/geometry/load ~4 维是慢变。
env_mlp 被迫把**对象状态**压进 8 维 latent，使 latent 成为状态观测器（实测时间
方差 0.25 ≈ 总方差一半）。actor 的 proprio 输入只有 `[finger_q, cur_targets]=32`，
**不含对象状态** → 策略对姿态的唯一感知通道就是这个 latent。

**这解释了本次所有失败**：开环准（历史与状态相关）、闭环必崩（latent 需跟踪
状态、状态偏→latent 更偏，正反馈）、常数不可行（状态无法用常数表示）、on-policy
在 alpha>0.6 发散（把脆弱 actor 推崩）。全部是同一个根因的表现。

**对上机约束的含义**：台架只有手部关节反馈、无螺丝刀位姿传感。
- MFR 路线（对象 euler 进 obs）需要真机测姿态 → 不可行。
- HORA 路线（盲策略 + 慢变 extrinsics latent）只需本体 → **这才是对的方向**。

**真正的修复（Stage-1 架构 + 重训，此前列为"数天"的选项，现确认为唯一正解）**：
1. `privileged` 收缩为**纯慢变 extrinsics**（friction, load, geometry, 可加 mass/com），
   移除 euler/angvel/quat/relpos/contact 等状态通道；
2. 给 actor 一个**常驻的本体历史编码器**（temporal-conv，Stage 1 就训，而非只在
   Stage 2），让姿态感知走"本体历史→actor"这条 train/deploy 一致的通道，而不是
   走"状态→privileged→latent→adapter 重建"这条会崩的通道。
这样 adapter 只需推断慢变 extrinsics（像 HORA 一样良定义），部署不再有协变量崩溃。

**注**：ep645 家族的 obs/privileged 划分对"纯本体部署"是错的，非去力重构所致；
去力重构本身成功（距离接触通道 proprio 可预测），问题在更底层的 RMA 划分。

### 2026-07-29　HORA-faithful 小验证：成功，部署鸿沟消失

实现 `slow_extrinsics_only` 划分（新任务 `Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora`）：
actor 的 latent 编码器只吃 2 维慢变 extrinsics（load proxy + contact friction），
euler/angvel/quat/rel-pos/contact 移出 actor 通路；asymmetric critic 经 central_value
保留全 20 维状态（不部署，不影响）。网络与 deploy policy 未改。obs 契约 smoke test
确认 policy=34（32本体+2慢变）、critic=20、hist=(30,32)。

从头短训 stage-1（4096 envs，单 64mm，全 DR 除几何），~1h/ep240（环境挂起中断，
详见下）得到会转策略：oracle net_turns 1.02、fall 17%、UprightGate 0.875 —— **单帧
本体 + 慢变 latent 足以稳住并旋转**，证伪了"移除状态 latent 策略就站不住"的担心。
stage-2 纯 off-policy adapter（慢变 latent）iter-40 收敛 latent_mse 0.030。

**判定（deploy-eval，同 seed，原生 DR）**：

| 架构 | ORACLE fall/turns | ADAPTER 闭环 fall/turns | latent_mae |
|---|---|---|---|
| **HORA-faithful(新)** | 0.170 / 1.021 | **0.175 / 0.753** | 0.467 |
| 旧(状态进 latent, ep645) | 0.070 / 2.310 | 0.639 / 0.007 | 0.555 |

**结论：假设证实。** 新架构下纯 off-policy adapter 闭环 fall 与 oracle 几乎相同
（17.0%→17.5%）、turns 保留 74%，无协变量崩溃；旧架构同类 adapter 崩到 64% fall /
零 turns。关键机制证据：**两者 latent_mae 相近（0.47 vs 0.56），部署结果却天差地别**
—— 证明崩溃根因是 latent *编码什么*（快变状态会复合放大），而非误差幅值。收敛的
残余 turns 差（1.02→0.75，−26%）是正常 RMA 适配差，且此为 phase-1 轻训策略 +
iter-40 轻训 adapter，全量训会收紧。

**环境问题（需处理）**：Isaac/PhysX 在长跑 ~1h 后间歇性挂起（无 traceback / NaN /
GPU XID / 过热），进程孤儿化占显存。已见 3 次（on-policy stage-2、HORA stage-1、
及早期一次）。全量训练须每 <1h 存 checkpoint 并支持 resume，或排查根因。

**下一步（全量训练，通向可部署策略）**：
1. Stage-1 全量：全几何桶(60/64/68) + 完整 curriculum，训到成熟（fall→5%级）；
   分 <1h 段 + resume 以规避挂起。
2. Stage-2 off-policy adapter（本架构下 off-policy 已足够，无需 on-policy）。
3. deploy-eval 确认 oracle≈adapter 后，导出 deploy.pth 上真机。

### 2026-07-30　全尺度 deploy-gap 确认（ep855，主判据 PASS）

全量 Stage-1（-Hora，200M 步，ep1530）在 ~ep855/112M 即 plateau：oracle fall ~13%、
net_turns ~0.65、UprightGate ~0.79，未达成熟门槛（fall≤8%/turns≥1.8/upright≥0.9）；
phase 2/3 反而掉 upright（0.836→0.75-0.79）无 net_turns 收益。根因：单帧 actor 缺
时序本体信息 + curriculum 权重按旧架构调，非训练量问题（后 90M 步无改善）。

取 ep855（fall 最低）训 off-policy adapter（iter-40 收敛 latent_mse 0.034），四组 deploy-eval：

| 条件 | oracle fall/turns | adapter fall/turns | latent_mae |
|---|---|---|---|
| native | 0.134 / 0.648 | **0.111 / 0.669** | 0.513 |
| bench  | 0.151 / 0.583 | **0.161 / 0.598** | 0.586 |

**主判据（鸿沟闭合）决定性 PASS**：全尺度成熟策略上，adapter 闭环与 oracle **无可辨差异**
（fall ±≤2pt，turns 持平甚至略高），比小验证（turns −26%）更干净。latent_mae 0.5-0.6
仍不小却不致崩——再次证明崩溃根因是 latent 编码状态（会复合），非误差幅值。旧架构对照：
fall 7→64%、turns 2.31→0。**HORA-faithful 架构端到端证实。**

**次判据（策略质量）FAIL**：bench oracle fall 15%>10%、turns 0.58<1.5。策略欠强。

**deploy_last.pth 可作首个上机候选**（会转、fall ~15%，用于验证真机去力部署管线）。
**下一步**：实现"可选增强 2"（actor 加 3 帧/temporal-conv 历史编码器）重训 Stage-1 提质量；
gap 已在全尺度确认，可放心投入。可并行：先拿此 bundle 上机做 pipeline smoke test。

### 2026-07-30　硬件 bring-up：startup 姿态偏差排查（进行中）

首个 HORA-faithful bundle（ep855 + iter-40 adapter）上机跑 startup。四指姿态与 sim
渲染基本一致；**拇指偏差明显**（外张、不对握）。排查记录：

1. **`hand_check wiggle` 结论：拇指 yaw/roll 符号均正确，无需 flip。**
   wiggle 按余量选测试方向：yaw 从 base 0.674 朝 hi（+0.2）→ 观察到朝小指转 =
   横过手掌朝对握，符合预期；roll 从 base 1.120 朝 lo（−0.2）→ 观察到朝食指靠拢 =
   内收，等价于「值增大 = 外展离开掌平面」，亦符合预期。
2. **据此否决了「拇指 yaw/roll 转绝对角映射」的候选**（`linker_calib_deploy_thumbabs.json`）：
   符号确认后可推出该改动会把 roll 推到满量程外展（range 28→0），使外张更严重。
   保留文件仅作记录，**勿用**。
3. **未解决的表可信度问题**：`linker_sdk_map.L20_L_*` 的 docstring 自述复现 SDK 的
   `arc_to_range_left(..., hand_joint="L20")`，**但本任务按 G20 驱动**
   （`deploy_linker._validate_can_identity` 硬性要求 `--hand-joint G20`）。我们用
   `api.finger_move` 直接发 raw 0..255、**绕过 SDK 自身弧度换算**，故物理结果完全取决于
   这张表。若 SDK 存在与 `l20_l_*` 不同的 `g20_l_*` 端点，则我们 vendor 错表，
   足以解释拇指偏差（四指量程接近故影响小）。**待在有 SDK 的机器上核对**
   （本机无 SDK checkout，`test_tables_match_sdk` skip）。
4. 因 (3)，此前「sim 拇指外展 1.22 rad > 硬件 0.683 rad ⇒ 够不到」的结论**降级为待验证**
   —— 它依赖我们 URDF 弧度与 SDK 表弧度同零点同定义，而拇指链恰是 docs 标注的 provisional 部分。
5. **新增表无关判据**：`tools/analyze_deploy_record.py` 现同时输出 raw 0..255 的
   `cmd` vs `state` 对比（`COMMAND AT RAIL` / `NOT TRACKING`），不经任何弧度换算，
   可在表可信度未定的情况下判断执行器是否到位/顶限位。

**流程教训**：`--dry-run` 走 `_run_dry`，**不经过 CAN 路径的身份/安全守卫**，因此
dry-run 通过**不能**证明传输层参数（如 `--hand-joint`）正确。dry-run 只验证 bundle
加载与关节映射。

### 2026-07-30　拇指偏差排查：SDK slot 交换假设被排除

背景：机器上有 3 份 G20 驱动，拇指命令映射不一致
（`/home/user/linkerhand-ros-sdk` 与 `dex-manipulation/.vendor` 为 `[5,10,...]`；
`dex_teleop` 与 git HEAD 为 `[10,5,...]`）。`[5,10]` 是**未提交的本地修改**
（mtime 2026-07-06），其注释称原表与 `joint_state_to_cmd_state` 回读映射相反。
slot5=拇指侧展、slot10=拇指旋转，恰好是观察到偏差的两个关节，且不影响四指——
故列为头号嫌疑。

**A/B 实验（空载 startup，仅 `--sdk-root` 不同，其余完全一致）**：
- 造了结构合规的 B 变体 `/home/user/sdk_variant_B_thumb105`（整份拷贝，**仅**改拇指那一行为 `[10,5]`，diff 已验证只差该行）。
- 结果：**两者真机姿态照片无可辨差异**；settled 下发指令逐 slot 相同（差 ≤1.8 单位，为 ramp 时序噪声）——符合预期，因为交换发生在 SDK 驱动内部、不在我们的命令侧。
- **结论：SDK 拇指 slot 交换不是本次偏差的原因，该线索排除。**

**方法学记录（避免重复踩坑）**：
- `--startup-only` 的 `--record` **只写 tgt/cmd，不写 q/state**（回读列为空），
  因此该模式的采集**无法**做 cmd↔state 客观比对；需回读得走实跑或 `hand_check echo`。
- `--dry-run` 走 `_run_dry`，**不经过 CAN 路径的身份/安全守卫**，故 dry-run 通过
  **不能**证明传输层参数正确（`--hand-joint` 之误即由此漏过）。
- 顶视部署强制 `--hand-joint G20`（`_validate_can_identity`），默认亦为 G20。
- 一次只改一个变量：曾在两次 wiggle 间同时改了关节顺序/delta/sdk-root，导致
  "方向互换" 的表象无法归因（wiggle 按 `--joints` 给定顺序逐个测，应以其打印的
  关节名为准而非出场顺序）。

**重新定位问题**：空载真机姿态与 sim free-space 渲染实际相当接近（四指下卷、拇指
自中间垂下）；**显著偏差出现在放入螺丝刀之后**（真机拇指落在手柄旁而非压住手柄）。
指向**夹具/几何对位**（手柄相对手掌的位置），而非关节映射。下一步按 sim 带螺丝刀
渲染图校正夹具位置，再复评。

**待办**：`[5,10]` 修复仍未提交，有被 checkout 冲掉的风险；`dex_teleop` 那份是旧的，
若遥操作流程在用，同一只手会被两套相反映射驱动 —— 需统一并提交。

### 2026-07-30　拇指偏差解决：反向拖动读数标定

**症状**：首个 HORA bundle 上机，四指姿态与 sim 一致，拇指外张、无法压住 64mm 手柄
（指腹间距 84–85mm vs 手柄 64mm，空 ~20mm）。

**排查中被证伪的假设**（均由数据否掉，记录以免重复）：
① 三份 SDK 拷贝的拇指映射不一致 → A/B 实验姿态与指令均相同，排除；
② 修复在两次测试间生效 → 文件 mtime 2026-07-06，排除；
③ slot 11/12 是被我们发 0 钉死的真实执行器 → 探针显示不动，排除；
④ sim/硬件符号反转 → sim 实测增大 roll 同样是外展，方向一致，排除；
⑤ G20 驱动跨指串扰 → 串扰落在惰性位（探针证实 11/12 无效），排除；
⑥ 运动学不匹配 → 断电手动可摆到 sim 姿态，几何可达，排除。

**根因**：`thumb_cmc_yaw` / `thumb_cmc_roll` 仍停留在**比例映射**（其余拇指关节已是
绝对映射），而 SDK 弧度表的拇指行本就是 provisional，导致这两个关节的数值映射偏大。

**解法（方法本身可复用）**：手断电摆成 sim startup 姿态 → 上电、在任何位置指令之前
用只读工具 `tools/read_hand_state.py` 读回 20 槽编码器状态 → **一次同时测得 16 个
关节**。关键内部锚点：`thumb_cmc_pitch`（已是绝对映射）读数与 sim 仅差 **0.003 rad**，
证明测量可信、非噪声。据此单点拟合（固定 lo=0 拟合 hi）：

| 关节 | 原 hi | 新 hi | startup 指令 0..255 |
|---|---|---|---|
| thumb_cmc_roll | 1.22 | **1.479** | 28 → 68（= 手摆读数 68） |
| thumb_cmc_yaw  | 1.40 | **1.667** | 110 → 133（= 手摆读数 133） |

产物 `linker_calib_thumbfit.json`（四指指令不变）。**上机验证：拇指姿态显著改善。**
今后硬件命令用 `--calib linker_calib_thumbfit.json`。

**遗留**：`thumb_mcp` 读数偏 −0.221 未修 —— 四指 pip 呈同方向偏差，判断为末端关节
手摆普遍卷曲不足的系统性摆放误差，非标定错；若实跑发现指尖卷曲不足再单独调。
此为**单姿态单点拟合**，startup 附近准确，远离该姿态可能有残差，必要时补第二姿态做两点拟合。

**方法学**：`--dry-run` 不经 CAN 守卫，不能验证传输层参数；`--startup-only` 的
`--record` 不写 q/state；一次只改一个变量；几何量必须先确认两侧定义一致
（曾用 sim 的指尖连杆原点距离对比真机指腹距离，不可比，由此推出的结论全部作废）。

### 2026-07-30　首次真机闭环运行（管线跑通，策略锁死）

拇指标定修好后首次实跑（ep855 bundle + iter-40 adapter，`--calib linker_calib_thumbfit.json`，
10 Hz，60 tick）。

**管线验证通过**：CAN/SDK/身份校验、关节映射、adapter 推理、10 Hz 实时循环全部正常，
60 policy steps、仅 1 次 tick 超时，**关节跟踪误差 <0.01 rad**（硬件忠实执行命令）。
**去力策略可在真机闭环运行**这一核心未知已被证实。

**问题：策略 1.5 s 内饱和锁死。** |action|≈0.65（空间 ±1），
`action_delta_scale=0.05` → 每步 0.033 rad，15 拍把 middle_mcp_pitch、四指 mcp_roll、
thumb roll/pitch/mcp 全推到策略限位，之后 45 拍完全不动。**自锁机制**：目标顶限位 →
cur_targets 冻结 → proprio 冻结 → 30 帧历史退化为同一帧 → latent 冻结 → 动作冻结。

**已排除**：控制频率（sim `dt=1/60, decimation=6` = 10 Hz，与部署一致）。

**对比**：同一策略 sim bench deploy-eval 为 net_turns 0.598（会转，不冻结）→ 冻结是
**真机特有**。首要怀疑：**螺丝刀在孔位底座中的实际转动阻力超出 DR 范围**
（`screwdriver_load_torque_range` 0.5–4.0× ≈ 0.022–0.18 N·m，该上限一直是估计值、
从未实测）。转不动 → 策略加大动作 → 饱和顶死。待手感/扭矩实测确认。

**接触门禁不可用**：五指触觉里 index/middle/pinky 的 72 格总和恒为 0（无信号），
故"3 指接触"永不可满足，本次用 `--min-contact-fingers 0` 绕过。另基线在 approach
**之前**采集（拇指采到 1046），逻辑上也需改为闭合前的稳定姿态采样。二者待修。

**设计隐患**：限位自锁对任何策略都可能发生，值得在部署侧加保护（如目标贴限位持续
N 拍则告警/退出）。

### 2026-07-30　actor 3 帧堆叠 + 部署端加固（已实现，待训练验证）

**动机**：单帧 actor 无时序信息 —— 首轮全量训练 ~112M 步 plateau（NetTurns 0.35、
UprightGate 卡 0.79），真机 1.5 s 内所有关节顶限位后冻结。

**改动**：
1. **actor obs 32 → 96**（最近 3 帧堆叠，对齐 HORA 的 32×3）。新增 cfg
   `actor_frame_count`（默认 1，`-Hora` 任务设 3）；codec spec 新增同名参数并随 bundle 落盘；
   `train.py::_sync_proprio_dim` 从任务 cfg 推导 `network.proprio_dim`（共用 yaml 不改，
   避免任务间静默错切）。**部署端零改动** —— `assemble_actor_input` 本就按该值组装。
   契约实测：policy=98（96+2）、critic=20、hist=(30,32)、帧序（最旧在前、最新在后）正确。
2. **顺带修复 sim2real 时序不一致**：历史缓冲原在 `_pre_physics_step` 更新，存的是物理步进
   **前**的 q，而 `DeployPolicy.act` 存的是当前实测 q —— 实测每帧 q 差 ~0.03 rad（target 分量
   一致）。现移到 `_get_observations` 开头，最新帧 = (步进后 q, 产生它的 target)，与部署一致。
   该偏差原本就在影响 adapter，改 3 帧后会污染 actor，故必须修。
3. **删除触觉接触门禁**：本手三个 72 格触觉垫恒为 0（无信号），"N 指接触"永不可满足，
   只会拦住正常运行。保留 `_contact_snapshot` 作纯遥测。
4. **新增 rail-lock 保护**：`--rail-lock-ticks`（默认 20）/`--rail-lock-joints`（默认 4）——
   目标顶限位持续 N 拍即中止（退出码 5）。真机已观察到顶限位后 45 拍完全冻结，且
   目标冻结 → proprio 冻结 → latent 冻结 → 动作冻结，策略无法自行恢复。

**测试**：deploy 测试 26 项全过（门禁测试替换为 3 项 rail-lock 测试）；全量 215 passed /
1 failed，唯一失败为 `test_inhand_topdown_cfg` 的抓握缓存种子余量问题（本会话开始前
`assets/grasp_cache/*.npy` 即为 modified 状态，与本次改动无关）。

**注意**：obs 契约 34 → 98，**无法热启动 ep855，必须从头重训**；上一轮 200M 步结果作废。
交接指令见 `docs/handoff-full-training-prompt.md`（含与单帧基线的逐点对照表和提前中止判据）。

### 2026-07-31　3 帧堆叠全量重训：Phase 1 出现退化（进行中，待 Phase 2 判定）

3 帧 Stage-1 从头训练（`full3f_20260731`，4096 envs，seed 42）。契约确认：
日志 `proprio_dim 32 -> 96`、actor obs 98-D、critic 20-D。

**Phase 1 与单帧基线逐点对照（同一 phase、同一步数区间）**：

| Step | 单帧 NetTurns | 3帧 NetTurns | 单帧 FwdVel | 3帧 FwdVel | 3帧 Upright |
|---|---|---|---|---|---|
| ~19M | 0.291 | **0.427**(峰) | 0.181 | 0.171 | 0.839 |
| ~24M | 0.313 | 0.354 | 0.190 | 0.170 | 0.834 |
| ~29M | 0.309 | 0.265 | 0.197 | 0.176 | 0.818 |
| ~34M | 0.297 | **0.167** | 0.195 | **0.225** | 0.808 |

**3 帧峰值更高、来得更早（0.427 @18.7M vs 单帧 0.313 @24.6M），但随后单调下滑并跌破单帧**；
同期 UprightGate 0.851→0.808、而 **FwdVel 仍在上升**（0.225 > 单帧 0.195）。单帧的 FwdVel 与
NetTurns 是一起 plateau 的，无此背离。TurnRew 同期 8.7→20.1，总 reward 亦上升 —— 即
**奖励在涨、任务指标在跌**。

**假说（未证实）**：spin-fast-and-drop。**无法从日志确认** —— 训练日志不含回合长度/掉落率，
且 FwdVel 与 FwdTurns 量纲不同，不能相除推算回合长度。相关事实：**screwdriver 任务没有掉落
惩罚**（全仓搜索 `fall_penalty` 仅 inhand 任务有，且曾 -10→-25 专为治此问题）；但单帧用同一
套奖励未出现该现象，故掉落惩罚缺失不足以单独解释。

**处置**：继续训至 40M 的 Phase 2 切换（turn 权重 120→170、回合 25s→45s，正是可能改变该
平衡的杠杆）再判定。若 Phase 2 后仍下滑 → 停训跑 eval 取 `fall_rate` 直接证据（唯一能证实/
证伪该假说的手段），再决定是加掉落惩罚还是回退 3 帧改动。

**基础设施修复（本轮踩坑）**：
- 训练**不可**作为监控脚本的子进程启动 —— 监控退出会连带杀死训练（已实测）。改用
  `tools/supervise_hora_full_training.py` + `setsid` 独立会话。
- 该 supervisor 原本纯 mtime 判停滞、无启动宽限，而 4096 envs 启动需 ~5 分钟、默认阈值 120s
  → **每次启动都被杀的死循环**。已加 `_log_has_training_progress()` 前提（stage1 看 `fps step`、
  stage2 看 `AdaptLoss`）并单测；同时 `--stall-seconds` 提到 300。

### 2026-07-31　3 帧堆叠结论：Phase 1/2 显著优于单帧，Phase 3 失稳（已停训）

全量重训跑到 121.9M 步后按预设判据停训。**结论是分阶段的，不是简单的成功或失败。**

**各 phase 峰值 NetTurns（同任务、同 DR、同 seed）**：

| Phase | 3 帧 | 单帧 | |
|---|---|---|---|
| 1/3（0–40M） | **0.427** | 0.313 | 3 帧 +36%，且更早达到 |
| 2/3（40–90M） | **0.755** | 0.463 | 3 帧 **+63%** |
| 3/3（90M–） | 0.211 | 0.509 | **3 帧 −59%，失稳** |

**Phase 1 后段曾退化**（0.427@18.7M → 0.052@38.3M，同期 FwdVel 升至 0.312、TurnRew 升至 28.96
—— 转速涨、净转跌），**Phase 2 切换后自愈并反超**（8M 步内回到 0.755）。推测与 Phase 2 把回合
从 25s 拉到 45s、放大掉落代价有关（未取 fall_rate 直接证据）。

**Phase 3 则未能自愈**：切换后 32M 步（Phase 2 自愈窗口的 4 倍）持续恶化 ——
NetTurns 0.138→0.097、**OscRatio 0.250→0.441**（>0.15 即差）、reward +4989→−2158。
UprightGate 反而维持 0.78–0.85，说明**不是掉落，是原地来回晃、转不出净进展**。
对照：单帧进 Phase 3 是**上升**的（0.346→0.480），两者反应相反。

**判读**：3 帧堆叠给 actor 的时序信息确实有效（Phase 1/2 全面领先），但在 Phase 3 的高 turn
权重（200）下与之交互失稳。**正确的收尾是用 Phase 2 权重定稿，而不是回退 3 帧改动。**

**处置**：停训，从 Phase 2 区间（40–90M）取候选 checkpoint 做 oracle eval 择优，进 Stage-2。
候选 ep360/495/540/585/675（reward 峰值在 ep540 = 70.8M / rew 7327，而训练态 NetTurns 峰值
在 ~48M —— 两者不一致，故以 eval 实测为准，不信训练日志的噪声指标）。

**遗留（若后续要再训一版）**：Phase 3 的 `reward_turn_weight=200` 对 3 帧过激；可考虑降到
Phase 2 的 170、或延长 Phase 3 的回合长度、或为 3 帧单独调 phase 表。本轮不做，先用 Phase 2
最优 checkpoint 走完 Stage-2 拿到可部署产物。

### 2026-07-31　严重 bug：帧堆叠 actor 在 oracle eval 中读到冻结观测（已修）

**症状**：3 帧 checkpoint 的 oracle eval 全部异常 —— fall 0.96–0.99、授权净转恒为 +0.000、
物理净转为负；而同期训练日志显示 UprightGate ~0.83、NetTurns 0.755。

**根因（自引入）**：`eval.py` 只在 `--deploy_eval` 时设 `env_cfg.asymmetric_obs = True`
（该行注释原本就写明"缓冲仅在 asymmetric_obs 打开时更新"）。而 3 帧改动把 actor 的 proprio
改为 `assemble_actor_input(self._prop_hist_buf)`，该缓冲的更新恰恰被 `if self.cfg.asymmetric_obs`
守着。于是普通 oracle eval 中缓冲自 reset 后再未更新，**actor 一直看着 3 帧完全相同的冻结初始
观测** —— 表现为"策略极差"，实为观测未更新。改动前 actor 用 `encode_frame()` 现算，不依赖该
缓冲，故此前无碍。

**修复**：缓冲更新条件改为 `asymmetric_obs or actor_frame_count > 1`（在 env 层修，覆盖所有
消费方，而非只补 eval）。

**影响范围**：
- **训练结论仍然有效**（训练恒设 `asymmetric_obs=True`，缓冲正常）：3 帧在 Phase 1/2 明显优于
  单帧（峰值 0.427 vs 0.313、0.755 vs 0.463）、Phase 3 失稳，这些基于训练日志的结论不受影响。
- **本日所有 3 帧 eval 结果作废**（oracle / eval_phase 1 与 3 / 确定性与随机、共 12 次），
  需在修复后重跑。

**同期发现的其它错误（均已修，记录以免重犯）**：
1. `_sync_proprio_dim` 只加进 `train.py`，漏了 `eval.py` → 3 帧 checkpoint 在 eval 中因
   `proprio_dim` 仍为 32 而形状不匹配、加载失败（actor_mlp 104 vs 40、env_mlp 2 vs 66）。已补。
2. 用默认 `--eval_phase final`（Phase 3）评估专为规避 Phase 3 而挑的 Phase 2 checkpoint —— 自相矛盾。
3. **epoch→步数换算忽略 resume 会重置 epoch 计数**：第三个 run 目录自 13,762,560 步 resume，
   故真实步数 = 13,762,560 + ep×131072。据此，先前选的 5 个候选无一位于训练峰值附近，
   其中两个实际已进 Phase 3。
4. **度量错配**：训练日志 `NetTurns` = `eval_net_turns` = **授权**净转；eval 的
   `net_turns_mean` = `eval_raw_net_turns` = **物理**净转；与训练可比的是 eval 的
   `authorized_net_turns_mean`。此前把训练 0.755 与 eval 0.648 直接对比是错的。

**教训**：契约类改动必须在**所有消费方**（train / eval / deploy）各跑一次；本轮只验证了
train 与 deploy 的 smoke test 就宣布完成，eval 这条路径的两个 bug 都是因此漏掉的。

### 2026-07-31　3 帧候选评估（修复冻结观测 bug 后重做）

修复后重跑，训练日志与 eval 终于吻合（ep270 授权净转 eval 0.848 vs 训练日志 0.755）。

**Phase 2 区间（eval_phase 1）高度一致**：

| ep | 步数 | fall | 授权净转 |
|---|---|---|---|
| 210 | 41.3M | 0.080 | 0.913 |
| 270 | 49.2M | 0.089 | 0.848 |
| **330** | **57.0M** | **0.070** | **0.975** |
| 420 | 68.8M | 0.108 | 0.911 |
| 510 | 80.6M | 0.090 | 0.917 |

**Phase 3 区间（eval_phase 2）单调恶化**，与训练日志独立互证：

| ep | 步数 | fall | 授权净转 |
|---|---|---|---|
| 600 | 92.4M | 0.178 | 0.726 |
| 690 | 104.2M | 0.189 | 0.465 |
| 780 | 116.0M | 0.376 | 0.215 |

**最优候选 ep330 三口径**：Phase2 fall 0.070/授权 0.975；final phase fall 0.241/物理 0.794；
**bench fall 0.216/物理 0.858**。

**与单帧基线同口径（bench）**：单帧 ep855 fall 0.151/物理 0.583 vs 3帧 ep330 fall 0.216/物理 0.858
—— **转动 +47%，掉落 +43%，非干净胜利**。根因：ep330 只训到 Phase 2，一出该口径即退化
（Phase2 fall 7% → bench 22%）。

**两者均不满足验收线**（bench oracle fall ≤10%、net_turns ≥1.5）。

**建议（证据充分、非猜测）**：定向重训，把 Phase 3 的 `reward_turn_weight` 由 200 降至 170
（与 Phase 2 一致）或延长过渡，使策略能走完 curriculum 而不失稳。三条互证证据：训练日志
Phase 3 恶化、eval 在 Phase 3 单调下滑、Phase 2 内部 5 候选高度一致。
