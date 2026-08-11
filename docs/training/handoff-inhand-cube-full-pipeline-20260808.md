# Linker G20 64 mm 自由立方体：Stage‑1 + Stage‑2 + 实机部署 handoff

状态：**两个旧 300 M Stage‑1 均已完成但统一 oracle checkpoint gate 失败；2026-08-09 获授权的
bottom-up 奖励/控制修复 pilots 也未产生合格 teacher。用户随后明确批准按本机 HORA 实现修复
teacher observation；9-D HORA parity、双任务 12,288-env asymmetric-critic smoke 和 40 M
bottom-up bounded learning pilot 均已完成测试，但 learning oracle 仍失败**。最新 9-D pilot 只有
`0.0367 turns/20 s`、fall `12.01%`、palm `0.0547%`，因此没有证据恢复 300 M。当前没有合格
Stage‑2 teacher；top-down 9-D 全量、Stage‑2、导出均未启动，GPU 已释放。本文不是训练完成
报告，也不表示已有可评估或可上真机的 policy。

任务 ID 保留兼容名称：

- `Isaac-LinkerL20-Inhand-Rotation`：bottom-up，掌面朝上、手指方向上倾 45°；
- `Isaac-LinkerL20-Inhand-Rotation-Topdown`：top-down，向下倾斜 45°；
- 对应缓存生成任务为 `Isaac-LinkerL20-Inhand-GraspGen` 和
  `Isaac-LinkerL20-Inhand-GraspGen-Topdown`。

虽然 ID 中仍写 L20，资产、语义映射、标定和部署约束都对应当前左手 G20：
`linker-g20-left-semantic-16-v1`、手腕外部固定、操作者装入物体。

## 已确认的任务边界

- 物体：名义边长 64 mm 的 PLA 立方体；训练几何为 60/64/68 mm 三档。
- 名义质量 100 g；DR 覆盖 30–200 g。实物称重后必须用实测质量重跑 bench 验收。
- COM 随机化 ±8 mm，摩擦 0.4–2.5，PD 增益 0.967–1.033。
- 目标轴固定为重力方向 world `-Z`。
- 不允许物体由物理掌面 `hand_base_link` 承载；手指侧面、近节和中节接触可以接受。策略验收
  将掌面物体接触力 `>0.05 N` 超过 1% 采样判为失败。生产缓存生成仍采用更严格的全非指尖
  剔除，以给训练 reset 留余量；该超额约束不能被误写成实机拒绝条件。
- 不添加腕部动作；实机由外部机构固定腕位，操作者按部署包姿态装入立方体。
- 不改变原自由物体任务的奖励结构。握姿、缓存、观测拆分、指标、验收和部署保护可以改；
  若需要新奖励或新策略架构，必须先停下另行批准。
- 2026-08-09 用户已授权为达到全量训练目标进行 bottom-up 奖励/控制修复，并在核对本机
  `/home/fresh/Workspace/allegro_inhand_rotation` 后明确批准把 teacher tail 从 slow6 修为 HORA
  原实现的 `object local xyz + slow6 = 9-D`。这不授权改成 full19、加入姿态/速度、改 actor
  trunk，或自行更换策略架构。

## 网络与运行时契约

```text
policy rate       = 20 Hz (dt 1/120 × decimation 6)
action            = 16 个当前 G20 独立关节的累计位置增量
actor raw input   = proprio 96 + HORA teacher tail 9 = 105
proprio 96        = 最近 3 帧 × [scaled q16, effective target16]
teacher tail 9    = object local position xyz + geometry scale, mass, friction, COM xyz
teacher latent    = 8
critic input      = 19（object xyz/quat/linear vel/angular vel + 6 slow extrinsics）
Stage-2 history   = 30 × 32（1.5 s）
deployment input  = 只有 q16 与 acknowledged effective target16
```

Stage‑1 使用 asymmetric central critic。Stage‑2 只回归 actor 的 8 维 HORA teacher latent；
teacher 包含物体局部位置但不包含 quaternion、linear velocity 或 angular velocity，这三类快速量
只给 critic。Stage‑2 默认 off-policy，
不要加 `--adapt_onpolicy`。

Stage‑2 对这两个任务 fail-closed：若不能组装带确定性装载姿态、缓存 provenance、actor、
normalizer 和 adapter 的 `deploy.pth`，训练直接失败，不接受 adapter-only 结果。

## 姿态选择证据

bottom-up 预先固定判据：strict `+Z` 只有在 timeout survival 比 45° 倾斜提高超过 5 个百分点
时才替换；搜索阶段使用了比部署更严格的全非指尖过滤。512 env × 300 step 结果：

| bottom-up | timeout | timeout survival | stable acceptance | accepted tips-only |
|---|---:|---:|---:|---:|
| tilted 45° | 119 | 0.404% | 4.331% | 100% |
| strict +Z | 0 | 0% | 0% | 无 accepted row |

因此保留 45°。这里的低总体 survival 是宽噪声搜索的采样效率，不是生产 reset 质量；v2
生产缓存只收集完整释放后坚持 20 s 的行，并要求每行通过四次 full-DR、零速度重建回放。证据：
`artifacts/inhand_refactor_20260808/orientation_selection.json`。

top-down 先测试 strict `-Z`。静态网格、宽搜和局部物理搜索中，64 mm 立方体释放后
`0` timeout；手指无法在禁止掌面承载时形成下侧支撑。按已授权的“倾斜或严格 ±Z，效果好的
胜出”，改为向下倾斜 45°。旧 2.5 s 候选没有直接用于生产；重新进行 20 s 搜索后，candidate
205/rank 0 和 candidate 7/rank 1 均为 12/12 timeout、0 termination。生产采用 candidate 205，
名义 64 mm、±0.02 rad 噪声 pilot 的四重 full-DR 认证为 229/256（89.45%）。证据：
`artifacts/inhand_refactor_20260808/topdown_longhold_search_round1.json`。

生产 top-down root/物体 seed：

```text
hand pos  = (0, 0, 0.725)
hand quat = (0.27059805, 0.65328148, 0.65328148, -0.27059805)  # wxyz
cube pos  = (-0.00906503, -0.08247901, 0.56956439)
fall z    = 0.542
```

训练和部署的实际初始姿态不是单个静态 seed，而是版本化缓存中的 q16 + squeeze target16；
部署包从名义 64 mm 缓存选稳健中心行，先走 collision-safe measured q，再缓慢闭合到 squeeze
target。

## 缓存生成与前置关卡

Python 以下以本机为准：

```bash
PY=/home/fresh/miniconda3/envs/env_isaac/bin/python
```

不要同时运行两个 Isaac 任务。生成器输出的是未认证 source bank；每个 source row 必须再由
`tools/certify_inhand_grasp_cache.py` 在主任务中执行四次 20 s full-DR 回放。示例（top-down
64 mm；另外两档和 bottom-up 同样执行）：

```bash
SRC=/tmp/dex_forge_inhand_topdown_v2_source
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u tools/gen_inhand_grasp_cache.py \
  --task Isaac-LinkerL20-Inhand-GraspGen-Topdown \
  --shape cuboid --scale 1.0 --num_envs 8192 --num_states 22000 \
  --seed 20260808 --out "$SRC" --max_steps 4500 --headless --device cuda:0

env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u tools/certify_inhand_grasp_cache.py \
  --task Isaac-LinkerL20-Inhand-Rotation-Topdown \
  --cache_dir "$SRC" --scale 1.0 --num_envs 8192 --replays 4 \
  --max_rows 22000 --target_rows 12500 \
  --output_cache assets/grasp_cache/linker_l20_topdown_cube64_v2_grasp_cub0_s1.npy \
  --json_output artifacts/inhand_refactor_20260808/cache_cert/topdown_s1.json \
  --seed 20260808 --headless --device cuda:0

"$PY" tools/validate_inhand_grasp_cache.py
```

生产银行：

- `assets/grasp_cache/linker_l20_bottomup_cube64_v2_*`
- `assets/grasp_cache/linker_l20_topdown_cube64_v2_*`

每个 manifest 绑定当前 G20 model/semantic/URDF digest、64 mm cube 三档、姿态 namespace、
39 列 row layout、四重 20 s replay certification 和每个 `.npy` 的 SHA‑256。改变 root pose、
canonical seed、URDF、语义映射、
几何档或 row layout 时必须换 cache namespace 并重生成；不能复用看似数值合法的旧缓存。

本轮每档正式保留 12,500 行；认证产率如下（产率是 source 效率，不是训练指标）：

| orientation | 60 mm | 64 mm | 68 mm |
|---|---:|---:|---:|
| bottom-up | 13,344/22,000 (60.65%) | 16,041/28,000 (57.29%) | 12,767/22,528 (56.67%) |
| top-down | 12,570/20,480 (61.38%) | 12,701/14,336 (88.60%) | 13,422/14,336 (93.62%) |

静态验证通过后，必须在主训练环境而不是 GraspGen 环境跑 full-DR zero-action reset：

```bash
mkdir -p artifacts/inhand_refactor_20260808/reset_probe
for TASK in Isaac-LinkerL20-Inhand-Rotation Isaac-LinkerL20-Inhand-Rotation-Topdown; do
  NAME=${TASK##*-}
  env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u tools/probe_inhand_resets.py \
    --task "$TASK" --num_envs 512 --steps 900 --headless --device cuda:0 \
    --json_output "artifacts/inhand_refactor_20260808/reset_probe/${NAME}_palm_gate.json"
done
```

进入训练前预先固定的 reset gate：

- doomed（10 step 内掉落）≤10%；
- zero-action timeout survival ≥50%；
- 掌面 `hand_base_link` 承载力 `>0.05 N` 的 step-env 比例 ≤1%；
- 报告必须有完整 episode，且三档 manifest/digest 均已验证。

任何一项不满足就停下报告并回到握姿/缓存；不要靠改奖励掩盖坏 reset。

2026‑08‑08 的 512 env × 900 step full-DR 结果全部通过：

| task | timeout survival | doomed ≤10 | palm >0.05 N | all non-tip diagnostic |
|---|---:|---:|---:|---:|
| bottom-up | 98.4% | 1.0% | 0.178% | 4.698% |
| top-down | 94.7% | 2.7% | 0.437% | 1.652% |

最后一列只用于观察允许的手指侧面接触，不是拒绝条件。正式报告是
`reset_probe/bottomup_v2_palm_gate.json` 和 `reset_probe/topdown_v2_palm_gate.json`。

## Stage‑1 smoke

每个任务先独立跑最小 smoke。日志必须显示 actor 102-D、critic 19-D、无未知参数 warning、
无 NaN/shape error，并产出 checkpoint：

```bash
TASK=Isaac-LinkerL20-Inhand-Rotation  # 然后换 Topdown 再跑
OUT="runs/$TASK/smoke_stage1_ready_20260808"
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u train.py --task "$TASK" --stage 1 --num_envs 12288 \
  --max_epochs 2 --save_interval_steps 98304 --seed 20260808 \
  --output "$OUT" --headless
```

一次 checkpoint 审计发现，旧 YAML 把 `central_value_config` 放在 RL-Games 不读取的位置；旧
smoke 虽打印 critic 19-D，checkpoint 却没有 `assymetric_vf_nets`，因此全部作废。加载器现已将
free-object 配置迁到 `params.config.central_value_config`，并补齐 central trainer 必需的
`normalize_input`/`clip_value`。

修复后的两个目标规模 smoke 均已通过。两次日志都显示 actor raw obs 102-D、central input 19-D、
`[train] Central critic: enabled via params.config.central_value_config`、
`Adding Central Value Network`、`build mlp: 104` 和 `build mlp: 19`；均完成 2 epoch、每 epoch
98,304 steps，且未出现 OOM、NaN、shape error 或未知参数：

- bottom-up phase checkpoint：
  `runs/Isaac-LinkerL20-Inhand-Rotation/smoke_stage1_ready_cvfix2_20260808/linker_l20_inhand_rotation_08-20-18-32/nn/linker_l20_inhand_rotation_phase1.pth`
- top-down phase checkpoint：
  `runs/Isaac-LinkerL20-Inhand-Rotation-Topdown/smoke_stage1_ready_cvfix2_20260808/linker_l20_inhand_rotation_08-20-29-40/nn/linker_l20_inhand_rotation_phase1.pth`

两个 phase checkpoint 均已用 CPU `torch.load` 审计：顶层包含 `assymetric_vf_nets` 和
`central_val_stats`，`assymetric_vf_nets` 各有 14 个 tensor，central running mean shape 为
`(19,)`，central 第一层 MLP weight shape 为 `(512, 19)`，actor raw normalizer shape 为
`(102,)`。完整测试结果为 `251 passed, 2 skipped, 1 xfailed`。因此双任务可进入经用户另行
授权的全量 Stage‑1；这些 2-epoch smoke checkpoint 只证明 pipeline 可启动，不是可评估或
可部署 policy，也不能作为 Stage‑2 teacher。

正式 64 mm cache reset 的六视角渲染位于
`artifacts/inhand_refactor_20260808/renders_stage1_ready/`。零动作沉降 30 步时 bottom-up 平均下降
2.83 mm，top-down 平均下降 0.61 mm，均无 reset；多视角目视确认立方体未落在掌面。

## 全量 Stage‑1

2026‑08‑08 用户已授权两个方向各 300 M steps 的全量 Stage‑1，按本节契约串行运行。当前状态：

- bottom-up 已以 0 退出完成 3,052 epochs / 300,023,808 steps：
  `runs/Isaac-LinkerL20-Inhand-Rotation/full_stage1_20260808/`；训练日志为
  `inhand_bottomup_stage1.log`；
- top-down 也已以 0 退出完成 3,052 epochs / 300,023,808 steps：
  `runs/Isaac-LinkerL20-Inhand-Rotation-Topdown/full_stage1_20260808/`；训练日志为
  `inhand_topdown_stage1.log`；
- 两个 supervisor 均为 `hang_count=0`，central critic 接线和 checkpoint 保存正常；
- Stage‑2 虽获条件授权，但前提是先从统一 oracle gate 选出合格 teacher。当前没有候选满足
  该前提，因此禁止启动 Stage‑2。

### Stage‑1 oracle checkpoint gate：FAILED

评估协议固定为 deterministic oracle actor、256 env、seed 0、名义 64 mm cube、full DR、
800 policy steps（每 episode 400 steps / 20 s）。原始 JSON 与日志位于
`artifacts/inhand_refactor_20260808/stage1_oracle_candidates/`。

| task | checkpoint epoch | NetTurns / 20 s | fall | palm >0.05 N | success ≥1 turn |
|---|---:|---:|---:|---:|---:|
| bottom-up | 2400 | 0.185 | 99.974% | 3.724% | 0% |
| bottom-up | 2600 | 0.190 | 99.972% | 3.092% | 0% |
| bottom-up | 2800 | 0.187 | 100.000% | 3.381% | 0% |
| bottom-up | 3000 | 0.188 | 100.000% | 3.176% | 0% |
| bottom-up | 3052 | 0.188 | 100.000% | 2.895% | 0% |
| top-down | 2400 | 0.025 | 31.241% | 1.332% | 0% |
| top-down | 2600 | 0.031 | 27.316% | 1.588% | 0% |
| top-down | 2800 | 0.029 | 25.714% | 2.843% | 0% |
| top-down | 3000 | 0.030 | 24.667% | 1.651% | 0% |
| top-down | 3052 | 0.025 | 27.480% | 1.244% | 0% |

Promotion 要求 oracle 至少 `1.5 turns / 20 s`，且掌面物体力 `>0.05 N` 的比例 `≤1%`；
上表十个候选全部失败。bottom-up 是明确的 spin-and-drop；top-down 保持较久但没有学到有效旋转。
对照同一正式 cache 的零动作 reset gate（bottom-up timeout 98.4%、top-down 94.7%），这些掉落和
no-palm 退化是 actor 行为造成的，不是 reset/cache 自身不稳定。
Stage‑2 冻结 actor、只回归 teacher latent，不能修复这些 Stage‑1 能力缺失。根据“不改奖励、不
自行选择新策略架构”和“发现明显矛盾就停止报告”的边界，必须在此停下 GPU 工作，不能用无效
checkpoint 生成或宣传 deploy-ready policy。

只读根因证据：`_get_rewards()` 从 `raw_rotate` 直接计算 `rotate_reward`；它虽然计算并记录
`contact_gate`，却没有用该 gate 约束旋转收益。bottom-up 五次评估的 contact gate 仅
32.96–37.84%，且无接触时 forward velocity 高于接触时，符合 coasting/spin-and-drop。当前配置
又从 step 0 使用单一 `reward_turn_weight=1.0` phase；同一配置中的历史注释已明确规定，如果
RotateReward 上升而 EpLen 崩塌的签名重现，应恢复约 8 M steps 的短 hold-first phase。
bottom-up 的 fall-age p50 为 54–58 steps；在 PPO `gamma=0.995` 下，终止时一次性 `-25` 的惩罚
折算到 episode 开始约为 `-19`，不足以稳定压过此前累计的未门控旋转收益。top-down 则相反：
contact gate 为 98.04–99.30%，但 forward velocity 只有约 0.018–0.021 rad/s，落入稳定持有但
几乎不旋转的局部最优。下一轮若获授权，必须先用小预算 A/B 验证课程/奖励修复，不能直接重跑
另一个 600 M campaign，也不能靠 Stage‑2 掩盖 actor 失败。

以下命令保留为单任务启动/恢复参考。

RTX 5090 上先用 12,288 env；用户已确认 8,192 过于保守。free-cube PPO horizon 是 8，
supervisor 会按任务自动解析，12,288 env 时每 epoch 为 98,304 steps，300 M 目标为
3,052 epochs / 300,023,808 实际 steps。

两个方向必须串行训练，不能共享 GPU：

```bash
TASK=Isaac-LinkerL20-Inhand-Rotation
OUT="runs/$TASK/full_stage1_20260808"
setsid nohup "$PY" tools/supervise_hora_full_training.py \
  --stage 1 --task "$TASK" --output "$OUT" --python "$PY" \
  --log "$(pwd)/inhand_bottomup_stage1.log" \
  --num-envs 12288 --seed 20260808 --target-global-steps 300000000 \
  --stall-seconds 300 > inhand_bottomup_supervisor.log 2>&1 < /dev/null &

# bottom-up 完成并释放 GPU 后，再将 TASK/OUT/log 改为 Topdown 启动。
```

启动即验：

- 命令行确实有 `--task`、`--num_envs 12288`、`--headless`；
- 日志无 `[train] WARNING: these arguments were not recognised`；
- `[Stage 1] Actor obs : 102-D ... Critic obs: 19-D`；
- PPO 开始输出 epoch/fps，supervisor state 的 `steps_per_epoch=98304`、`task` 为当前任务。

每 15–30 分钟记录：NetTurns、FwdVel、RevVel、OscRatio、FallFrac、EpLen、HoldFrac、
RotateReward、TOTAL REWARD。红旗为：

- RotateReward 上升但 EpLen/HoldFrac 崩塌：spin-and-drop；
- RevVel 接近或超过 FwdVel，OscRatio 持续上升：来回振荡；
- NetTurns 长期不升或为负：方向/学习失败；
- NaN、回报突变、checkpoint 评估中掌面承载上升。

不要仅用训练日志选最终 checkpoint。至少选择 4 个晚期 checkpoint，在相同 seed、DR 和
20 s episode 协议下做 oracle eval；保留 plateau 前的 checkpoint，防止末段退化。300 M 是
首轮预算，不是“跑到即成功”。

## Stage‑2 smoke 与全量训练

先对 Stage‑1 smoke checkpoint 验证一轮 adapter 和自包含 bundle：

```bash
S1=/abs/path/to/stage1.pth
OUT=/abs/path/to/the/same/task/output
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u train.py --task "$TASK" --stage 2 --checkpoint "$S1" \
  --num_envs 256 --adapt_iters 2 --adapt_rollout_steps 16 \
  --adapt_save_interval 1 --seed 20260808 --output "$OUT" --headless
test -s "$OUT/stage2_nn/deploy.pth"
```

全量 Stage‑2 用最终候选 Stage‑1 checkpoint：

```bash
setsid nohup "$PY" tools/supervise_hora_full_training.py \
  --stage 2 --task "$TASK" --output "$OUT" --python "$PY" \
  --checkpoint "$S1" --log "$(pwd)/inhand_stage2.log" \
  --num-envs 2048 --adapt-iters 150 --adapt-save-interval 20 \
  --seed 20260808 --stall-seconds 300 \
  > inhand_stage2_supervisor.log 2>&1 < /dev/null &
```

Stage‑2 才是关卡。`adapter_target_mse` 两个 checkpoint 连续 ≤0.02 或明显 plateau 才停止；
Stage‑1 数字好看不能宣布成功。若 closed-loop adapter 崩而 open-loop 正常，这是协变量漂移，
先报告，不自行改架构或奖励。

## 四路闭环验收

每个方向单独运行：

```bash
AD="$OUT/stage2_nn/deploy.pth"
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u tools/evaluate_inhand_stage2_acceptance.py \
  --task "$TASK" --checkpoint "$S1" --adapter_checkpoint "$AD" \
  --output "artifacts/${TASK}_acceptance_20260808" \
  --python "$PY" --num_envs 256 --seed 0
```

它串行运行 native/bench × oracle/adapter，固定 64 mm cube，并检查：

- native 与 bench：adapter fall ≤40%，且 ≤ oracle +5 percentage points；
- native 与 bench：adapter NetTurns ≥ oracle 的 70%；
- native 与 bench：oracle 和 adapter 都 ≥1.5 turns / 20 s；
- bench oracle 保留 native turns 的 ≥90%，fall 增量 ≤10 points；
- 四路掌面物体接触力 `>0.05 N` 的比例均 ≤1%。

若实物已称重，加 `--fixed_object_mass_g <grams>`；最终实机候选的 acceptance 必须包含这个
实测值。adapter closed-loop 失败时，额外用 `eval.py --deploy_openloop` 定位，但不能用
open-loop 结果替代 promotion gate。

## 不可变部署包与实机顺序

先提交训练代码、缓存和 manifest，确保 tracked tree clean。导出器会拒绝 dirty training code：

```bash
"$PY" tools/export_inhand_policy_package.py \
  --deploy "$AD" \
  --acceptance "artifacts/${TASK}_acceptance_20260808/acceptance.json" \
  --calibration assets/calibrations/linker_g20_left_lht20_010_415_v1.json \
  --output "artifacts/policy_packages/${TASK}_20260808"
```

不可变包保存 actor/adapter safetensors、20 Hz codec、装载姿态、缓存/URDF/标定/Stage‑1/
Stage‑2/验收 digest，promotion 状态为 `sim-qualified-hardware-commissioning-required`。
当前 live controller 仍直接加载同一已验收的 `deploy.pth`。

实机必须按以下顺序，且两个方向分别 commissioning：

1. `--dry-run --max-ticks 100`：离线验证 20 Hz、30 帧 fresh-history 和 rail-lock；
2. CAN `--no-send --max-ticks 100 --record ...`：读取真实 G20 状态，不发命令；
3. 固定腕部，操作者把 64 mm cube 对准包内 task frame；运行 `--startup-only`，在
   collision-safe approach 到 contact-home 的慢闭合期间托住物体；检查 fault 和 tactile telemetry；
4. 人工确认物体未由掌面承载；允许手指侧面接触；
5. 首次 live 加 `--task-frame-confirmed --max-ticks 20 --record ...`，现场 E-stop；
6. 逐步扩到 100/300 ticks。rail-lock、stale state、fault、方向错误、掉落或掌面承载立即停；
7. ROS live 对固定腕任务 fail-closed；参考路径只允许 CAN。

示例：

```bash
"$PY" -m screwdriver_rl.deploy.deploy_linker --checkpoint "$AD" \
  --dry-run --max-ticks 100

"$PY" -m screwdriver_rl.deploy.deploy_linker --checkpoint "$AD" \
  --transport can --no-send --max-ticks 100 --record no_send.csv

"$PY" -m screwdriver_rl.deploy.deploy_linker --checkpoint "$AD" \
  --transport can --startup-only --record startup.csv

"$PY" -m screwdriver_rl.deploy.deploy_linker --checkpoint "$AD" \
  --transport can --task-frame-confirmed --max-ticks 20 --record first_live.csv
```

## 2026-08-09 bottom-up 修复结果与 HORA 9-D observation 修复

用户授权启动全量训练后，先按统一 Stage-1 oracle gate 审核已有 300 M checkpoint。由于没有
合格 teacher，未把失败模型送入 Stage-2；随后只对 bottom-up 做了串行、止损式修复 pilots。
所有 oracle 均为 deterministic actor、256 env、seed 0、固定 64 mm、full DR、800 policy
steps。关键结果：

| candidate | NetTurns / 20 s | fall | palm >0.05 N | 结论 |
|---|---:|---:|---:|---|
| 300 M stable phase5（原控制） | 0.162 | 3.88% | 0.488% | 稳定但静止 |
| stability gate + low explore，epoch 3200 | 0.152 | 7.16% | 0.401% | actor mean 未吸收 stochastic motion |
| direction drive=2，epoch 3200 | 0.15 | 9.4% | 1.1% | FAIL |
| direction drive=100 / sigma=-3 | 0.15 | 54.2% | 2.4% | 过强 shaping，FAIL |
| home-window 0.35，未重训 | 0.04 | 10.2% | 1.8% | 证明旧 0.15 主要是单向 target 漂移 |
| window + bound + drive，15 M | 0.11 | 10.1% | 2.2% | 未学出循环 |
| 同上，允许无惩罚回程，再 15 M | 0.11 | 5.4% | 1.5% | 最新稳定候选，仍差 13.6x |

最新报告：

- `artifacts/inhand_refactor_20260808/stage1_repair_20260809/bottomup/window035_bound100_drive50_returnfree_final.json`
- checkpoint：
  `runs/Isaac-LinkerL20-Inhand-Rotation/window035_bound100_drive50_returnfree_cont15m_20260809/linker_l20_inhand_rotation_09-05-44-46/nn/linker_l20_inhand_rotation_phase5.pth`

诊断结论：原累积 target 会漂到 URDF 极限，产生一次性约 0.15 turn 后停车；home-relative window
消除了该假进展，但 3 帧 proprio（150 ms）+ 6 个 episode-constant slow extrinsics 的 feed-forward
actor 没有学出持续的 drive/release/return gait。放大奖励、降低探索噪声、加入稳定门控、边界代价
以及移除回程奖励冲突均未解决。继续按相同 102-D contract 增加训练步数没有证据基础。

本机 `/home/fresh/Workspace/allegro_inhand_rotation`（HEAD
`29f75fa7024577e93e5f6fd70a81d7ef9e9cffa6`）审计确认，原 HORA teacher priv 不是 slow6，
也不是 full19，而是 `object position xyz + scale + mass + friction + COM xyz = 9-D`。物体
quaternion、linear velocity 和 angular velocity 只用于仿真奖励/诊断，不进入 actor。

用户已明确批准按这一实现修复。当前代码将 Stage‑1 raw policy observation 从
`proprio96 + slow6 = 102` 改为 `proprio96 + HORA tail9 = 105`，`env_mlp` 输入从 6 改为 9；
actor trunk 仍只消费 `96 + latent8 = 104`，central critic 仍为 19-D。现有 Stage‑2 trainer
继续通过 live `env_mlp` 生成 8-D teacher target，adapter 输出仍为 8-D；最终 deploy actor
仍是 `proprio96 + predicted latent8`，无需增加实机物体传感器。旧 env_mlp(6) checkpoint
不兼容，必须从头训练并由 checkpoint fail-fast 拒绝误载。

结构与静态验证已通过：targeted tests 为 `47 passed`；全量为
`252 passed, 2 skipped, 1 xfailed`；`git diff --check` 通过。两个方向均完成 12,288 env、2 epoch
的真正 asymmetric-critic smoke：

- bottom-up：
  `runs/Isaac-LinkerL20-Inhand-Rotation/smoke_stage1_hora9d_cvfix_20260809/linker_l20_inhand_rotation_09-11-43-15/nn/linker_l20_inhand_rotation_phase1.pth`
- top-down：
  `runs/Isaac-LinkerL20-Inhand-Rotation-Topdown/smoke_stage1_hora9d_20260809/linker_l20_inhand_rotation_09-11-47-46/nn/linker_l20_inhand_rotation_phase1.pth`

两者 runtime 均打印 actor `105-D`、critic `19-D`、`Adding Central Value Network`、actor trunk
`build mlp: 104`、central `build mlp: 19`。CPU 审计均为 actor normalizer `(105,)`、env_mlp
首层 `(256, 9)`、central normalizer `(19,)`、central 首层 `(512, 19)`、14 个
`assymetric_vf_nets` tensor。

随后从零运行了 12,288 env、407 epoch、`40,009,728` global steps 的 bottom-up 有界学习
测试，使用已审计的 home-relative `joint_motion_range=0.35`、target-bound `100` 和 drive `50`：

- checkpoint：
  `runs/Isaac-LinkerL20-Inhand-Rotation/hora9d_window035_bound100_drive50_pilot40m_20260809/linker_l20_inhand_rotation_09-11-54-00/nn/linker_l20_inhand_rotation_phase5.pth`
- deterministic oracle：
  `artifacts/inhand_refactor_20260808/stage1_repair_20260809/bottomup/hora9d_window035_bound100_drive50_pilot40m_final.json`
- oracle 条件：256 env、seed 0、固定 64 mm、full DR、800 policy steps；533 个完整 episode。
- 结果：net/authorized turns `0.03670`，total turns `0.17909`，success `0%`，fall `12.0075%`，
  palm `>0.05 N` step-env fraction `0.05469%`。

因此 observation 修复的 wiring 是正确的，掌面门槛也通过，但持续旋转与稳定性均明确失败；
9-D parity 本身不能证明增加到 300 M 会跨过 `1.5 turns` gate。主要后续风险仍包括策略能否学出
drive/release/return gait，以及 30x32 proprio history 能否稳定回归含物体位置反馈的 latent；后者
必须由 Stage‑2 四路 closed-loop gate 判定，不能用 Stage‑1 oracle 代替。当前停止在失败证据处，
不直接重启 300 M。

此外，若生产策略继续采用修复 pilot 的 home-relative `joint_motion_range` 与 reset hold/ramp，
必须把这些参数写入 deploy bundle 并在 `DeployPolicy` 中执行相同 clamp/ramp；当前 exporter 只
持久化 `action_delta_scale` 和硬件 joint limits，尚不足以证明 sim/deploy 控制语义一致。

在新的 9-D teacher 重新通过 bottom-up oracle 前，禁止启动 top-down 修复全量、Stage-2、
导出或宣传 deploy-ready policy。当前 GPU 空闲。

## 最终报告

每个方向生成独立报告，至少包含：commit SHA、cache manifest SHA、实物质量、任务 ID、
seed/env/horizon/总步数、所有候选 checkpoint、Stage‑1 指标走势与挂起/resume、Stage‑2 MSE、
四路验收表、open-loop 诊断（若有）、no-palm 分布、`deploy.pth` SHA、不可变 package ID、
dry/no-send/startup/live commissioning 结果。

只有四路 closed-loop PASS 且硬件 commissioning 完成，才可写“可上真机”。否则明确写
`rejected` 或 `sim-qualified-hardware-commissioning-required`，不能用 Stage‑1 好看、MSE 小、
open-loop 正常或视频观感替代。
