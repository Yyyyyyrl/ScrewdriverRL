# 任务：HORA-faithful + 3 帧堆叠架构的全量 Stage-1 + Stage-2 训练

## 背景（必读，决定你怎么判断对错）
LinkerHand 顶视螺丝刀旋转的 RMA（HORA 式）策略。两轮架构修复已完成，**你的任务是训练，不要改架构**。

**第一轮修复（已验证）**：旧架构把快变对象状态（euler/角速度/四元数/相对位置/接触）塞进 actor 的
privileged latent，导致 Stage-2 adapter 真机闭环协变量崩溃（oracle fall 7%/turns 2.31 →
adapter fall **64%**/turns 0）。改为 **actor latent 只编码慢变 extrinsics**（load proxy + contact
friction，2 维），对象状态移出 actor 通路、仅保留在 asymmetric critic。
全尺度验证 PASS：native oracle fall 0.134/turns 0.648 → adapter **0.111/0.669**（鸿沟消失）。

**第二轮修复（本次，未经训练验证）**：单帧 actor 缺时序信息。首轮全量训练在 ~112M 步 plateau
（NetTurns 0.35、UprightGate 卡在 0.79，后 90M 步无改善），真机上 1.5 s 内把所有关节推到限位后
完全冻结。故 **actor obs 改为最近 3 帧堆叠**（32×3=96，对齐 HORA）。同时修复了一个 sim2real
不一致：历史缓冲原在 `_pre_physics_step` 更新（存物理步进**前**的 q），现移到 `_get_observations`，
与 `DeployPolicy.act` 一致。

**关键期望**：3 帧堆叠应同时改善 plateau 和真机冻结。若 UprightGate 仍卡在 0.8 以下、
NetTurns 仍在 0.4 附近 plateau，说明该假设不成立，**停下并报告**，不要盲目续训。

## 环境
- 目录 `/home/user/dex-forge`；Python `/home/user/miniconda3/envs/env_isaaclab/bin/python`
- 每条 python 命令前加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 单张 24GB GPU。**绝不并行跑训练和 eval**（抢显存并诱发挂起）。
- 任务 ID：`Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora`（固定 64mm，全 DR 除几何）

## 当前架构契约（smoke test 已验证，勿改）
```
actor obs   = 98 = proprio 96 (最近 3 帧 × 32) + 慢变 extrinsics 2
critic obs  = 20 (完整特权状态，仅训练用，不部署)
proprio_hist= (30, 32)  —— adapter 输入，同时也是 actor 3 帧的来源
```
`actor_frame_count=3` 定义在 `screwdriver_rotation_topdown_hora_env_cfg.py`，随 codec spec 写进
bundle；`train.py::_sync_proprio_dim` 自动把 `network.proprio_dim` 设为 96（勿手改 yaml）。
部署端 `DeployPolicy` **无需改动**——它用同一个 codec 的 `assemble_actor_input` 组装。

## DR 配置（已确认，勿改）
慢变（每-episode 常数）：contact_friction 0.6–2.5× | screwdriver_load_torque 0.5–4.0× |
rotation_damping 0.5–3.0× | tilt_damping 0.5–2.0× | mass 0.5–2.0× | finger_stiffness/damping 0.8–1.2×
放置/复位：xy ±8mm | z ±2mm | 底座 tilt ±1.4°/yaw ±5.2° | 螺丝刀 tilt ±1.7° | 关节零偏 ±0.015rad
每步：obs_noise 0.01。几何 DR 关闭。
> 负载力矩上界 4.0× 已由实机手感确认合理（阻力很小）。只有 friction + load 两维进 actor latent。

## ⚠️ 已知环境问题：训练会间歇性挂起
Isaac/PhysX 长跑约 1 小时后挂起（日志 fps 行突然停、无 traceback/NaN/GPU XID/过热，python 进程
孤儿化占显存）。**必须**实现"分段+自动 resume"监督循环，否则跑不完。

## 阶段一：Stage-1
```bash
cd /home/user/dex-forge
OUT=runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full3f_$(date +%Y%m%d)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/user/miniconda3/envs/env_isaaclab/bin/python train.py \
  --task Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora --stage 1 \
  --num_envs 4096 --seed 42 --output $OUT --headless > stage1.log 2>&1 &
```
**启动后立刻确认日志出现** `[train] proprio_dim 32 -> 96 (32 x 3 frames, from the task cfg)`。
没有这行说明帧堆叠没生效，**停下检查**。

### 挂起监督（必须实现）
每 60s 检查 `stage1.log` mtime：若 **>180s 未增长、train.py 进程还在、且日志中已出现
`fps step`** → 判定挂起 →

> ⚠️ **`fps step` 这个前提不能省**。4096 envs 的场景初始化 + 首次 rollout 要好几分钟，
> 期间日志完全不输出（进程 `Rl`、GPU 利用率却有 80%）。只看 mtime 会在启动阶段误判为挂起
> 并把正常训练杀掉 —— 实测已踩过一次。停滞判据只在训练循环真正开始后才成立。
`kill -9` 所有 train.py python 进程 → 确认 GPU 释放（`nvidia-smi` 显存 <1GB）→ resume：
```bash
LAST=$(ls -t $OUT/*/nn/last_*_ep_*.pth | head -1)
STEPS=$(grep -oE "Step +[0-9,]+" stage1.log | tail -1 | tr -d ' ,' | grep -oE "[0-9]+")
... python train.py ... --stage 1 --checkpoint $LAST --init_global_steps $STEPS ... >> stage1.log 2>&1 &
```
**`--init_global_steps` 不能漏**，否则 curriculum 从 phase 0 重来，前面白训。

**可直接用的自动监督循环**（实测有效，挂起时自动续训；`131072 = num_envs 4096 × horizon 32`）：
```bash
cd /home/user/dex-forge
L=stage1.log; OUT=<你的 OUT>
while true; do
  sleep 60
  pgrep -f "train.py.*Hora" >/dev/null || { echo "PROCESS GONE"; break; }
  [ "$(grep -cE 'fps step' $L)" -gt 0 ] || continue      # 尚未进训练循环，不判停滞
  s1=$(stat -c %s $L); sleep 90; s2=$(stat -c %s $L)
  if [ "$s2" = "$s1" ] && pgrep -f "train.py.*Hora" >/dev/null; then
    CK=$(ls -t $OUT/*/nn/last_*_ep_*.pth | head -1)
    EP=$(echo "$CK" | grep -oE "_ep_[0-9]+" | grep -oE "[0-9]+"); ST=$(( EP * 131072 ))
    echo "HANG -> auto-resume from ep$EP (steps $ST)"
    for p in $(pgrep -f "train.py"); do kill -9 $p; done
    for i in $(seq 1 25); do sleep 3; [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" = "0" ] && break; done
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/user/miniconda3/envs/env_isaaclab/bin/python train.py \
      --task Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora --stage 1 \
      --num_envs 4096 --seed 42 --output $OUT --headless \
      --checkpoint "$CK" --init_global_steps $ST >> $L 2>&1 &
    sleep 30
  fi
done
```
**推荐直接用仓库自带的 supervisor**（独立会话托管训练并自动 resume，比手写循环健壮）：
```bash
cd /home/user/dex-forge
OUT=runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full3f_$(date +%Y%m%d)
setsid nohup /home/user/miniconda3/envs/env_isaaclab/bin/python tools/supervise_hora_full_training.py \
  --stage 1 --output $OUT --log $(pwd)/stage1.log --num-envs 4096 --seed 42 \
  --stall-seconds 300 > supervisor.log 2>&1 < /dev/null &
```
> **`--stall-seconds` 必须 > 启动耗时（实测 ~5 分钟）**。默认 120s 会在场景初始化期间就判定挂起
> 并 kill，于是每次重启都被杀 —— **永远起不来的死循环**。已给 supervisor 加了
> `_log_has_training_progress()` 前提（stage1 看 `fps step`、stage2 看 `AdaptLoss`）双重保险。
>
> **不要把训练作为监控脚本的子进程启动** —— 监控一旦退出会连带杀死训练（本轮实测踩过）。
> supervisor 用 `setsid` 独立会话，监控脚本只负责观察和报告。

> **所有长任务都必须 `setsid` 启动**（训练、Stage-2、supervisor 皆然）。用普通 `cmd &` 起的
> 子进程会在启动它的 shell/工具调用结束时被回收 —— 现象是**进程打印完启动横幅就静默消失、
> 日志无任何报错**，极易误判为崩溃。本轮以三种形式各踩一次（监控子进程、`&`、`nohup &`），
> 唯一可靠的是：
> ```bash
> setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True <python> train.py ... \
>   > run.log 2>&1 < /dev/null &
> ```
> 启动后**必须验证**：`pgrep -a python | grep train.py` 能看到进程，且关键参数确实在命令行里
> （本轮 `--stage2_phase 1` 就曾未生效而未被发现，导致 adapter 差点在错误的 curriculum 下训练）。

> 挂起间隔并不稳定：本轮首次挂起出现在 **~15 分钟**（此前观察是 ~1 小时）。检查点每 15 epoch
> 一次，故每次挂起最多损失 15 epoch。**统计挂起次数**，最终报告里要写。

### 监控（每 15–30 分钟）
每个 epoch 块看：`NetTurns`（主进度，应单调升）| `UprightGate`（>0.9 优）|
`OscRatio`（<0.1 优）| `FwdVel > RevVel` | `Curriculum Phase x/3` | `TOTAL REWARD`

**对照基线（上一轮单帧的表现，新架构应显著超过）**：
| 步数 | 单帧 NetTurns | 单帧 UprightGate |
|---|---|---|
| 25M | 0.25 | 0.87 |
| 112M | 0.42 | 0.80 |
| 201M（终） | 0.35 | 0.79 |

### 停训标准
curriculum 进入 **phase 3/3**，NetTurns 连续 ~300 epoch plateau，且 UprightGate ≥0.9、OscRatio ≤0.1
→ 停训跑 oracle eval：**目标 fall ≤8%、net_turns ≥1.8**。未达标则 resume 续训。
**若 ~120M 步时 UprightGate 仍 <0.85 且 NetTurns <0.5（即与单帧基线无明显差异）→ 停下报告**，
说明 3 帧假设不成立，不要继续烧算力。

## 阶段二：Stage-2（off-policy，**不要**加 `--adapt_onpolicy`）
```bash
S1=<Stage-1 最终 checkpoint>
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/user/miniconda3/envs/env_isaaclab/bin/python train.py \
  --task Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora --stage 2 \
  --checkpoint $S1 --num_envs 2048 --adapt_iters 150 --adapt_save_interval 20 \
  --seed 42 --output $OUT --headless > stage2.log 2>&1 &
```
挂起监督同上，resume 用 `--adapt_resume_checkpoint $OUT/*/stage2_nn/proprio_adapt_last.pth`
（不需 `--init_global_steps`）。收敛监控：
```bash
python3 -c "import json,glob;r=json.load(open(glob.glob('$OUT/*/stage2_nn/adaptation_validation.json')[0]));print(r['iter'],r['adapter_target_mse'])"
```
`adapter_target_mse` 连续两个 checkpoint ≤0.02 或明显 plateau 即停（通常 60–120 iter）。
参考：上一轮 iter-40 = 0.034。

## 验收（deploy-eval）
确认训练进程全部结束、GPU 释放后，跑四组：
```bash
S1=<Stage-1 最终 checkpoint>; AD=$OUT/<run>/stage2_nn/deploy.pth   # 无则用 deploy_last.pth
T=Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora
P="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/user/miniconda3/envs/env_isaaclab/bin/python eval.py"
$P --task $T --checkpoint $S1 --num_envs 256 --seed 0 --json_output oracle_native.json
$P --task $T --checkpoint $S1 --deploy_eval --adapter_checkpoint $AD --num_envs 256 --seed 0 --json_output adapter_native.json
$P --task $T --checkpoint $S1 --bench_dr --fixed_geometry_diameter_mm 64 --num_envs 256 --seed 0 --json_output oracle_bench.json
$P --task $T --checkpoint $S1 --deploy_eval --adapter_checkpoint $AD --bench_dr --fixed_geometry_diameter_mm 64 --num_envs 256 --seed 0 --json_output adapter_bench.json
```
读 `episodes.fall_rate` / `net_turns_mean` / `success_rate` / `authorized_net_turns_mean`；
adapter 另读 `latent_diagnostics.latent_mae`。

### 验收标准（PASS 需全满足）
- **主判据（部署鸿沟）**：native 和 bench 下 adapter `fall_rate` ≤ oracle + 5pt（不得 >40%），
  且 adapter `net_turns_mean` ≥ oracle 的 70%。
- **次判据（策略质量）**：bench 下 oracle `fall_rate` ≤10%、`net_turns_mean` ≥1.5。
- 主 PASS 次不达标 → 架构对、策略欠训 → 回 Stage-1 续训重验。
- adapter 相对 oracle 崩溃 → 加 `--deploy_openloop` 区分协变量漂移 vs wiring 后报告。

### 额外检查：动作饱和（新增，针对真机冻结）
真机上一版策略输出恒定饱和动作把关节顶死。验收时**必须**检查 oracle eval 是否也饱和：
```bash
python3 -c "
import json; r=json.load(open('oracle_bench.json'))
ep=r['episodes']; print('net_turns',ep['net_turns_mean'],'fall',ep['fall_rate'])
print('若 net_turns 正常但真机仍冻结，检查部署端 --rail-lock-ticks 是否触发')"
```

## 最终报告（生成 `HORA_3FRAME_TRAINING_REPORT.md`）
1. **配置**：任务 ID、seed、num_envs、Stage-1 总步数/epoch、Stage-2 iter、所有 checkpoint 路径
2. **契约确认**：日志里 `proprio_dim 32 -> 96` 那行；bundle 里 `proprio_codec.actor_frame_count`
3. **Stage-1 摘要**：NetTurns/UprightGate/OscRatio 随步数走势（含与上表单帧基线的逐点对比）、
   总墙钟时间、**挂起次数与每次 resume 步数**
4. **Stage-2 收敛**：`adapter_target_mse` 走势 + 最终值 + iter 数
5. **验收表**：native/bench × oracle/adapter 四行，列 fall_rate、net_turns、success、latent_mae
6. **判定**：逐条对照标准 PASS/FAIL，并与基线对比（旧架构 adapter 崩 fall 64%/turns 0；
   单帧全尺度 adapter fall 0.111/turns 0.669）
7. **交付物**：`deploy.pth` 绝对路径 + 对应 Stage-1 checkpoint
8. **问题与偏差**：异常、未达标项、需人工决策点

## 交付
PASS → 给出 `deploy.pth` 路径 + 验收表，声明可上真机。
不 PASS → 报告 + 明确下一步建议。**任何时候若观察与"背景"中的期望明显矛盾，停下报告，不要自行改架构。**

---
## 附：真机部署备忘（本轮训练不涉及，交付后使用）
- 上机命令用 `--calib linker_calib_thumbfit.json`（拇指标定已修，2026-07-30）
- 必须 `--sdk-root /home/user/linkerhand-ros-sdk`（机器上有多份 SDK 拷贝）
- 顶视部署强制 `--hand-joint G20`（默认值，勿改）
- **接触门禁已删除**：该手三个触觉垫无信号，门禁永不可满足
- **新增 rail-lock 保护**：`--rail-lock-ticks`（默认 20）——目标顶限位持续 N 拍则中止（退出码 5）
- SDK 里 `linker_hand_g20_can.py` 的拇指修复**仍未提交**，被 checkout 冲掉会退回旧行为

_生成于 2026-07-30。架构与排查全过程见 `docs/topdown-force-free-refactor-plan.md` 的
2026-07-29 / 2026-07-30 各条目。_
