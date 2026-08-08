# 任务：refit 姿态 + additive tilt price 的全量 Stage-1 + Stage-2 训练

交给另一台机器上的 agent 执行。**你的任务是训练和验收，不要改架构、不要改奖励。**

代码：`git@github.com:Yyyyyyrl/dex-forge.git`，分支 `rand`，commit **`ff4ad25`**。
该 commit 已验证可从干净 clone 直接训练（245 个测试通过 + 冒烟训练产出 checkpoint）。

---

## 背景（决定你怎么判断对错，必读）

LinkerHand L20 左手、顶视 64 mm 螺丝刀在手旋转，RMA（HORA 式）两阶段策略。

**这一轮之前发生了什么**：硬件视觉标定被提升为正式版，9/16 个关节收紧，其中
`thumb_cmc_yaw` 上限 1.40 → 1.12、`pinky_mcp_pitch` 1.40 → 1.14。旧姿态把
`thumb_cmc_yaw` 定在 1.2407，**超出新上限 0.12 rad**，改一个数救不回来，所以整个抓取姿态
从头重做。全过程与十条测量教训见 `records/topdown_posture_refit_20260806/FINDINGS.md`。

**姿态的已知性质**（不是缺陷，是约束，别试图"修"）：

- **四指承载，中指主动缩回**。五指指腹接触在这只手上不可达：中指唯一的指腹贴壁构型在
  ±8 mm 复位随机化方向上运动学发抖，实测吃到 26.1 N（上限 8 N）。现姿态 4 个接触全部
  指腹、0 个指背、DR 下峰值 6.03 N、倾斜 0.026。
- **零动作时螺丝刀自转 ~0.085 rad/s**（约 74 s 一圈），与姿态和预压无关，是把手在自身负载
  力矩下转动。**策略必须把它按住**，这也是评估里"无接触转速"的本底。
- 把手挂在**万向节**而非轴承上，没有东西能反作用横向力，所以径向不平衡直接变成倾斜。

**本轮 A/B 结论**（`records/stage1_ab_20260807/`）：additive tilt price 胜出并**已设为默认**，
无需设任何环境变量。反向开关 `DEXFORGE_HORA_GATED_TILT_PRICE=1` 只用于复现旧行为。

---

## 环境与铁律

- Python 用 Isaac Lab 环境的解释器（本机是 `/home/user/miniconda3/envs/env_isaaclab/bin/python`，
  你的机器按实际路径替换）。
- 每条训练/评估命令前加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **绝不并行跑训练和 eval**（抢显存并诱发挂起）。
- **所有长任务必须 `setsid` 启动**：
  ```bash
  setsid nohup env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    <python> -u train.py ... > run.log 2>&1 < /dev/null &
  ```
  用普通 `cmd &` 起的子进程会在启动它的 shell/工具调用结束时被回收 —— 现象是**打印完启动
  横幅就静默消失、日志无任何报错**，极易误判为崩溃。历史上以三种形式各踩过一次。
- **stdout 会块缓冲**，不加 `PYTHONUNBUFFERED=1` + `python -u` 你将看不到任何训练进度，
  并会把正常训练误判为挂起。

### ⚠️ 参数会被静默吞掉

`train.py` 用 `parse_known_args()`（Kit 要自己消费 argv），**拼错或不存在的参数不报错**。
本项目已被咬两次：一次 `--stage2_phase 1` 未生效、一次 `--run_name`（**这个参数根本不存在**，
运行目录参数是 `--output`）。

commit `ff4ad25` 加了警告。**启动后第一件事就是确认日志里没有这一行**：

```
[train] WARNING: these arguments were not recognised by train.py and have NO effect: ...
```

出现了就停下改命令，不要让 9 小时的跑带着一个静默失效的参数进行。

---

## 启动即验的三项契约

启动后立刻在日志里确认，**任何一项不符就停下报告**：

1. `[train] proprio_dim 32 -> 96 (32 x 3 frames, from the task cfg)` —— 3 帧堆叠生效。
2. 没有上面那条未知参数 WARNING。
3. 首个 epoch 块里 `Curriculum Phase 1/3`。

架构契约（**勿改**）：
```
actor obs   = 98 = proprio 96 (最近 3 帧 × 32) + 慢变 extrinsics 2 (load proxy + contact friction)
critic obs  = 20 (完整特权状态，仅训练用，不部署)
proprio_hist= (30, 32)
```

---

## 阶段一：Stage-1（约 9 小时）

8192 envs × horizon 32 = **262,144 样本/epoch**。300 M 步 → **1145 epochs**。
curriculum 边界：40 M = **epoch 153**，90 M = **epoch 344**。

**推荐用仓库自带 supervisor**（独立会话托管 + 挂起自动 resume）：

```bash
cd <repo>
OUT=runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_$(date +%Y%m%d)
setsid nohup <python> tools/supervise_hora_full_training.py \
  --stage 1 --output $OUT --log $(pwd)/stage1.log \
  --num-envs 8192 --seed 20260807 \
  --target-global-steps 300000000 --horizon-length 32 \
  --stall-seconds 300 > supervisor.log 2>&1 < /dev/null &
```

> **`--stall-seconds` 必须 > 启动耗时（实测 ~5 分钟）**。默认 120 s 会在场景初始化期间
> 判定挂起并 kill，于是每次重启都被杀 —— **永远起不来的死循环**。
>
> **已知环境问题**：Isaac/PhysX 长跑会间歇性挂起（日志 fps 行突然停、无 traceback / NaN /
> GPU XID，python 进程孤儿化占显存）。间隔不稳定，观察过 ~15 分钟到 ~1 小时。supervisor
> 就是为此存在的。**统计挂起次数，最终报告要写。**
>
> 手工 resume 时 **`--init_global_steps` 不能漏**，否则 curriculum 从头开始，前面白训。

启动后必须验证进程确实在跑、且关键参数确实在命令行里：
```bash
pgrep -a python | grep train.py
```

### 监控（每 15–30 分钟）

每个 epoch 块看：`NetTurns`（主进度）| `UprightGate`（>0.9 优）| `OscRatio`（<0.1 优）|
`FwdVel > RevVel` | `Curriculum Phase x/3` | `TOTAL REWARD`。

**本轮 A/B 参考点**（36.7 M 步，pinned 中间档评估，256 envs × 512 episodes）：

| | 掉落 | 授权净转 | 成功率 |
|---|---|---|---|
| 本配置 ep_140 | 0.0% | **2.89** | 92.2% |
| 旧 baseline 配置 ep_140 | 0.0% | 2.61 | 85.2% |

**历史对照**（协议不同，量级参考）：ep330 掉落 6.8% / 授权净转 1.014；
ep645 掉落 8.41% / 1.951（这是史上最好的 Stage-1，但 adapter 崩了）。

### 红旗：Phase 3

**epoch 344 之后是这个任务历史上真正出事的地方。** 上一轮三帧运行在 Phase 3 失稳：
振荡比 0.25 → 0.44，回报 +4989 → −2158。另有一轮在 Phase 3 单调退化，授权净转
0.726 → 0.465 → 0.215。

所以：**进入 Phase 3 后加密监控**，若振荡比翻倍或回报转负 → **停下报告**，不要续训烧算力。
同时**保留 Phase 3 之前的 checkpoint**，它可能比最终 checkpoint 好。

### 停训标准

curriculum 进入 phase 3/3，NetTurns 连续 ~300 epoch plateau，且 UprightGate ≥0.9、
OscRatio ≤0.1 → 停训跑 oracle eval。

---

## 阶段二：Stage-2（off-policy，**不要**加 `--adapt_onpolicy`）

```bash
S1=<Stage-1 最终 checkpoint>
setsid nohup env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <python> -u train.py --task Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora --stage 2 \
  --checkpoint $S1 --num_envs 2048 --adapt_iters 150 --adapt_save_interval 20 \
  --seed 20260807 --output $OUT --headless > stage2.log 2>&1 < /dev/null &
```

收敛监控：
```bash
python3 -c "import json,glob;r=json.load(open(glob.glob('$OUT/*/stage2_nn/adaptation_validation.json')[0]));print(r['iter'],r['adapter_target_mse'])"
```
`adapter_target_mse` 连续两个 checkpoint ≤0.02 或明显 plateau 即停（通常 60–120 iter）。

> **Stage-2 才是这个任务真正的关卡。** Stage-1 好看从来不足以证明可部署：ep645 的
> Stage-1 是史上最好的，其 adapter 在闭环下从 7% 掉落崩到 **64%**。**不要因为 Stage-1
> 数字漂亮就宣布成功。**

---

## 验收

确认训练进程全部结束、GPU 释放后跑四组（native/bench × oracle/adapter）：

```bash
S1=<Stage-1 最终 checkpoint>; AD=$OUT/<run>/stage2_nn/deploy.pth
T=Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora
P="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True <python> eval.py"
$P --task $T --checkpoint $S1 --num_envs 256 --seed 0 --json_output oracle_native.json
$P --task $T --checkpoint $S1 --deploy_eval --adapter_checkpoint $AD --num_envs 256 --seed 0 --json_output adapter_native.json
$P --task $T --checkpoint $S1 --bench_dr --fixed_geometry_diameter_mm 64 --num_envs 256 --seed 0 --json_output oracle_bench.json
$P --task $T --checkpoint $S1 --deploy_eval --adapter_checkpoint $AD --bench_dr --fixed_geometry_diameter_mm 64 --num_envs 256 --seed 0 --json_output adapter_bench.json
```

> **`--eval_phase` 是 0-based，控制台显示是 1-based。** `--eval_phase 1` 是**中间档**
> （@40 M、DR 0.60、45 s），显示成 "Phase 2/3"。不加 `--eval_phase` 则用 checkpoint 原生档。
> 比较不同 checkpoint 时**必须 pin 同一档**：同一个 ep330 在 Phase 2 读 6.8% 掉落、
> Phase 3 读 21.6%，历史上曾因此把结论读反。
>
> **不要只评最后一个 checkpoint**，至少取 4 个。历史上有运行在末段单调退化，
> 只看最后一个会得到相反结论。

### 验收标准（PASS 需全满足）

- **主判据（部署鸿沟）**：native 和 bench 下 adapter `fall_rate` ≤ oracle + 5 pt（不得 >40%），
  且 adapter `net_turns_mean` ≥ oracle 的 70%。
- **次判据（策略质量）**：bench 下 oracle `fall_rate` ≤10%、`net_turns_mean` ≥1.5。
- 主 PASS 次不达标 → 架构对、策略欠训 → 回 Stage-1 续训重验。
- adapter 相对 oracle 崩溃 → 加 `--deploy_openloop` 区分协变量漂移 vs wiring 后报告。

### 附加检查

- **接触驱动**：读 `eval` 的 coasting 探针，有接触 vs 无接触的前向转速。无接触转速应接近
  本底 0.085 rad/s；若计分旋转有很大比例发生在无接触状态，说明策略在"蹭"而不是在拧。
  本轮 ep_140 参考：接触门开 95.8% 的步，接触 0.522 rad/s vs 无接触 0.091 rad/s。
- **非指尖接触**：`WrongSurf N` 行下面的 `└ vs object N` 应为 **0.000**。上面那行统计所有
  接触来源（含手自碰），会有尖峰，**那是正常的**；只有 `vs object` 非零才说明有非指尖
  连杆压到了把手。
- **动作饱和**：真机上一版策略输出恒定饱和动作把关节顶死。若 net_turns 正常但真机冻结，
  检查部署端 `--rail-lock-ticks` 是否触发。

---

## 最终报告（生成 `TRAINING_REPORT_20260807.md`）

1. **配置**：commit sha、任务 ID、seed、num_envs、Stage-1 总步数/epoch、Stage-2 iter、
   所有 checkpoint 绝对路径
2. **契约确认**：`proprio_dim 32 -> 96` 那行；无未知参数 WARNING
3. **Stage-1 摘要**：NetTurns / UprightGate / OscRatio 随步数走势（**标注 epoch 153 和 344
   两个 curriculum 边界前后的变化**）、总墙钟、**挂起次数与每次 resume 步数**
4. **Stage-2 收敛**：`adapter_target_mse` 走势 + 最终值 + iter 数
5. **验收表**：native/bench × oracle/adapter 四行，列 fall_rate、net_turns、success、latent_mae；
   Stage-1 至少 4 个 checkpoint 的 pinned 对比
6. **判定**：逐条对照标准 PASS/FAIL
7. **交付物**：`deploy.pth` 绝对路径 + 对应 Stage-1 checkpoint
8. **问题与偏差**：异常、未达标项、需人工决策点

## 交付

PASS → 给出 `deploy.pth` 路径 + 验收表，声明可上真机。
不 PASS → 报告 + 明确下一步建议。

**任何时候若观察与本文"背景"中的期望明显矛盾，停下报告，不要自行改架构或改奖励。**

---

## 这一轮的方法论教训（值得照做）

`records/topdown_posture_refit_20260806/FINDINGS.md` 里记了十次测量失误，全部同一个模式：
**量了一个便宜的代理量，而不是被判定的那个量**。例子——用指尖标记点代替接触点测高度、
用链杆名代替几何判定驱动接触、只对最近的零件查穿透（结果漏掉另一个零件里 4.86 mm 的
互穿）、用离线重建代替仿真内测量。

推论两条，请在本轮沿用：

- **"我没搜到" ≠ "不存在"**。一次 2-D 扫描宣布某构型不可达，只因为它在拟合值 ±0.30 rad
  内采样，而解在两个轴上都是 −0.32/−0.38。
- **判据在看数据之前定死**。本轮 A/B 的判定规则是在跑之前写下的，所以 +9.7% 这个结果
  没有事后挑选的余地。
