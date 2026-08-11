# G20 旧 Top-down Checkpoint 坐标 Adapter 与部署门槛

> **SUPERSEDED (2026-08-08).** Coordinate adapter for the *legacy* top-down checkpoints. The current deployment path exports a self-describing bundle instead; see [docs/deploy/DEPLOY.md](../deploy/DEPLOY.md). Kept because the legacy checkpoints under deliverables/ still need this transform to be interpreted.

日期：2026-08-05  
适用手：LinkerHand G20 left，serial `LHT20-010-415-L-B-1-D`  
当前状态：`offline_validated_live_gate_required`

## 1. 目的与结论

旧 fixed64 Stage-2 checkpoint 可以保留 actor 与 learned proprioceptive adapter 权重，但不能
原生搭配 2026-08-05 的生产校准直接启用。解决方式是在旧 policy 和新 SDK mapping 之间增加
一个 deterministic、双向、内容寻址的 joint-coordinate adapter。

这不是重新训练出来的 neural adapter，也不修改 actor 权重：

```text
physical raw
  -> production piecewise LUT
  -> hardware OG-local q
  -> inverse coordinate adapter
  -> legacy virtual q / virtual target history
  -> old actor + learned latent adapter
  -> legacy virtual target
  -> forward coordinate adapter
  -> hardware OG-local q
  -> production piecewise LUT
  -> SDK raw command
```

当前实现已经通过离线等价和部署 dry-run。它可以用于 no-send、startup-only 和最多 100 ticks
的受限实机动态 gate；在这些 gate 人工通过前，普通 live policy 仍 fail closed。

## 2. 绑定资产

旧 checkpoint：

```text
deliverables/linker_l20_topdown_pip108_d64_300m_20260724/checkpoints/deploy_fixed64_release.pth
SHA256 ec9e706f01f7078c40d4f5b3bc545496011bcbce9248e23f889c07c11934aedd
```

新 production overlay：

```text
linker_calib_deploy.json
SHA256 b728cd58f15408193080a0bd35dc6ae281adff9a96d6115ded012498b6f7ecb7
```

新 calibration artifact：

```text
assets/calibrations/linker_g20_left_lht20_010_415_v1.json
file SHA256 75ea78f32f0b6e154c9e720f0b39c5ef613d934690fa4f8216422a7c71806a10
artifact digest 354d057686e2d47e562b0cf3e40993c677e39bd651302b14b1b88495c28a62dd
semantic schema digest 1bb09533ebf945a693d2726f394b5004cbf0dc2132f045e7d45d3c106a5181ad
```

coordinate adapter：

```text
assets/calibrations/linker_g20_left_legacy_topdown_fixed64_policy_adapter_20260805.json
adapter digest 1156c77148079c3fa331674cce4057eac8b24e97c8a77390f111f6d3cc67de25
```

任一 SHA/digest、joint order、旧 policy contract 或新 SDK limit 不一致时，loader 必须拒绝。

## 3. 变换推导

旧 policy 的 16 轴 action rails 与新硬件验证范围逐轴求交后，只有 thumb yaw 超界：

| 项 | 旧虚拟 policy | 新硬件合同 |
|---|---:|---:|
| thumb yaw action lower | 0.8906667 | 可用 |
| thumb yaw action upper | 1.3800000 | 1.1200000 |
| thumb yaw home | 1.2406667 | 原生超界 |
| thumb yaw reset | 0.7977368 | 可用 |

为保持旧 action span、delta scale 和状态变化量，使用上限对齐的等距平移：

```text
offset = new_upper - old_upper = 1.12 - 1.38 = -0.26 rad
hardware_q = legacy_policy_q - 0.26
legacy_policy_q = hardware_q + 0.26
```

结果：

| 项 | adapter 后硬件 q |
|---|---:|
| thumb yaw action lower | 0.6306667 |
| thumb yaw action upper | 1.1200000 |
| thumb yaw home | 0.9806667 |
| thumb yaw reset | 0.5377368 |

其余 15 轴 `scale=1, offset=0`。不采用全轴 range-fraction remap，因为生产 LUT 已经把
physical raw 映射到与 URDF 相同的 OG-local q；再次比例缩放会破坏已经视觉验证的物理角。

## 4. 运行时实现

- `screwdriver_rl/deploy/policy_coordinate_adapter.py`：digest/SHA/contract 校验、双向变换及
  `PolicyCoordinateAdapter` wrapper；
- `screwdriver_rl/deploy/deploy_linker.py`：新增 `--policy-coordinate-adapter`；
- wrapper 对真实 measured q 做 inverse transform，再调用原 `DeployPolicy.act()`；
- 原 policy 的 `cur_targets` 与 history 保持 legacy virtual coordinate；
- 输出 absolute target 做 forward transform 后才进入 `joints16_to_sdk_range()`；
- startup reset、contact home、acknowledged effective target 和 rail-lock limits 均使用对应侧的
  正确坐标；
- CSV 中 `q`/`tgt` 是硬件 OG-local q，`act` 是旧 actor 的归一化 action。

## 5. 已完成的离线 gate

测试文件：`tests/test_policy_coordinate_adapter.py`。

已检查：

1. 16 轴 forward/inverse round-trip；
2. 只有 thumb yaw 发生 `-0.26 rad` 变化；
3. adapted actor action 与原 actor 完全一致；
4. legacy virtual proprio history 与原 policy 完全一致；
5. target integration 经 forward transform 后等价；
6. checkpoint SHA、overlay SHA、adapter digest 篡改均 fail closed；
7. active SDK joint order 与 16 个生产 limits 必须完全一致；
8. 真实旧 checkpoint + production LUT 的完整 deploy dry-run 通过；
9. 未晋升 adapter 的普通 live policy 被拒绝。

最终相关回归为 `76 passed, 1 xfailed`；xfail 是既有 production endpoint 合同项。真实旧
checkpoint 的 100-tick dry-run 为 `100 policy steps / 0 held / 0 overruns`，记录全部有限，证据：

```text
records/g20_og_local_q_asset_promotion_20260805/legacy_adapter/dry_run.csv
SHA256 34fd18a2a82c0bf85ba1e78cc369631d6e4fd29e0c782f8036d1ee6a289e0ffc
```

## 6. 分阶段命令

所有命令从 `/home/user/dex-forge` 运行。

### 6.1 离线 dry-run

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint deliverables/linker_l20_topdown_pip108_d64_300m_20260724/checkpoints/deploy_fixed64_release.pth \
  --calib linker_calib_deploy.json \
  --policy-coordinate-adapter assets/calibrations/linker_g20_left_legacy_topdown_fixed64_policy_adapter_20260805.json \
  --dry-run --max-ticks 100 --ramp-s 0 --contact-ramp-s 0 \
  --record records/g20_og_local_q_asset_promotion_20260805/legacy_adapter/dry_run.csv
```

### 6.2 实机 no-send

只读 joint/fault state、运行 policy 并计算 command，不向手发送命令：

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint deliverables/linker_l20_topdown_pip108_d64_300m_20260724/checkpoints/deploy_fixed64_release.pth \
  --calib linker_calib_deploy.json \
  --policy-coordinate-adapter assets/calibrations/linker_g20_left_legacy_topdown_fixed64_policy_adapter_20260805.json \
  --transport can --can can0 --no-send --max-ticks 100 \
  --record records/g20_og_local_q_asset_promotion_20260805/legacy_adapter/no_send.csv
```

### 6.3 实机 startup-only

该步骤会移动真手，但绝不启用 policy loop。先确认 64 mm fixture/task frame，再运行：

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint deliverables/linker_l20_topdown_pip108_d64_300m_20260724/checkpoints/deploy_fixed64_release.pth \
  --calib linker_calib_deploy.json \
  --policy-coordinate-adapter assets/calibrations/linker_g20_left_legacy_topdown_fixed64_policy_adapter_20260805.json \
  --transport can --can can0 --startup-only --speed 20 \
  --ramp-s 6 --contact-ramp-s 8 \
  --record records/g20_og_local_q_asset_promotion_20260805/legacy_adapter/startup_only.csv
```

### 6.4 候选 adapter 的受限动态 gate

只有人工确认 startup posture、fixture 和 task frame 后执行。该模式要求 CAN、精确 serial、
`--task-frame-confirmed`、CSV record 和 `--max-ticks 1..100`，缺一项都会拒绝：

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint deliverables/linker_l20_topdown_pip108_d64_300m_20260724/checkpoints/deploy_fixed64_release.pth \
  --calib linker_calib_deploy.json \
  --policy-coordinate-adapter assets/calibrations/linker_g20_left_legacy_topdown_fixed64_policy_adapter_20260805.json \
  --transport can --can can0 --candidate-adapter-gate --task-frame-confirmed \
  --speed 20 --max-ticks 20 \
  --record records/g20_og_local_q_asset_promotion_20260805/legacy_adapter/bounded_gate_20ticks.csv
```

## 7. 实机晋升标准

状态从 `offline_validated_live_gate_required` 改为 `live_promoted` 前必须全部满足：

1. exact serial、side、firmware identity 通过；
2. production calibration/adapter/checkpoint SHA 全部匹配；
3. no-send 至少 100 ticks：state/fault 正常，target 与 raw command 有限且无 rail lock；
4. startup-only：reset/home 视觉姿态与 fixture 接触人工通过，fault 全零；
5. 受限动态 gate 从 20 ticks 开始，逐级增加但每次不超过 100 ticks；
6. 全程记录 CSV，并保留同步相机视频；
7. 无错误方向、突然跳变、thumb self-collision、持续 rail saturation 或异常温升；
8. 明确记录 operator approval 与所有 evidence SHA；
9. 更新 adapter `validation`、`status` 后重新计算 canonical `adapter_digest`；
10. 再跑本文件第 5 节全部测试。

旧 checkpoint + adapter 是兼容/复用路径，不等于重新训练后的最优策略。长期首选仍是在新 URDF、
新 limits 和新 grasp baseline 上重新训练 Stage 1/Stage 2，并导出原生绑定新 calibration digest 的
immutable package。

## 8. 回退

- 不删除或覆盖旧 checkpoint；
- adapter 是独立 JSON，移除 `--policy-coordinate-adapter` 即回到原 policy 行为；
- 但原 policy 不允许直接配新生产 calibration 做 top-down live deployment；
- production calibration 的回退仍以
  `records/g20_og_local_q_asset_promotion_20260805/promotion_manifest.json` 与 `rollback/`
  为唯一依据；不要为了适配旧 policy 修改生产 LUT 或扩大 URDF limit。
