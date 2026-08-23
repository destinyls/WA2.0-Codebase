# Franka 真机交付清单

交付人填写并签字；`REQUIRED` 未替换即视为未交付。

## 1. 交付身份

| 字段 | 值 |
|---|---|
| 交付日期/时区 | `REQUIRED` |
| 交付人 | `REQUIRED` |
| 测试负责人 | `REQUIRED` |
| 现场安全负责人/E-stop 操作人 | `REQUIRED` |
| 项目代码绝对路径 | `REQUIRED` |
| clean Git commit | `REQUIRED` |
| `git status --short` | 必须为空 |
| Python/torch/runtime identity | `REQUIRED` |
| 设备/节点 | `REQUIRED` |

注意：必须填写目标 GitHub 分支上的最终 commit；本地开发基线或未提交工作树都不能作为交付 revision。

## 2. 模型与数据身份

| 字段 | 值 |
|---|---|
| 任务 | `wipe` |
| 模型 prompt | `Wipe the marked area clean.` |
| checkpoint 候选路径 | `/mnt/data/task/n0_twam_track32_franka_xyzw_20260817_v1/runs/franka_xyzw_wipe_2n16_step10000_20260822_v55-currentstate-fit/checkpoints/checkpoint_step_10000` |
| checkpoint identity SHA-256 | `REQUIRED` |
| checkpoint receipt 路径/SHA-256 | `REQUIRED` |
| normalizer 路径/SHA-256 | `REQUIRED` |
| source manifest SHA-256 | `REQUIRED` |
| serve bundle 绝对路径 | `REQUIRED` |
| serve bundle receipt SHA-256 | `REQUIRED` |
| serve output 新目录 | `REQUIRED` |
| Policy schema-v4 配置路径/SHA-256 | `REQUIRED` |

## 3. 官方接口身份

| 字段 | 值 |
|---|---|
| WorldArena root | `REQUIRED` |
| WorldArena revision | `6f5a981b34232fe77812b818a6ad7a4e6b8728ac` |
| legacy bridge SHA-256 | `f5d264a1af4ff6b3eb22cd9cfedc9b4ebaff9a1f1d753ff08e60f2428340535e` |
| Hub URL | `REQUIRED`，不得提交真实内部 URL |
| worker key/policy ID | `REQUIRED` |
| token 注入方式 | `REQUIRED`，只允许环境变量或 secret manager |

## 4. 真机安全标定

| 字段 | 值 |
|---|---|
| workspace min/max | `REQUIRED` |
| max translation/action | `REQUIRED` |
| max rotation/action | `REQUIRED` |
| gripper min/max width_m | `REQUIRED` |
| max gripper step | `REQUIRED` |
| robot-side action frequency | 目标 `15 Hz`；Stage B 前可为 `PENDING_STAGE_B`，之后填实测值 |
| consecutive applied-step interval | 目标约 `66.7 ms`；Stage B 前可为 `PENDING_STAGE_B`，之后填实测值 |
| six-action batch occupancy | 从首步 apply 开始到第六步 hold 完成目标约 `400 ms`；Stage B 前可为 `PENDING_STAGE_B`，之后填实测值 |
| homing 责任方/方法 | `REQUIRED`；禁止模型全零 action homing |
| E-stop 验证时间/人员 | `REQUIRED` |

## 5. 证据目录

所有测试使用一个全新的只增不改目录：

```text
REQUIRED_EVIDENCE_ROOT/
├── manifest/
├── preflight/
├── model-worker/
├── hub/
├── robot/
├── video/
├── observations/
├── actions/
└── incident/
```

| 字段 | 值 |
|---|---|
| evidence root | `REQUIRED` |
| clock/timezone | `REQUIRED` |
| model/Hub/robot clock correlation方法 | `REQUIRED` |
| 测试 session/episode IDs | `REQUIRED` |

## 6. 签字

- 交付人：`REQUIRED`
- 测试负责人：`REQUIRED`
- 现场安全负责人：`REQUIRED`
- 是否允许进入带物体 Wipe：`NO` / `YES`

`PENDING_STAGE_B` 只允许一次无物体、单 batch cadence 标定。进入 Wipe 前，本节所有 timing 字段必须为实测值，[ROBOT_SIDE_PROCEDURES.md](ROBOT_SIDE_PROCEDURES.md) 必须填写并签字，且三方明确实际容差。
