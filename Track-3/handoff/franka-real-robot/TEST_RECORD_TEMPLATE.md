# Franka 真机测试记录

## 基本信息

| 字段 | 记录 |
|---|---|
| 日期/时区 | |
| 测试人员 | |
| 现场安全负责人 | |
| session ID | |
| episode ID | |
| task/prompt | `wipe` / `Wipe the marked area clean.` |
| source commit | |
| checkpoint identity | |
| serve bundle receipt | |
| Policy config SHA-256 | |
| evidence root | |

## 前置结果

| Gate | 结果 | 证据路径 |
|---|---|---|
| targeted tests | PASS/FAIL | |
| history transport simulation | PASS/FAIL | |
| bridge audit | PASS/FAIL | |
| worker dry run | PASS/FAIL | |
| E-stop/homing | PASS/FAIL | |
| gripper calibration | PASS/FAIL | |

## 每轮闭环记录

| Plan | observation seq | image ts | state ts | history frames | plan seed | infer ms | selected range | returned actions | queue after | median applied interval ms | batch occupancy ms | next obs fresh | result |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 0 | | | | | | | `[6,12]` | 6 | 0 | | | | |
| 1 | | | | | | | `[6,12]` | 6 | 0 | | | | |
| 2 | | | | | | | `[6,12]` | 6 | 0 | | | | |

## Action/状态记录

对每个 action 保存完整机器可读文件；此表仅做索引。

| Plan/step | command XYZ | command XYZW quaternion | model width_m | wire open_ratio | applied timestamp | measured XYZ | measured width_m | safety intervention | 文件路径 |
|---|---|---|---:|---:|---:|---|---:|---|---|
| 0/0 | | | | | | | | | |
| 0/1 | | | | | | | | | |
| 0/2 | | | | | | | | | |
| 0/3 | | | | | | | | | |
| 0/4 | | | | | | | | | |
| 0/5 | | | | | | | | | |

## 任务表现

- 起始状态说明：
- Wipe 成功判据：
- 是否接触目标物：
- 是否抓取/保持毛巾：
- 是否覆盖标记区域：
- 是否存在危险、停顿、震荡或错误方向：
- safety intervention 次数：
- 正常结束/人工停止/E-stop：
- 视频路径和 SHA-256：

## 结论

- 最高证据等级：CODE / SMOKE / LINK / CLOSED-LOOP / TASK / OFFICIAL
- 结果：PASS / FAIL / INCONCLUSIVE
- 主要问题：
- 下一步：
- 测试负责人签字：
- 安全负责人签字：
