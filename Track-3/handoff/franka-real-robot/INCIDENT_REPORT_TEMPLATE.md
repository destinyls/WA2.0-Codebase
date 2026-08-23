# Franka 真机测试故障报告

## 一句话摘要

`时间 + 阶段 + 现象 + 是否触发停止/E-stop`

## 身份

| 字段 | 值 |
|---|---|
| session/episode | |
| source commit | |
| checkpoint identity | |
| serve bundle receipt | |
| Policy config SHA-256 | |
| WorldArena revision | |
| worker key | |
| evidence root | |

## 故障分类

- [ ] 启动/preflight
- [ ] checkpoint/bundle/config identity
- [ ] Hub 网络或认证
- [ ] 双相机/history/timestamp
- [ ] current pose/state
- [ ] Policy inference/nonfinite/OOM
- [ ] action shape/XYZW/quaternion
- [ ] stale queue/fresh-replan
- [ ] gripper width/open_ratio
- [ ] robot executor cadence
- [ ] safety intervention/workspace
- [ ] 任务表现
- [ ] 其他

## 时间线

| 绝对时间/时区 | 组件 | 事件 |
|---|---|---|
| | model worker | |
| | Hub | |
| | robot | |

## 最小必要证据

- 完整 model worker 日志路径：
- 完整 Hub 日志路径：
- 完整 robot command/applied/measured trace 路径：
- observation/history 时间戳文件：
- 现场视频路径：
- 最后一个正常 plan/step：
- 第一个异常 plan/step：
- 原始 traceback/error（不要只截最后一行）：
- HCU/GPU/CPU/内存状态：
- 是否 OOM/Killed/nonfinite：
- 是否执行 E-stop：

## 故障前最后一轮契约

```text
observation_sequence_id =
image_timestamp_ns =
state_timestamp_ns =
history_frame_count =
plan_seed =
execution_mode =
selected_prediction_range =
returned_future_actions =
queue_depth_after =
requested_action_hz =
median_applied_interval_ms =
batch_occupancy_ms =
model_gripper_width_m =
wire_gripper_open_ratio =
robot_measured_width_m =
safety_intervention_count =
```

## 已采取动作

- 停止方法：
- 是否修改任何文件/配置：
- 是否重试：
- 重试是否使用新 session/output/log：

不要删除失败输出，不要覆盖日志，不要未经同意放宽安全参数、绕过 fail-closed 校验或更换训练/动作语义。

## 初步判断

- 已确认事实：
- 尚未确认：
- 建议的最小复现：
- 是否需要开发者介入：YES / NO
