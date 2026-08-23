# Franka 真机测试验收标准

## 证据等级

| 等级 | 含义 | 本次可证明内容 |
|---|---|---|
| CODE | 静态实现和配置契约 | future6、XYZW、单位转换、fail-closed 逻辑存在 |
| SMOKE | 单元测试、模拟、dry run | 软件协议在模拟输入下成立 |
| LINK | 真实 Hub 数据链路 | 收到真实双相机 history 和真实机器人状态 |
| CLOSED-LOOP | 机器人实际应用并反馈 action | command/applied/measured/timing 可追踪 |
| TASK | 完成 Wipe 任务 | 由视频和任务判据确认，不等同官方成绩 |
| OFFICIAL | organizer 接受的结果 | 只能由官方系统确认 |

## 硬性通过条件

### 1. 身份

- clean Git commit、checkpoint、normalizer、serve bundle、receipt、Policy config hashes 完整且相互匹配。
- task=`wipe`，prompt=`Wipe the marked area clean.`。
- official WorldArena revision 和 bridge SHA-256 与 manifest 一致。

### 2. 输入

- 每次 inference 有 global/top 和 wrist 两路 RGB。
- history 长度为正的 `4k+1`，目标 cadence 10 Hz，无未来帧。
- `observation_sequence_id` 与 `image_timestamp_ns` 每轮递增。
- current state 来自本轮 packet，base-frame pose 和 gripper feedback finite。
- state-image skew 不超过 schema-v4 配置限制。

### 3. 模型输出

- `actions.shape == (6,8)`，全部 finite。
- action schema 为 `[x,y,z,qx,qy,qz,qw,gripper_width_m]`。
- quaternion 为 XYZW，范数满足现场接口容差。
- `execution_mode == future6_then_fresh_replan`。
- 代码切片是 Python half-open `prediction[6:12]`；metadata 记录为 `selected_prediction_range == [6,12]`，二者都表示索引 6 到 11 的六步。
- `discarded_future_prediction_count == 0`。
- `queue_depth_after == 0`，每轮 `generated == true`。

### 4. 真机执行

- 一个 Policy response 只对应六个 applied robot steps。
- 六步之后没有来自旧计划的额外动作。
- 下一轮 inference 发生在新的 observation packet 上。
- 相邻 applied-step interval 的 15 Hz 理论值约 `66.7 ms`；`t5-t0` 含五个间隔，约 `333 ms`。
- 从首步 apply 开始到第六步 hold 完成的完整 batch occupancy 目标约 `400 ms`；实际容差必须由机器人/organizer 契约给出并记录，不能由模型端自行宣称通过。
- 六步执行期间相机仍持续采集并在下一 packet 回填。
- command、applied、reached/measured 三种状态可以逐步关联。

### 5. 夹爪

- 模型输出记录为 `width_m`，范围满足现场标定。
- wire command 记录为 `open_ratio`，范围 `[0,1]`。
- measured width 回到 `width_m` 后与同一校准契约一致。
- 不再只观察到与模型命令无关的固定 `0.8`，除非该值确实由对应 width 命令产生并有 roundtrip 证据。

### 6. 安全

- 所有 action 在 workspace 和 per-step safety envelope 内。
- 任何 safety intervention 都写入记录；连续或大量 intervention 不得被当成任务成功。
- E-stop、停止和恢复均由现场责任人确认。

## 立即停止条件

- 输入时间戳倒退、双相机缺失、history 断档或 future frame。
- Policy 返回非 `(6,8)`、非 finite、非单位 quaternion 或错误 action schema。
- `queue_depth_after != 0`，或六步后继续出现旧计划动作。
- state/action/gripper 单位无法确认。
- command 与 measured state 持续不一致、明显向错误方向运动或超出安全边界。
- worker/Hub/robot 任一端日志丢失，导致动作无法关联。
- onsite operator 或 E-stop 不可用。

## 结果表述

- 只有 CODE/SMOKE：写“软件契约通过”，不得写“真机可用”。
- 达到 LINK：写“真实接口数据链路通过”，不得写“闭环成功”。
- 达到 CLOSED-LOOP：写“受控闭环 action trace 通过”，同时报告 timing 和失败次数。
- 达到 TASK：写清 episode 数、成功判据和视频证据；仍不得称官方成绩。
- OFFICIAL 结果必须引用 organizer receipt/页面。
