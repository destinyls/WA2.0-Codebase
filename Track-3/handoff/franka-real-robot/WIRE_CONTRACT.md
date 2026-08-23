# Franka 真机 Wire Contract

本文件把模型内部字段、WorldArena packet 和机器人端物理量逐项对齐。现场实现若不同，必须先修改配置/adapter 并重新验证，不能靠口头别名继续测试。

## 1. 相机与历史

| WorldArena 输入角色 | 模型内部 key | 契约 |
|---|---|---|
| global camera | `cam_high` | README 中的 global/top 是同一相机角色，不是两台相机 |
| left **或** right wrist camera | `cam_left_wrist` | 单臂 Franka 的 legacy 内部名；现场只能激活一个 wrist role，同时提供 left/right 会因重复映射而拒绝 |

- 当前帧使用 packet 的 `timestamp_ns`；回填历史使用 `frame_history_timestamps_ns`。
- 默认 `max_history_frames=5`，稳定阶段为训练对齐的 `4k+1=5` causal frames；warm-up 可以少于 5，但不得使用未来帧。
- 相机在六步执行和 inference 期间持续以目标 `10 Hz` 采集；下一 packet 回填这段时间内的新帧。
- global 与 wrist 的 role、分辨率、颜色顺序和时间戳单位必须在 manifest 中固定。

## 2. 当前机器人状态

- 每轮 inference 使用本轮 packet 的真实末端 base-frame pose、关节状态和 gripper feedback。
- pose action/state 四元数顺序固定为 `qx,qy,qz,qw`（XYZW），且为单位四元数。
- `state_timestamp_ns` 不得早于当前 RGB；允许 skew 由 schema-v4 配置给出。
- 绝对 pose 的 frame、米/弧度单位和 gripper feedback 单位必须由机器人团队签字确认。

## 3. Future action

- 模型内部预测 12 个候选 action。
- 实际代码切片为 Python half-open `prediction[6:12]`，即索引 6、7、8、9、10、11。
- metadata 写作 `selected_prediction_range=[6,12]`，这是 half-open 范围，不表示包含索引 12。
- Policy response 形状固定 `(6,8)`；字段为 `[x,y,z,qx,qy,qz,qw,gripper_width_m]`。
- 六步只返回一次，`queue_depth_after=0`；执行完成并收到更新 observation 后重新 inference，不保留第二批旧 action。

## 4. Timing

- requested action rate 为 `15 Hz`，理论相邻 applied-step interval 约 `66.7 ms`。
- 六个 applied timestamp 的 `t5-t0` 只有五个间隔，理论约 `333 ms`。
- 从第一步 apply 开始到第六步 hold 完成的完整 six-action batch occupancy 理论约 `400 ms`。
- 现场 PASS 容差必须来自 robot executor/organizer 契约，并写入 manifest；模型端理论值不构成通过证据。

## 5. Gripper 单位

- 模型 action 的最后一维是物理宽度 `width_m`。
- official wire 使用归一化 `open_ratio`，换算为：`open_ratio = width_m / gripper_max_width_m`。
- Policy 先按配置安全范围裁剪 `width_m`；bridge 对超出 `[0, gripper_max_width_m]` 的值 fail closed，再生成 `[0,1]` 的 ratio。
- 现场必须用至少三个安全开度点，保存 `model width_m -> bridge open_ratio -> robot applied ratio -> measured width_m` 的 roundtrip 证据。
- 若机器人端显示固定 `0.8`，必须分清它是 ratio、宽度、状态还是 command；未完成 roundtrip 前不得解释为模型输出正常。

## 6. 最小日志字段

每轮保存：observation sequence、两路 RGB capture timestamps、state timestamp、history length、plan seed、inference latency、selected range、六个 model actions、六个 wire actions、六个 applied/measured states、safety intervention、step intervals 和 batch occupancy。
