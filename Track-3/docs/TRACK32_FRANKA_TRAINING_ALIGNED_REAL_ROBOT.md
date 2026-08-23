# Track 3.2 Franka 训练对齐真机部署

## 1. 实现目标

本实现把 Franka Policy 的训练语义和真机执行语义固定为同一个闭环：

1. 双相机 RGB 在动作执行期间持续采集，并在 Policy 请求时重建为训练使用的 `10 Hz`、`4k+1` causal history。
2. 当前末端位姿和夹爪宽度来自本次 `ObservationPacket`，不得沿用上一轮状态。
3. 模型仍输出 `[20, 2, 6]`，展平后共有 12 个 action slots。
4. 前 6 个 slots 是 conditioning/current 段，不下发给机器人。
5. 仅把索引 `6:12` 的 6 个 future actions 返回 Hub。
6. 这 6 步按训练契约请求的 `15 Hz` 执行，预计覆盖 `400 ms`。
7. 六步结束后，必须收到更新后的 RGB history 和实测位姿，才能发起下一次推理；不会缓存或继续执行上一计划中的动作。

因此，旧行为 `plan_12_execute_6x2` 已被移除。新行为标识为：

```text
future6_then_fresh_replan
```

## 2. 数据流

```text
机器人持续采集双相机 RGB（10 Hz）
          +
本轮 ObservationPacket 的实测末端位姿/夹爪
          |
          v
FrankaLiveObservationAdapter
  - 重建 causal 4k+1 history
  - 校验时间戳、帧间隔、双相机和 state-image skew
          |
          v
Policy.infer
  - 每轮重新 reset 推理随机状态
  - 生成 12 slots
  - 跳过 [0:6]
  - 安全裁剪并返回 [6:12]
          |
          v
Hub/机器人以 15 Hz 执行 6 步
          |
          v
等待新的 ObservationPacket 后重新规划
```

## 3. 配置方法

先生成 schema-v4 配置模板：

```bash
cd /path/to/N0-TWAM-Franka-PostTraining
python -m n0_twam.cli track32 policy-template \
  --output /absolute/path/to/franka-policy-v4.json
```

替换模板中的 serve bundle、receipt、输出目录和安全边界。与本闭环相关的字段必须保持：

```json
{
  "schema_version": 4,
  "max_chunk_actions": 12,
  "external_chunk_actions": 6,
  "live_contract": {
    "conditioning_mode": "training_aligned_rgb_replan_v1",
    "target_fps": 10,
    "frame_interval_tolerance_ms": 15.0,
    "max_state_image_skew_ms": 50.0,
    "max_history_frames": 5,
    "action_hz": 15,
    "actions_per_replan": 6,
    "future_start_index": 6,
    "require_dual_camera_history": true,
    "require_fresh_observation_after_chunk": true
  }
}
```

`max_history_frames` 必须是正的 `4k+1`。若部署数据允许携带更长 causal history，可按训练设置增加；不得改成非 `4k+1`。

schema-v4 会 fail closed：`external_chunk_actions` 必须等于 6，`action_hz/actions_per_replan/future_start_index` 必须分别为 `15/6/6`。

## 4. 启动方法

### 4.1 软件传输模拟

先确认相机在阻塞推理和动作执行期间仍能持续采集、下一包能够回填历史：

```bash
python script/track3_2/simulate_franka_live_history_transport.py \
  --policy-config /absolute/path/to/franka-policy-v4.json \
  --gap-seconds 0.4 \
  --cycles 3 \
  --output /absolute/path/to/live-history-transport.json
```

预期：`status=verified`、`future_prediction_range=[6,12]`、`return_batches_per_plan=1`、`fresh_plan_per_returned_batch=true`。

### 4.2 Hub worker 预检

```bash
python -m n0_twam.cli track32 worker \
  --worldarena-root /absolute/path/to/official/WorldArena \
  --config /absolute/path/to/franka-policy-v4.json \
  --hub-url https://ORGANIZER_HOST/policy \
  --worker-key YOUR_POLICY_ID \
  --dry-run
```

预期报告中的 `live_observation_adapter` 包含：

```text
execution_mode = future6_then_fresh_replan
requested_action_hz = 15
actions_per_replan = 6
future_prediction_range = [6, 12]
requires_fresh_observation_after_chunk = true
```

### 4.3 启动正式 worker

```bash
export WORLD_ARENA_ROOT=/absolute/path/to/official/WorldArena
export N0_TRACK32_POLICY_CONFIG=/absolute/path/to/franka-policy-v4.json
export HUB_POLICY_URL=https://ORGANIZER_HOST/policy
export POLICY_ID=YOUR_POLICY_ID
export HUB_TOKEN=YOUR_TOKEN

./run_track32_franka_worker.sh
```

## 5. 运行时验收

每次 Policy 返回结果必须满足：

- `actions.shape == (6, 8)`；
- `policy_metadata.execution_mode == "future6_then_fresh_replan"`；
- `selected_prediction_range == [6, 12]`；
- `conditioning_slots_skipped == 6`；
- `returned_future_actions == 6`；
- `discarded_future_prediction_count == 0`；
- `requested_action_hz == 15`；
- `expected_execution_duration_ms == 400`；
- `requires_fresh_observation_after_chunk == true`；
- `policy_timing.generated == true`；
- `policy_timing.queue_depth_after == 0`。

连续两轮还应满足：

- `observation_sequence_id` 递增；
- `image_timestamp_ns` 递增；
- `training_aligned_video_timestamps_ns` 严格递增且接近 10 Hz；
- history 最后一帧就是本次推理的当前图像；
- state timestamp 不早于图像，且偏差不超过配置上限；
- `plan_seed` 每轮递增，证明是新推理而不是旧队列。

以下情况会直接拒绝推理：缺任一相机、没有 history、未来帧、10 Hz 历史断档、时间戳倒退、state-image skew 超限、位姿/夹爪非有限值或超出标定范围。

## 6. 已修改的代码

- `n0_twam/integrations/worldarena/franka_policy.py`
  - 增加 schema-v4 execution contract；
  - 删除 strict 12-action 队列的两次下发逻辑；
  - 每轮只选择 future `[6:12]`，返回 6 步并清空队列；
  - 输出 future range、15 Hz、400 ms、fresh replan 等机器可读 metadata。
- `n0_twam/integrations/worldarena/franka_live_contract.py`
  - 固定 `10 Hz RGB / 15 Hz action / 6 actions per replan / future start=6`；
  - schema-v4 要求显式声明执行契约。
- `n0_twam/integrations/worldarena/franka_live_observation_adapter.py`
  - 输出 packet step/timestamp、时间戳来源、双相机最新时间戳和 history 覆盖范围；
  - 保留动作执行期间的 causal RGB backfill。
- `n0_twam/integrations/worldarena/franka_official_worker.py`
  - 在启动报告中公开新闭环契约及未验证的私有 executor cadence 边界。
- `n0_twam/evaluation/franka_policy_conditioning_parity.py`
  - 离线 Policy parity 与新 future6、zero-queue 语义一致。
- `script/track3_2/simulate_franka_live_history_transport.py`
  - 模拟并核验连续采集、单批 future6 和 fresh replan。

## 7. 证据边界

上述实现可以在模型端强制：输入 history 的因果性、时间戳新鲜度、当前状态绑定、只返回 `[6:12]`、不缓存旧动作，以及每个新 ObservationPacket 重新推理。

模型端无法单方面证明 organizer 私有 Hub/机器人 executor 确实以 `15 Hz` 执行，也无法证明六步执行期间相机进程实际持续运行。启动报告因此把 `private_executor_cadence_verified` 保持为 `false`。正式真机前必须从机器人端 timing trace 核验：六步耗时约 400 ms、六步期间 RGB 持续采集、下一轮 packet 携带完整回填历史。该核验通过后，才能称为真机时序与训练契约一致；不能仅依据离线测试作真机成功声明。

## 8. 本地验证

2026-08-23 执行：

```bash
uv run pytest \
  tests/unit/test_franka_policy.py \
  tests/unit/test_franka_official_worker.py \
  tests/unit/test_franka_policy_conditioning_parity.py -q
```

结果：`29 passed`。该结果证明 CODE/SMOKE 层面的选择范围、zero queue、fresh replan、RGB history 和 metadata 契约；不等同于 CLOSED-LOOP 或真实机器人验证。
