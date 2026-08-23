# 真机测试前检查表

任何一项为 `NO` 时，不得进入下一阶段。

## A. 人员与现场

- [ ] onsite operator 在机械臂旁，知道 E-stop、断电和恢复流程。
- [ ] 独立安全负责人已确认 workspace、速度、单步位移/旋转和夹爪限制。
- [ ] 人员、线缆、相机和待操作物体均不在非预期碰撞区域。
- [ ] 首次运动使用低风险场景；不直接从完整 Wipe 任务开始。
- [ ] homing 由机器人批准流程执行；模型不会发布全零 action 作为 homing。

## B. 源码与运行环境

- [ ] `HANDOFF_MANIFEST.md` 的身份、安全和责任人字段没有 `REQUIRED`；进入 Wipe 前也没有 `PENDING_STAGE_B`。
- [ ] `git status --short` 为空。
- [ ] 当前 commit 等于 manifest 中的 clean Git commit。
- [ ] 当前 commit 包含 schema-v4、`future6_then_fresh_replan` 和 causal history 修改。
- [ ] Python、torch、设备和容器/runtime identity 已记录。
- [ ] 定向测试通过：

```bash
uv run pytest \
  tests/unit/test_franka_policy.py \
  tests/unit/test_franka_official_worker.py \
  tests/unit/test_franka_policy_conditioning_parity.py -q
```

## C. 权重和配置

- [ ] checkpoint 的 `checkpoint_complete.json` 完整且 identity 匹配 Wipe。
- [ ] serve bundle receipt、checkpoint、normalizer 和 source hashes 全部匹配。
- [ ] serve output 是全新目录，不在 immutable serve bundle 内。
- [ ] Policy config 是 schema-v4，且 `policy_id == worker key`。
- [ ] `max_chunk_actions/external_chunk_actions == 12/6`。
- [ ] `action_hz/actions_per_replan/future_start_index == 15/6/6`。
- [ ] `require_dual_camera_history` 和 `require_fresh_observation_after_chunk` 均为 `true`。
- [ ] action schema 为 `[x,y,z,qx,qy,qz,qw,gripper_width_m]`。
- [ ] workspace 和 per-step safety 参数来自当前机器人标定，不是训练默认值。

## D. 官方接口与传感器

- [ ] official WorldArena revision 为 `6f5a981b34232fe77812b818a6ad7a4e6b8728ac`。
- [ ] bridge audit 验证 legacy bridge SHA-256 和双向 XYZW。
- [ ] Hub observation 同时包含 global/top 与 wrist RGB。
- [ ] 相机在六步执行和模型推理期间持续采集，不是请求时临时抓一帧。
- [ ] packet 可以回填带 capture timestamps 的 camera history。
- [ ] 当前 packet 含真实末端 base-frame pose、关节状态和 gripper feedback。
- [ ] state timestamp 不早于当前 RGB，且 skew 在 Policy config 上限内。
- [ ] model、Hub、robot 三端时钟关联方式已经记录。

## E. Action 和夹爪

- [ ] Policy 每次返回 `(6,8)`，所有值 finite，quaternion 为 XYZW 且单位化。
- [ ] `selected_prediction_range == [6,12]`。
- [ ] `queue_depth_after == 0`，下一轮 `generated == true`。
- [ ] 模型输出夹爪单位是 `width_m`；官方 wire 是 `open_ratio`。
- [ ] 已通过至少三个开度点验证 `width_m -> open_ratio -> measured width_m` roundtrip。
- [ ] 机器人端日志能区分 command 与 measured state，不能只记录 Hub 返回。

## F. 软件 smoke

- [ ] continuous-history simulation 为 `verified`。
- [ ] worker `--dry-run` 为 `ready`。
- [ ] `private_executor_cadence_verified=false` 被理解为待真机 trace 验证，而非报错。
- [ ] 日志和视频目标目录是新的、可写的、空间充足的目录。
- [ ] 停止命令和现场 E-stop 均已实际演练。
- [ ] [ROBOT_SIDE_PROCEDURES.md](ROBOT_SIDE_PROCEDURES.md) 已由机器人团队填写并签字。
- [ ] [WIRE_CONTRACT.md](WIRE_CONTRACT.md) 的相机角色、state frame、gripper 标定和 timing 容差已与现场接口逐项核对。

## 放行

- 软件负责人：`____________`  时间：`____________`
- 测试负责人：`____________`  时间：`____________`
- 安全负责人：`____________`  时间：`____________`
- 放行阶段：`NO MOTION / CONTROLLED MOTION / WIPE`

`CONTROLLED MOTION` 只允许无物体、单 six-action batch；若 cadence 尚为 `PENDING_STAGE_B`，完成该批次后必须停止、填写实测值并重新签字，不能直接进入 Wipe。
