# Franka 真机测试 Runbook

## 0. 原则

- 命令仅从模型端 A-side 发起，不替代机器人现场安全流程。
- 每次实验使用新的 evidence root、session ID、serve output 和日志。
- 不修改 checkpoint、serve bundle、receipt、normalizer 或官方 checkout。
- 任何 fail-closed 异常、时间戳异常、单位不明或运动异常都立即停止，不现场绕过校验。

## 1. 加载交付环境

把 `runtime.env.example` 复制到 Git 仓库外，填写后：

```bash
chmod 600 /secure/path/franka-runtime.env
source /secure/path/franka-runtime.env
cd "$FRANKA_POSTTRAINING_ROOT"
```

不要把真实 token、Hub URL 或现场安全参数写回仓库。

## 2. 固化本次证据目录

```bash
test ! -e "$FRANKA_EVIDENCE_ROOT"
mkdir -p "$FRANKA_EVIDENCE_ROOT"/{manifest,preflight,model-worker,hub,robot,video,observations,actions,incident}

git rev-parse HEAD | tee "$FRANKA_EVIDENCE_ROOT/manifest/source.commit.txt"
git status --short | tee "$FRANKA_EVIDENCE_ROOT/manifest/source.status.txt"
test ! -s "$FRANKA_EVIDENCE_ROOT/manifest/source.status.txt"
sha256sum "$N0_TRACK32_POLICY_CONFIG" \
  | tee "$FRANKA_EVIDENCE_ROOT/manifest/policy.sha256"
python -m json.tool "$N0_TRACK32_POLICY_CONFIG" >/dev/null

if rg -n 'REQUIRED|REPLACE_WITH' \
  /secure/path/franka-runtime.env "$N0_TRACK32_POLICY_CONFIG"; then
  echo "unresolved handoff placeholder" >&2
  exit 1
fi
```

将填写完成的 `HANDOFF_MANIFEST.md` 复制到 `manifest/`。JSON 语法通过不代表语义正确；后续 worker `--dry-run` 必须完成 schema-v4、路径、identity 和安全字段校验。

## 3. 定向软件验证

```bash
uv run pytest \
  tests/unit/test_franka_policy.py \
  tests/unit/test_franka_official_worker.py \
  tests/unit/test_franka_policy_conditioning_parity.py -q \
  | tee "$FRANKA_EVIDENCE_ROOT/preflight/targeted-tests.log"
```

预期：全部通过。该结果属于 CODE/SMOKE，不是真机证据。

## 4. 连续相机 history 模拟

```bash
python script/track3_2/simulate_franka_live_history_transport.py \
  --policy-config "$N0_TRACK32_POLICY_CONFIG" \
  --gap-seconds 0.4 \
  --cycles 3 \
  --output "$FRANKA_EVIDENCE_ROOT/preflight/live-history-transport.json" \
  | tee "$FRANKA_EVIDENCE_ROOT/preflight/live-history-transport.log"
```

必须看到：

```text
status = verified
future_prediction_range = [6, 12]
returned_future_actions = 6
return_batches_per_plan = 1
fresh_plan_per_returned_batch = true
requested_action_hz = 15
expected_batch_duration_ms = 400
```

## 5. 官方 bridge audit

```bash
python -m n0_twam.cli track32 bridge-audit \
  --worldarena-root "$WORLD_ARENA_ROOT" \
  | tee "$FRANKA_EVIDENCE_ROOT/preflight/bridge-audit.json"
```

必须核对 pinned revision、bridge SHA-256、input XYZW、output XYZW 和 control arm probe。

## 6. Worker dry run

```bash
python -m n0_twam.cli track32 worker \
  --worldarena-root "$WORLD_ARENA_ROOT" \
  --config "$N0_TRACK32_POLICY_CONFIG" \
  --hub-url "$HUB_POLICY_URL" \
  --worker-key "$POLICY_ID" \
  --dry-run \
  | tee "$FRANKA_EVIDENCE_ROOT/preflight/worker-dry-run.json"
```

必须看到：`status=ready`、`execution_mode=future6_then_fresh_replan`、`future_prediction_range=[6,12]`、`actions_per_replan=6`。`private_executor_cadence_verified=false` 表示必须在下一阶段采集机器人 trace。

## 7. 现场准备与 homing

1. 现场人员清空工作区并确认 E-stop。
2. 使用机器人方已批准的 homing 程序完成回零。
3. 确认回零后的实测末端位姿、夹爪宽度和相机画面合理。
4. **不要**通过模型发布全零末端 action 进行 homing。
5. 先建立不含任务物体的受控运动场景。

## 8. 启动 Worker

```bash
bash run_track32_franka_worker.sh \
  2>&1 | tee "$FRANKA_EVIDENCE_ROOT/model-worker/worker.log"
```

同时在 Hub 和机器人侧分别保存原始日志，不要只保留模型端 stdout。现场视频必须可见机械臂、夹爪、毛巾/标记区域和 E-stop 操作人。

## 9. 分阶段测试

### Stage A：无运动链路检查

- 建立 Hub session，但机器人 executor 保持不应用 action。
- 检查双相机、实测位姿、夹爪反馈和时间戳连续性。
- 检查 Policy 返回 `(6,8)`、`[6,12]`、zero queue 和 fresh generation。

### Stage B：受控六步运动

- 现场安全负责人放行后，只允许一个 six-action batch。
- 记录六个 command、六个 applied/reached state、相邻 applied-step interval 和 batch occupancy。
- `t5 - t0` 只有五个 15 Hz 间隔，目标约 `333 ms`；从首步 apply 开始到第六步 hold 完成才是完整六步占用时间，目标约 `400 ms`。
- 具体容差必须由机器人/organizer 契约给出并写入 manifest，不能用上述理论值自行放行。
- 六步结束后停止，确认没有旧计划第二批动作继续执行。
- 验证执行期间 RGB 持续采集，下一 packet 包含回填 history。

### Stage C：夹爪 roundtrip

- 使用现场批准的多个安全开度点。
- 同步记录 `model_gripper_width_m`、`bridge_gripper_open_ratio`、`robot_applied_open_ratio`、`robot_measured_width_m`。
- 任一点单位不明、饱和或 roundtrip 不一致即停止。

### Stage D：单次 Wipe episode

- prompt 必须是 `Wipe the marked area clean.`。
- 从 approved homing state 开始，只运行一个 episode。
- 不因“看起来有趋势”延长或重复实验；先按验收表复盘。
- 通过后再由测试负责人决定是否增加 episode，禁止自动连续重试。

## 10. 正常停止和紧急停止

- 正常停止模型端：前台 worker 使用 `Ctrl-C`，等待进程退出并保留日志。
- 机器人运动异常：由 onsite operator 使用机器人批准的 stop/E-stop；不要等待模型端退出。
- 停止后保存进程状态、最后观测、最后 command/applied/measured state 和现场视频。
- 不删除失败日志，不覆盖本次 evidence root。

## 11. 测试结束

1. 填写 `TEST_RECORD_TEMPLATE.md`。
2. 按 `ACCEPTANCE_CRITERIA.md` 标注证据等级和结果。
3. 若失败，填写 `INCIDENT_REPORT_TEMPLATE.md`，不要只发送截图或一句错误描述。
4. 对 evidence root 生成完整文件清单和 SHA-256；在安全存储中只读归档。

机器人端具体的 observe-only、homing、运动使能、normal stop、E-stop 和恢复命令不在模型仓库中臆造。测试前必须由机器人团队填写并签署 [ROBOT_SIDE_PROCEDURES.md](ROBOT_SIDE_PROCEDURES.md)。
