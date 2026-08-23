# Franka 真机测试交接包

## 交接目标

本目录供不参与模型开发的测试人员独立完成 WorldArena Track 3.2 Franka Wipe 真机联调、受控测试、证据采集和故障回传。

测试目标不是直接证明官方任务成功，而是依次确认：

1. 运行的是指定代码、Wipe 权重、serve bundle 和 schema-v4 Policy 配置；
2. 输入为最新实测位姿和持续采集的双相机 10 Hz causal history；
3. 每轮仅执行模型 future `[6:12]` 的六个绝对 XYZW actions；
4. 六步后使用新观测重新规划，不执行旧计划的第二批动作；
5. 夹爪在模型侧使用 `width_m`，在官方 wire 侧使用校准后的 `open_ratio`；
6. 完整保存模型端、Hub 端和机器人端 timing/action/observation 证据。

## 当前交付状态

| 项目 | 当前状态 | 交付要求 |
|---|---|---|
| future6/fresh-replan 代码 | 已实现，定向测试 `29 passed` | 必须提交到一个 clean Git revision 后填写 manifest |
| 源码 revision | 不在模板中预填 | 必须填写目标 GitHub 分支上的最终 clean commit，且该 commit 包含 schema-v4 与本交接包 |
| Wipe checkpoint 候选 | `/mnt/data/task/n0_twam_track32_franka_xyzw_20260817_v1/runs/franka_xyzw_wipe_2n16_step10000_20260822_v55-currentstate-fit/checkpoints/checkpoint_step_10000` | 交付方须在服务器严格验证并填写 checkpoint identity |
| serve bundle | 尚未在本交接包中绑定 | 交付方必须填写绝对路径和 receipt SHA-256 |
| Policy config | schema-v4 模板已提供 | 使用现场安全参数生成最终文件并记录 SHA-256 |
| WorldArena bridge | revision/hash 已固定 | 接手人必须运行 bridge audit |
| 私有机器人 executor 15 Hz | 模型端无法证明 | 必须通过机器人 timing trace 验证 |
| 真机任务成功 | 未验证 | 只能由实际受控实验或官方结果确认 |

在源码/模型身份、安全参数和责任人等字段填完之前，本目录是“准备完成但未封版”的交接包，不得启动运动。机器人 cadence 实测字段可在唯一一次无物体、单 six-action batch 的 Stage B 标定前写为 `PENDING_STAGE_B`；进入带物体 Wipe 前必须替换为实测值，并由现场安全负责人签字。

## 文件顺序

1. [HANDOFF_MANIFEST.md](HANDOFF_MANIFEST.md)：交付身份、权重、配置、人员和证据目录。
2. [PRE_TEST_CHECKLIST.md](PRE_TEST_CHECKLIST.md)：上线前逐项签字检查。
3. [RUNBOOK.md](RUNBOOK.md)：从软件 smoke 到受控 Wipe 的标准命令。
4. [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md)：通过、停止和证据分级标准。
5. [TEST_RECORD_TEMPLATE.md](TEST_RECORD_TEMPLATE.md)：每次测试记录模板。
6. [INCIDENT_REPORT_TEMPLATE.md](INCIDENT_REPORT_TEMPLATE.md)：失败时的最小回传信息。
7. [runtime.env.example](runtime.env.example)：不含密钥的环境变量模板。
8. [policy.schema-v4.example.json](policy.schema-v4.example.json)：不可直接运行的 Policy 配置模板。
9. [ROBOT_SIDE_PROCEDURES.md](ROBOT_SIDE_PROCEDURES.md)：由机器人团队填写并签字的 homing、使能、停止和 E-stop 流程。
10. [WIRE_CONTRACT.md](WIRE_CONTRACT.md)：相机、状态、future action、timing 和夹爪单位的逐字段契约。

更完整的实现说明见 [训练对齐部署文档](../../docs/TRACK32_FRANKA_TRAINING_ALIGNED_REAL_ROBOT.md)。

## 五条禁止事项

- 不得使用任何早于 schema-v4/future6 修复的旧源码 revision。
- 不得复用 schema-v2/v3 的旧生产配置、旧 serve output 或旧测试日志。
- 不得把模型全零 action 当作 homing；homing 必须由已批准的机器人控制流程完成。
- 不得在缺少 onsite operator、E-stop 和现场安全边界时启动运动。
- 不得把 CODE/SMOKE、离线 regression 或一次可运动现象写成真机任务成功率或官方成绩。
