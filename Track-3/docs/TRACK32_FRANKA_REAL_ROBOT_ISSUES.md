# Track 3.2 Franka 真机测试：问题1与问题2

## 问题1：Action queue 导致视觉重规划严重滞后

### 当前行为

模型每次实际生成一个包含 12 个机器人动作的计划，但 Hub 每次只请求并返回
`chunk_size=1` 的一个动作。

剩余动作不会被删除，而是存入 Policy 的 `pending_actions` 队列。之后即使 Hub
送来了新的相机图像和机器人状态，只要队列尚未清空，Policy 就会继续返回旧计划
中的下一个动作，不会基于最新观测重新推理。

因此，`chunk_size=1` 只表示每次向 Hub 返回一个动作，不表示模型每执行一个动作
就重新规划一次。

### 日志证据

step7200 真机测试日志中：

```text
Hub infer completion：162 次
Infer One Chunk：14 次
```

二者满足：

```text
6 + 13 × 12 = 162
```

第一次冷启动计划只保留后 6 个动作，之后每次新计划保留 12 个动作。因此三分钟
测试中，虽然 Hub 获取了 162 个动作，模型实际上只进行了 14 次全新计划生成。

平均大约每 12.9 秒才重新规划一次。模型自身一次 transformer sampling 平均约
482 ms，因此三分钟只有 14 次新计划并不是模型单次推理需要十几秒，而是 action
queue 策略造成的。

### 影响

后续排队动作一直等待并依次发送，不会自动删除。动作队列末尾的指令可能仍然基于
十余秒前的图像和末端状态。

这会导致：

- 机械臂接近毛巾出现偏差后不能及时视觉纠正；
- 毛巾或夹爪位置变化后，Policy 仍继续执行旧计划；
- 抓取、接触和擦拭等需要闭环修正的动作容易失败；
- Hub 返回了很多动作，但其中大部分不是基于最新图像重新预测的动作。

### 建议修改

真机模式改为 receding-horizon：

1. 模型仍可生成原生 12-action 计划；
2. 每次只执行其中第一个有效动作；
3. 收到执行后的最新图像和机器人状态；
4. 删除尚未执行的旧计划动作；
5. 基于最新观测重新生成计划。

必须保证被删除、未执行的动作不会被记录为已经执行，也不会错误写入模型 cache。
如果当前 cache 不支持单步 grounding，应为真机增加明确的单步或 fresh-cache 模式，
而不是继续复用完整 12-action queue。

---

## 问题2：Gripper 的 `width_m` 与 `open_ratio` 单位契约不一致

### 当前行为

Franka 训练数据、normalizer 和模型 action 的 gripper 维度表示物理夹爪宽度：

```text
width_m，单位为米，通常约为 0.00～0.08 m
```

Challenge 官方真机接口使用：

```text
gripper_target_open_ratio，范围为 [0, 1]
```

当前 bridge 路径没有明确执行经过真机标定的 `width_m -> open_ratio` 转换，而是
可能把模型 action 的第 8 维直接写入官方 gripper command。

观测侧也直接使用 `joint_qpos[-1]` 作为当前 gripper 状态，但没有明确证明这个值
究竟是物理宽度 `width_m`，还是归一化的 `open_ratio`。

### 影响

物理宽度和归一化开合比例不是同一个量。如果直接传递：

- 模型预测的夹爪宽度会被官方接口按另一种尺度解释；
- 夹爪可能始终保持较大开度或无法正确闭合；
- 模型输入的 gripper feedback 与模型输出可能不在同一单位；
- 机械臂轨迹可以表现出接近毛巾的趋势，但夹爪仍无法完成抓取或接触。

真机界面持续显示 `0.8` 与单位转换问题相符，但现有日志还不能证明这个 `0.8`
具体由模型、bridge、机器人控制端还是 UI 的哪一层产生。

### 建议修改

对真实 Franka 夹爪进行标定，显式实现双向转换：

```text
open_ratio = (width_m - closed_width_m)
             / (open_width_m - closed_width_m)

width_m = closed_width_m
          + open_ratio × (open_width_m - closed_width_m)
```

转换结果需要裁剪到合法范围。如果官方机器人定义的开合方向相反，应在配置中明确
记录，而不是在不同代码位置隐式执行 `1 - ratio`。

每个控制周期至少记录以下四个值：

```text
model_gripper_width_m
bridge_gripper_open_ratio
robot_applied_open_ratio
robot_measured_width_m
```

只有完成 command/feedback roundtrip 后，才能确定真机显示的 `0.8` 来自哪一层，
并判断模型是否真正发出了闭合夹爪的意图。

---

## 当前结论

问题1是在线闭环问题：新观测已经到达，但旧 action queue 阻止模型及时重新规划。

问题2是执行接口问题：模型使用物理夹爪宽度，官方接口使用归一化开合比例，两者
缺少明确、可验证的转换。

两者都可能导致机械臂表现出一定任务趋势，却无法正确接近、抓住或操作毛巾。

当前 `cam_left_wrist`、`left_end_pose` 与 `control_arm="right"` 的单臂兼容映射先
保持不变，只增加必要日志，不作为本文件中的第三个主要问题。
