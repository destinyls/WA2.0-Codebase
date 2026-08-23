# Robot-side 现场流程确认书

本文件必须由机器人平台负责人填写。模型交付方不臆造私有机器人命令；任何空白项都会阻止进入带物体 Wipe。

## 平台身份

| 字段 | 现场填写 |
|---|---|
| 机器人/控制器型号与 ID | |
| executor 版本或 revision | |
| operator | |
| safety owner | |
| 日期/时区 | |

## 必填流程

| 流程 | 已批准命令/物理操作 | 成功信号与证据路径 | 责任人 |
|---|---|---|---|
| observe-only：接收 observation、禁止应用 action | | | |
| approved homing：不得使用模型全零末端 action | | | |
| single-batch enable：只允许恰好六步 | | | |
| normal stop | | | |
| physical E-stop | | | |
| E-stop 后检查、恢复和重新使能 | | | |
| robot command/applied/measured trace 导出 | | | |

## Timing 契约

| 字段 | 现场填写 |
|---|---|
| requested action rate | `15 Hz` |
| 允许的 applied-step interval 范围 | |
| 允许的 six-action batch occupancy 范围 | |
| 超时/漏步/重复步时 executor 行为 | |
| 收到 stop 后允许的最大残余动作数 | 必须为 `0`，否则不得测试 |

## 现场演练

- [ ] observe-only 下 Policy response 未驱动机器人。
- [ ] homing 成功，实测末端位姿、关节和夹爪状态合理。
- [ ] single-batch enable 只应用六个 action，未继续消费旧计划。
- [ ] normal stop 后没有残余 action。
- [ ] E-stop 已由 onsite operator 实际演练。
- [ ] 机器人 trace 可同时关联 command、applied、measured state 和绝对时间。

## 签字

- Robot platform owner：`____________`
- Onsite operator：`____________`
- Safety owner：`____________`
- 日期/时区：`____________`
- 允许阶段：`NO MOTION / CONTROLLED MOTION / WIPE`
