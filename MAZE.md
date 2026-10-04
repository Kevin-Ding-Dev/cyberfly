# 单糖源迷宫、气味搜索与 Apple GPU 训练

此任务在原有 FlyGym 身体、MuJoCo 接触物理和可伸缩口器上增加迷宫。
每局只有一个 0.1 µL 几何体积的蔗糖液滴；入口固定在第一格，目标放在从入口
沿通路计算的最远格子。最远位置并非画面上直线距离最远的位置。
生成器检查所有格子连通，因此至少存在一条到糖源的通路。

这是可运行、可训练的功能模型。身体运动和摄入需要实际推进 MuJoCo；
感觉导航、多巴胺信号和探索记忆仍是工程模型，没有复现完整果蝇神经连接组。

## 1. 先打开一个迷宫

在项目根目录、已安装依赖的终端运行：

```bash
source .venv/bin/activate
mjpython -m cyberfly maze --config configs/maze-corridor.json --seed 0 --viewer --output outputs/maze-corridor-demo
```

这会打开一个有转弯的 2×2 平面迷宫。蓝灰色实体是墙，浅黄色液滴是唯一糖源。
果蝇需要经过通路、用嘴接触糖源并完成摄入；走到附近本身不会获得吃完的奖励。
全部摄入、跌倒、越界、能量耗尽或时间用尽会结束回合。
墙体与身体之间启用独立碰撞；检测到躯干穿越墙体、越出场地或升到墙顶以上，
也会按违反平面迷宫边界终止，不能通过爬越或穿墙获得目标奖励。

换成其他布局：

```bash
mjpython -m cyberfly maze --config configs/maze-branching.json --seed 0 --viewer --output outputs/maze-branching-demo
mjpython -m cyberfly maze --config configs/maze-looped.json --seed 0 --viewer --output outputs/maze-looped-demo
```

- `maze-corridor.json`：2×2 曲折单通道，用于先检查转弯与取食。
- `maze-branching.json`：3×3 随机树形迷宫，可包含分支和死胡同。
- `maze-looped.json`：3×3 在连通树上增加通道，产生环路。
- `maze.json`：3×3，每次重置从三类迷宫中随机选择；正式训练默认使用这一配置。
- `maze-smoke.json`：1×2 直通道、最多 6 秒，用于验证训练流程，不用于证明迷宫智能。

`--seed` 同时控制迷宫、最远位置平局时的选择和身体初始化。
`corridor` 的蛇形连接固定，改变种子主要改变身体初始化；另外两类还会改变通道布局。
所有布局都只有一个糖源。默认格子边长 8 mm、墙厚 0.8 mm、墙高 3 mm。

在同一个窗口连续测试不同种子：

```bash
mjpython -m cyberfly maze --config configs/maze.json --seed 0 --episodes 0 --viewer --output outputs/maze-continuous-view
```

`--episodes 0` 表示持续运行，种子依次递增；也可改为 `--episodes 10`。
关闭窗口或在终端按 Control+C 停止。多回合报告放在各自的 `episode_*` 子目录。
重复使用输出目录及相同种子会覆盖对应的演示报告；保留历史时换一个目录名。
观看只执行策略，不更新神经网络。未提供 `--model` 时使用感觉与探索规则基线。

## 2. 按实际负载选择 CPU 或 Apple GPU

训练默认使用 `--device auto`，每次启动或续训进行一次负载校准：

1. 检查 MPS 是否可用，并实际执行 GPU 反向传播。
2. 用当前并行环境采样物理、传感器与进程通信的耗时。
3. 在独立模型副本上，按当前观察维度、环境数量、rollout 长度、批量和训练轮数，
   分别测量 CPU/MPS 的策略推理、数据传输及实际 PPO 更新。
4. 预热后各测三次取中位数，估算一轮“收集 + 更新”的耗时。
   **只有 MPS 的预计总耗时至少降低 10% 才选择 MPS，否则选择 CPU。**

更新基准使用采样观察和合成优势值，只衡量计算负载，不评估行为学习。
校准副本及环境采样不计入正式训练步数，也不改变正式模型权重和优化器状态。
正式训练会重新设置种子并重置环境。首次使用某种 GPU 运算时，预热可能稍久。

`state.json` 的 `device` 保存选择原因、实际选择、各项测量和预计耗时。
`last_device_report` 保存正式训练的参数、梯度和 Adam 动量所在设备。
短基准是当前机器负载下的估计，不包含完整验证、保存耗时，也不保证长期速度；
训练中保持设备不变，下一次启动或续训重新测量，不在更新中途来回迁移优化器。

可以手动覆盖：

- `--device auto`：根据负载自动选择，推荐；MPS 不可用或运算失败时记录原因并选 CPU。
- `--device cpu`：直接使用 CPU，不运行比较基准。
- `--device mps`：直接要求 Apple GPU，不运行比较基准；不可用会报错。

若设置了 `PYTORCH_ENABLE_MPS_FALLBACK=1`，自动模式会选 CPU，强制 MPS 会报错，
避免部分运算落在 CPU 却被误计为 GPU 性能。要参与 MPS 比较，先运行：

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
```

可选的 GPU 能力检查（不代表训练最终应选 GPU）：

```bash
python -m cyberfly maze-device --device mps
```

成功时会显示 `mps_available: true` 和 `backward_probe_passed: true`。
如果受限应用环境检测不到 GPU，可在 macOS 系统终端执行；不要混用 Rosetta x86
Python 和原生 ARM64 环境。当前小型策略和少量并行环境选择 CPU 是正常结果。

MuJoCo 物理始终由 CPU 执行。MPS 只负责策略网络与优化器，不能把整个物理环境
搬到 GPU。SB3 在 MPS 基准阶段可能提示非 CNN 策略的 GPU 利用率较低；
最终选择请看 `device_check` 中的 `policy_device`，不是基准期间出现过的设备名称。

## 3. 先完成一次短训练

```bash
python -m cyberfly maze-train --run runs/maze-device-check --config configs/maze-smoke.json --device auto --envs 1 --rollout-steps 128 --steps 128 --eval-every 128 --eval-episodes 1
```

程序会打印设备选择和测量结果，随后生成检查点、评估报告和 `state.json`。
设备报告中的 `parameter_devices`、`gradient_devices`、`optimizer_moment_devices`
应全部一致，为 `["cpu"]` 或 `["mps"]`。

专门验证 GPU 训练及跨设备续训时，可在完成上面命令后，将 `--device auto`
改成 `--device mps` 再执行；其余参数保持一致。再改回 `auto` 可验证自动重选。
设备可以改变，模型与优化器状态保留；设备间数值与随机轨迹不保证完全相同。

短训练证明接口、反向传播、更新和保存能运行；它的简单场景成功率不代表复杂迷宫
学习已经成功，也不代表神经网络比基线更好。

## 4. 开始正式训练

```bash
python -m cyberfly maze-train --run runs/maze-mps --config configs/maze.json --device auto --envs 4 --steps 16384
```

默认四个 CPU 物理环境，策略网络根据负载使用 CPU 或 MPS，每局随机选择迷宫类别和布局。
每个环境收集 256 步后做一轮 PPO 更新，因此一次更新包含 1,024 个环境步。
`--steps` 和 `--eval-every` 必须是 `envs × rollout-steps` 的整数倍。
默认每新增 8,192 步，在 6 个固定验证种子上评估一次。

训练会持续积累最新模型的经验；某次评估没有刷新最好成绩，也不会把下一轮的
学习退回初始模型。最好的已评估模型和用于继续训练的最新模型分别保存。

需要持续训练到手动停止：

```bash
python -m cyberfly maze-train --run runs/maze-mps --config configs/maze.json --device auto --envs 4 --steps 0
```

按一次 Control+C 后等待保存完成。再次执行同一命令会加载 `latest.zip`
和优化器状态，继续累计步数。中断时尚未完成的 rollout 可能不参与梯度更新，
恢复不承诺随机数流或物理轨迹逐位接续。

已有训练目录必须保持任务配置、环境数量、每次收集步数、基础种子和验证场次数一致。
想改变迷宫大小、奖励或控制范围时，使用新训练目录。原行走、平地觅食模型的
观察与动作维度不同，不能直接作为迷宫模型加载。

主要文件：

- `state.json`：训练状态、累计步数、配置、最佳检查点指针和设备检查。
- `initial.zip`：零修正的初始策略，保留用于对照。
- `latest.zip`：最近的策略与优化器状态，用于继续训练；可能尚未验证。
- `checkpoints/step_*.zip`：完成训练区段后保存的固定检查点。
- `checkpoints/step_*.evaluation.json`：对应检查点的逐场验证、设备和文件指纹。
- `tb/`：TensorBoard 日志。

验证按成功率、平均摄入比例、跌倒率、平均用时和累计奖励依次择优。
第一次验证会建立首个已评估候选基准；它被选中不表示已经胜过初始策略。
这是有限验证集上的模型选择，不保证每次更新进步，也不是严格的生物相似性检验。

## 5. 查看训练模型和独立测试

查看当前已评估的最好模型：

```bash
mjpython -m cyberfly maze --model runs/maze-mps --seed 12 --viewer --output outputs/maze-trained-demo
```

查看最新模型（可能尚未验证）：

```bash
mjpython -m cyberfly maze --model runs/maze-mps/latest.zip --seed 12 --viewer --output outputs/maze-latest-demo
```

模型包含完整任务配置，查看时不要同时传 `--config`。
首次验证完成前，训练目录仍指向 `initial.zip`；`training_timesteps: 0` 表示零修正基线。
观看和独立评估默认使用 CPU，适合逐个环境的小批量推理；评估仍可显式指定 `--device mps`。

```bash
python -m cyberfly maze-evaluate --model runs/maze-mps --device cpu --episodes 12 --seed-start 300000 --output outputs/maze-independent-evaluation.json
```

这些种子与训练过程中固定验证所用的 `200000…` 范围不同。
报告保存实际加载的检查点 SHA-256，避免训练期间更换模型导致报告署名混淆。
应检查每个种子的成功、跌倒、用时与覆盖格子数；少量成功场景不代表任意迷宫都成功。

录制一场固定相机视频：

```bash
python -m cyberfly maze --model runs/maze-mps --seed 12 --video outputs/maze-trained.mp4 --output outputs/maze-trained-video
open outputs/maze-trained.mp4
```

也可以去掉 `--model`，改用 `--config configs/maze-branching.json` 录制基线。
视频和 viewer 使用同一套初始场地相机；窗口中手动调整的相机不会保存到视频。
每两次控制步录一帧，25 fps。奇数步提前结束时保留最终状态，播放时间最多多 0.02 秒。
报告保存每帧的实际仿真时间戳。一段视频只对应一局，不能同时使用多回合 `--episodes`。

## 6. 感觉、返回判断与神经网络实际做什么

每个控制步长是 0.02 秒，包含 200 个 MuJoCo 物理小步。
每帧有 126 个观察值，默认保存 16 帧，即约 0.32 秒的感觉窗口。

- 双触角读取自身位置上的两个气味浓度。场地按可通行空间的距离构建衰减场，
  不使用穿过墙壁的直线距离。气味是与糖源配对的挥发性线索；纯蔗糖本身不是远距离气味源。
- 24 条水平视觉深度射线检测墙体，原有的双眼方向信号检测近处可见的糖源。
  墙会挡住糖源视觉。这是几何生成的功能视觉，没有从 RGB 图像识别食物或复现真实小眼光学。
- 基线导航器把自身速度与方向积分为位置估计，并根据射线建立已探索格子的记忆。
  它预先知道规则网格的格子尺度，使用理想化无噪声的自身运动与方向参考；这是较强的工程先验。
  它不会获得完整通道连接、目标坐标、目标格子编号或最短路线。
- 遇到岔路时，已测气味的空间/时间变化影响未探索方向的优先级。
  当前格子没有未探索出口时，导航器沿自己记录的进入路径返回。
  在狭窄死胡同中先反向步行退出，减少原地大幅掉头时身体扫到侧墙的风险。
  这属于基于局部感觉与探索记忆的返回规则，不是隐藏的全局寻路答案。
- 网络输出四个动作：左腿组修正、右腿组修正、口器开闭修正、岔路左右偏好。
  初始输出层为零，先保留可工作的导航基线；PPO 再学习调整运动、取食和岔路选择。
  最近感觉窗口用 MPS 支持的全连接层编码。较长的探索记忆来自显式导航器，并非无限长度循环网络。
- 接近目标后，仍要求嘴部接触、口器指令打开、速度足够低、身体直立且未饱足。
  食物库存真正减少才记摄入；吃完后移除糖源并结束回合。

因此，“能够解出迷宫”不能单独证明是强化学习带来的改善。
应把训练模型与 `initial.zip` 在相同、独立种子上对比，并做气味/视觉/记忆的消融实验。

## 7. 奖励与多巴胺代理

奖励有明确分项，全部写入报告：

- **气味推进**：`gamma × Phi(下一状态) − Phi(当前状态)`，默认 `Phi = 3 × 双触角平均气味`。
  向更强气味推进通常给正反馈；退回较弱气味给负反馈。
  这是带折扣的势函数塑形，反复往返或停在气味较强处不会不断赚取正奖励。
  真正终止时下一势函数置零；时间截断保留下一势函数以供价值自举。
  因为需要消除终点势函数，整局累计的气味分项可能为负，不表示吃到糖被判为失败。
- **真实摄入**：每摄入 1 个归一化食物单位奖励 8。
- **完成奖励**：唯一糖源全部吃完时额外奖励 15，只发放一次。
- **代价**：每秒 0.02 的时间代价、小幅动作代价，跌倒/越界有额外负反馈。

没有提供目标坐标奖励或最短路径监督。生成目标和事后计算覆盖率会使用环境真值，
这些信息只进入记录，不进入策略观察。

多巴胺模块以一个局内线性预测器估计预期价值，再计算时间差分误差。
正误差形成衰减的正向通道，负误差形成负向通道；两者和当前误差进入网络观察。
信号不再次叠加成另一份奖励，以免网络通过自产信号获得无来源的收益。
局内预测器和痕迹每回合重置；跨回合持久学习由保存的 PPO 参数承担。

这些数值没有真实浓度单位，也不能测量主观“快感”。模块仅表示糖摄入强化与
预期偏差的计算代理，未逐个复现 PAM/PPL1 神经元或蘑菇体的生理动力学。
默认时间常数、阈值和学习率是工程参数，尚未经真实果蝇数据标定。
糖液滴体积与归一化摄入量也尚未做生理摄食速率换算。

研究背景与软件依据：

- [Liu 等：果蝇气味记忆中的多巴胺奖励信号，Nature 2012](https://www.nature.com/articles/nature11304)。
- [Zolin 等：果蝇气味导航时多巴胺通路与运动、强化的关系，Nature Neuroscience 2021](https://www.nature.com/articles/s41593-021-00929-y)。
- [SB3 PPO 设备与并行环境建议](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)。
- [PyTorch MPS 后端说明](https://docs.pytorch.org/docs/2.14/notes/mps.html)。

上述研究支持建立奖励与气味导航模型的研究方向，不验证本项目全部方程和参数。

## 8. 验证命令

```bash
python -m unittest discover -s tests -v
python -m cyberfly maze-check
python -m cyberfly forage-check
python -m cyberfly check
```

测试覆盖连通性、最深目标、遮挡、气味沿通路传播、左右气味改变选路、死胡同返回、
动作接口、种子复现、奖励往返不能套利，以及真实 U 形迷宫中的运动与接触摄入。
物理回归还会主动选择一条错误分支，验证进入死胡同后能返回并最终吃完糖源。
另外强制让身体撞墙，检查确实产生 MuJoCo 接触且墙能阻挡运动，避免只有视觉墙体。
设备选择检查覆盖推理开销、物理瓶颈、相近耗时、MPS 不可用和不支持的运算，
并确认基准不修改原模型。需额外在目标机器上执行自动选择、强制 MPS 和跨设备续训；
普通单元测试不依赖本机 Metal 驱动。

`summary.json` 保存摄入、回合奖励分项、探索覆盖、返回次数、墙体接触次数和相机。
`trajectory.csv` 保存逐时刻位置、气味、多巴胺代理、口器与摄入；`events.json` 保存每次
实际摄入；`maze.json` 保存布局，供复查可达性。它们是本机产物，默认不纳入 Git。
