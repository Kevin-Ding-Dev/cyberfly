# 行走与强化学习

先按 [安装说明](../README.md#安装) 配置环境。以下命令均在项目根目录执行，
并假定已激活 `.venv`。生成的模型、报告和视频不随源码分发。

## 1. 基线行走

### 1.1 检查环境接口

```bash
python -m cyberfly check
```

预期最后显示 `PASS`。它实际运行 MuJoCo，检查 Gymnasium 和 SB3 的接口、
随机种子复现、有限数值、回合到时结束和非法动作拦截。
Gymnasium 可能提示 observation bounds 为 infinity，这是速度等连续观测没有硬上限
导致的提示；只要最后是 PASS 就通过。它与 NaN/数值发散不同。

### 1.2 打开果蝇行走窗口

```bash
mjpython -m cyberfly walk --viewer --seconds 5 --output outputs/my-walk.json
```

你应该看到果蝇在棋盘格地面上行走，镜头跟随身体。这里是 **5 秒模拟时间**，
本机执行可能比 5 秒慢。运行结束后窗口自动关闭；提前关窗也会退出。
`completed: false` 表示手动提前关闭，不能当成完整回合的验收结果。

通过标准：完整结束，输出包含 `completed: true`、`fallen: false` 和正的 `distance_mm`。
`policy: fixed CPG baseline` 表示使用固定步态控制器，尚未使用学习策略。

macOS 上交互窗口必须通过 `mjpython` 启动。无窗口训练用普通 `python`。

### 1.3 保存视频和基线指标

```bash
python -m cyberfly walk --seconds 2 --video outputs/my-baseline.mp4 --output outputs/my-baseline.json
open outputs/my-baseline.mp4
```

视频保存在 `outputs/my-baseline.mp4`，
指标在同目录的 `my-baseline.json`。这是固定控制器的视觉检查；正式比较会使用
多个相同随机种子、相同任务指令，避免拿不同条件下的两段视频判断好坏。

## 2. 强化学习

### 2.1 理解当前任务

- 物理步长：0.0001 秒；控制器每个物理步更新一次。
- 神经网络每 0.02 秒产生一次动作：50 Hz，一个 RL step 含 200 个物理步。
- 默认回合：2 秒模拟时间，即 100 个 RL steps；翻倒时提前结束。
- 动作：左右两个数，各在 [-1, 1]，映射为 [0.6, 1.4] 的步态驱动幅度。
- 33 个观测：滤波后的身体速度、角速度、姿态、目标速度与方向误差、
  步态相位与幅度、上一次动作。观测使用固定尺度，不依赖额外的归一化统计文件。
- 平地任务 `straight`：每回合随机目标速度 8–12 mm/s，目标方向为起始方向。
- 速度误差先经过 100 ms 的速度滤波，以减少足部落地瞬间的尖峰影响。
- 奖励：速度跟踪、方向和身体直立，扣除动作变化代价；翻倒额外扣分。
  动作变化代价只表示平滑性，不能当作真实能耗。

### 2.2 可选：先做一次很短的训练管线测试

```bash
python -m cyberfly train --run runs/my-smoke --steps 64 --rollout-steps 64 --episode-seconds 0.2 --eval-every 64 --eval-episodes 1
```

通过标准：出现 `train/n_updates`，最后出现 `Saved .../latest.zip`。
0.2 秒回合和 64 步只用于检查代码通路，不用于判断长期行走表现。
正式训练应使用下面的新目录和默认 2 秒回合，不要从 smoke 模型接着训练。

### 2.3 启动第一轮正式任务训练

```bash
python -m cyberfly train --run runs/my-first --steps 2048 --task straight
```

训练首先评估未训练网络，再收集经历、更新 PPO，最后再次评估并保存。
`--steps` 是本次新增的 RL 步数，不是物理步数。
PPO 按完整 rollout 更新；默认每批 256 步，不整除的请求会向上补足。
2048 步约包含 409600 个训练物理步，另有初始化和评估开销。

通过标准：

1. 日志显示 `Using cpu device`。
2. `total_timesteps` 增长到 2048，`train/n_updates` 大于 0。
3. 验证日志 `VALIDATION` 出现，程序正常保存模型。

这些标准证明训练系统正常，并不证明 2048 步就足以学好。
`fps` 指 RL 步/墙钟秒，不是视频帧率。

### 2.4 看懂保存结果

`runs/my-first/` 中包括：

- `initial.zip`：这一轮开始时的策略。
- `latest.zip`：最新策略、优化器状态、训练步数和任务配置。
- `best.zip`：**本轮**固定验证集上平均奖励最高的策略，可能仍是初始策略。
- `best_validation.json`：这个 best 的验证指标和训练步数。
- `checkpoints/`：每次验证时的历史快照。
- `config.json`：任务参数、命令参数、平台和实际依赖版本。
- `monitor.csv`：训练回合的奖励、长度和时间。
- `validation.jsonl`：固定验证种子 10000 起的评估记录。
- `summary.json`：本轮新增步数、总步数、耗时、是否被中断。
- `tb/`：TensorBoard 日志。

输出目录必须是新目录或空目录，程序会拒绝覆盖已有实验。

### 2.5 播放学习到的策略

```bash
mjpython -m cyberfly walk --model runs/my-first/best.zip --viewer --output outputs/my-policy.json
```

或者保存视频：

```bash
python -m cyberfly walk --model runs/my-first/best.zip --video outputs/my-policy.mp4 --output outputs/my-policy.json
open outputs/my-policy.mp4
```

加载模型后会使用模型保存的任务配置，包括回合时长；`--seconds` 只用于固定基线。
`best.zip` 的动作来自人工神经网络，但低层步态仍由 FlyGym 控制器产生。
查看输出的 `training_timesteps`：若为 0，说明该神经网络仍是初始权重。

## 3. 评估与恢复

### 3.1 用独立测试回合判断改善

```bash
python -m cyberfly evaluate --model runs/my-first/best.zip --reference runs/my-first/initial.zip --episodes 10 --output outputs/my-first-evaluation.json
```

程序比较固定步态基线、初始神经网络和选出的策略。三者使用相同测试种子和命令；
默认种子从 20000 开始，与训练时选模型使用的验证集不同。

关注以下输出：

- `velocity_error`：平均平面速度向量误差，单位 mm/s，越小越好。
- `heading_error`：平均朝向误差，单位度，越小越好。
- `falls`：翻倒比例，越低越好。
- `return`：组合奖励，越高越好，但应结合前三项判断。

记录每个回合的数据，避免只比较某一次成功的演示。
少量测试不能保证泛化；参数调优用训练/验证记录，最终测试可以更换一组新的
`--seed-start`，并用多个训练随机种子重复实验。
程序不会仅因奖励上涨就宣称整体性能改善。

### 3.2 打开训练曲线

另开一个终端：

```bash
# 在项目根目录运行
source .venv/bin/activate
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

浏览器访问 <http://127.0.0.1:6006>。
重点看 `validation/mean_velocity_error_mm_s`、`validation/fall_rate`、
`validation/mean_return`；`rollout/ep_rew_mean` 只是训练环境回报。
在该终端按 Ctrl+C 关闭 TensorBoard。

也可以在训练继续运行时，另一个终端用 `mjpython -m cyberfly walk --model ... --viewer`
播放已保存的检查点。播放器启动时读取一次模型，不会每帧自动切换权重；
下一次重新运行播放命令就会读取新保存的版本。并行播放会占用 CPU，可能降低训练速度。

### 3.3 从检查点继续训练

```bash
python -m cyberfly train --resume runs/my-first/latest.zip --run runs/my-second --steps 10000
```

模型和优化器状态会保留，累计步数继续增长；10000 表示本次追加，实际会补齐
到整批 rollout。新的输出写到 `my-second`，旧模型保持可比较。
如需从本轮验证最好的版本开始，可把 `latest.zip` 改成 `best.zip`。

恢复时继承检查点的网络、PPO 参数和任务配置；它会开始新的仿真回合。
这不是逐位完全一致的中断恢复：未完成 rollout、当时的物理状态与所有随机数状态
不做完整快照。采样可能少量重做，但已保存的网络学习成果可以继续使用。

### 3.4 再比较新旧模型

```bash
python -m cyberfly evaluate --model runs/my-second/best.zip --reference runs/my-first/best.zip --episodes 10 --output outputs/my-second-evaluation.json
```

`my-second/best.zip` 只是在第二轮内部最好，不能据此认为它必然优于第一轮。
若速度误差、翻倒率或方向误差变差，保留旧版本，检查奖励和训练设置。
持续学习按“训练一段 → 验证 → 独立比较 → 决定下一轮”的节奏进行。

### 3.5 暂停和再次启动

在训练终端按一次 Ctrl+C，等待程序打印保存路径并退出，然后从新目录恢复。
强制关闭终端、系统断电或 kill -9 不能保证保存当时的最新状态；这种情况使用最近
成功写入的检查点。训练时保持 Mac 唤醒；也可在命令前加 `caffeinate -i`：

```bash
caffeinate -i python -m cyberfly train --resume runs/my-second/latest.zip --run runs/my-third --steps 10000
```

## 后续：转向任务

直行和速度跟踪验证后，可以启动一个新转向实验：

```bash
python -m cyberfly train --run runs/my-turn --steps 10000 --task turn
```

该任务每回合增加 ±0.35 rad（约 ±20°）的方向目标。当前实现会从新网络开始，
不会自动把直行策略迁移成转向策略。`--resume` 要求同一任务配置，避免偷偷改变
奖励和评估条件。简化视觉、味觉与嗅觉接口见 [觅食指南](../FORAGING.md)。复杂地形、真实感觉回路、
飞行与低层关节学习仍需单独实现和验证。

## 常见问题

- `No module named cyberfly`：先进入包含 `cyberfly/` 的项目根目录，再运行命令。
- `mjpython: command not found`：重新 `source .venv/bin/activate`。
- `launch_passive ... mjpython`：交互窗口改用 `mjpython -m cyberfly ...`。
- `invalid CoreGraphics connection`：从已登录桌面的 macOS 终端运行图形命令；
  受限后台执行环境可能无法访问图形服务。无窗口训练不需要此服务。
- `Output directory is not empty`：换新的 `--run` 路径；继续学习时同时使用 `--resume`。
- 训练开始后暂时没有步数增长：程序正在编译模型或做开始前的独立验证。
- `best.zip` 对应 0 步：后续验证没有超过初始策略，不能称为学习成功。
- 学习结果没有改善：增加预算前先检查基线、验证指标和任务难度；继续训练不保证单调变好。
- 标准模型网格随 FlyGym 包提供；这里不需要下载完整分辨率或 FlyBody 的额外网格。
- 不要将 Linux 教程里的 `MUJOCO_GL=egl` 或 CUDA 设置直接套到此 Mac 环境。

## 实现文件与来源

- `cyberfly/env.py`：任务定义、单位、观测、奖励、终止规则和渲染。
- `cyberfly/learning.py`：PPO、检查点、验证和测试比较。
- `cyberfly/__main__.py`：命令行入口。
- `cyberfly/checks.py`：实际仿真接口和可复现性检查。
- `requirements.txt`：基础版本要求。
- `requirements-macos.lock.txt`：macOS ARM64 / Python 3.12 环境的依赖版本快照。

参考：[FlyGym 行走控制](https://neuromechfly.org/tutorials/4c_hybrid_controller/)、
[FlyGym 2.1.0](https://github.com/NeLy-EPFL/flygym/tree/v2.1.0)、
[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、
[MuJoCo Python viewer](https://mujoco.readthedocs.io/en/latest/python.html#passive-viewer)。
