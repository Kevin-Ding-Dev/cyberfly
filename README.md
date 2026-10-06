# Cyberfly

**果蝇运动、糖源觅食与强化学习的功能仿真。**

Cyberfly combines the FlyGym / NeuroMechFly body and locomotion controller with
MuJoCo physics and PPO learning. It supports walking, randomized sugar foraging,
functional sensory and feeding control, fixed-camera videos, and a training loop
that evaluates candidates before replacing the current policy.

这是面向研究与实验的功能模型。高层人工神经网络学习行走和口器控制修正，
低层步态由 FlyGym 控制器提供；并未实现完整果蝇大脑、真实连接组或飞行。
没有匹配的真实实验数据时，只报告任务表现，不报告生物行为相似度。
强化学习不保证每轮提高表现。


https://github.com/user-attachments/assets/dd176c66-52a1-46ec-aa55-e648799d7ff0


## 功能

- 42 自由度身体的平地行走、速度跟踪与转向任务。
- 随机数量与位置的糖源；可选 0.1 µL 蔗糖液滴几何预设。
- 简化双眼方向感知、足部/口器味觉、口器伸缩、摄入与饱足反馈。
- 工程化感觉控制器，以及基于短窗口 GRU 特征的 PPO 修正策略。
- 候选训练、成对验证、晋升门槛、检查点与中断恢复。
- 固定场地相机录制，以及按糖源数量和随机种子批量测试。
- 单糖源平面迷宫、沿通路衰减的双触角气味、墙体遮挡、探索记忆与死胡同返回。
- 可记录的多巴胺奖励预测误差代理，以及按实际负载选择 CPU / Apple MPS 的迷宫 PPO 训练。

## 安装

已测试环境：macOS / Apple Silicon（ARM64），Python 3.12，FlyGym 2.1.0，
MuJoCo 3.9.0。原行走/平地觅食训练使用 CPU；新迷宫训练默认实测负载后选择 CPU 或 Apple MPS GPU，
MuJoCo 物理仍由 CPU 执行，不需要 CUDA。
其他操作系统和 Python 版本尚未验证。

克隆或下载本仓库后，在包含 `requirements.txt` 的项目根目录运行。
需要预先安装原生 ARM64 的 Python 3.12：

```bash
python3.12 -c "import platform; print(platform.machine())"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-macos.lock.txt
python -m pip check
```

Apple Silicon 上第一条命令应输出 `arm64`。版本快照锁定了用于验证的依赖；
首次安装需要网络。运行代码时应留在项目根目录，无需执行 `pip install -e .`。
默认简化身体网格由 FlyGym 提供；完整分辨率或其他身体模型可能需要上游资源下载。
本仓库不附带这些第三方资产。

验证环境：

```bash
python -m unittest discover -s tests -v
python -m cyberfly check
python -m cyberfly forage-check
```

单元测试应报告 `OK`，两个仿真检查命令应报告 `PASS`。
检查包含实际物理仿真，耗时取决于机器与负载。

## 快速开始

无需训练即可观看工程化感觉控制器接近并摄入小型糖液滴：

```bash
mjpython -m cyberfly forage --config configs/forage-droplets.json --seed 0 --viewer --output outputs/forage-demo
```

macOS 交互窗口必须使用 `mjpython`。训练、无窗口测试与视频录制使用 `python`。
糖源吃完、回合到时或提前终止后，窗口会关闭。

保存固定相机视频：

```bash
python -m cyberfly forage --config configs/forage-droplets.json --seed 12 --video outputs/forage-demo.mp4 --output outputs/forage-video
open outputs/forage-demo.mp4
```

创建觅食训练实验，运行 3 轮候选训练与验证：

```bash
python -m cyberfly forage-loop --run runs/forage --config configs/forage-droplets.json --rounds 3 --steps 2048 --eval-episodes 8
```

每轮候选通过门槛后才替换当前模型。若仍使用 `initial.zip`，表示还没有候选
通过验证，不能把训练完成解释为行为改善。再次使用同一目录可继续训练；
改为 `--rounds 0` 会持续运行至按 Control+C。

默认策略对口器只作小幅修正，无法覆盖基线的开闭决定。若希望网络学习开闭口器，
新建训练目录并改用 `configs/forage-mouth-control.json`；具体操作与兼容规则见
[口器自主控制配置](FORAGING.md#口器自主控制配置)。

录制当前通过验证的模型，覆盖不同数量与位置：

```bash
python -m cyberfly forage-batch --model runs/forage --food-counts 2 5 8 --seeds 12 13 14 --camera-distance 36 --output outputs/forage-batch
```

该命令固定一份模型快照，生成 9 段视频及逐场报告。8 个糖源超出默认训练数量
范围；这是额外测试，不代表已经证明泛化能力。饱足反馈可能使果蝇留下部分糖源。

## 迷宫与 Apple GPU 训练

每局一个糖源，放在沿通路最远且可达的位置。三类迷宫包含墙体碰撞、视觉遮挡、
沿通路衰减的气味、探索与返回记忆，以及计算性的多巴胺奖励预测误差。

```bash
python -m cyberfly maze-train --run runs/maze-mps --config configs/maze.json --device auto --envs 4 --steps 16384
```

再次运行会恢复最新模型并累计训练；`--steps 0` 持续至 Control+C。
新任务使用独立检查点。自动模式比较推理、PPO 更新和物理负载，只有 MPS 预计总耗时
至少降低 10% 才启用；否则选 CPU。可用 `--device cpu` / `--device mps` 手动覆盖。
物理仿真始终在 CPU。短训练、设备报告、观看与录制见 [迷宫指南](MAZE.md)。

## 文档

- [迷宫、气味、多巴胺代理与 Apple GPU 训练](MAZE.md)
- [觅食、持续训练、视频与科研依据](FORAGING.md)
- [行走与强化学习](docs/WALKING.md)
- [验证与复现](docs/VALIDATION.md)
- [贡献指南](CONTRIBUTING.md)
- [第三方软件与资产说明](THIRD_PARTY_NOTICES.md)

运行 `python -m cyberfly --help` 查看命令列表；子命令也支持 `--help`。

## 模型边界

视觉使用几何计算的角度通道，预先知道哪些物体是糖源，没有真实图像识别与遮挡。
口器和味觉使用功能性邻近检测，没有液体流动或吞咽肌肉模型。
液滴几何体积、归一化摄入量和内部能量是不同量；摄入速度与饱足未按真实生理标定。
GRU 只处理最近 4 帧感觉，约 80 ms，不能视为完整果蝇记忆或全脑模型。
迷宫另用感觉窗口网络和显式探索记忆，视觉包含几何深度射线及墙体遮挡；
气味场、多巴胺代理和理想化里程计仍未按生理数据校准。

## 文件与许可证

`cyberfly/` 包含环境、感觉控制器、PPO 和检查代码，`configs/` 提供任务预设。
`runs/`、`outputs/`、虚拟环境和缓存由本机生成并被 Git 忽略；源码仓库不含预训练
模型、个人实验记录或视频。实验报告和模型可能包含本机路径，对外分发前需检查元数据。

Cyberfly 源码采用 [Apache-2.0](LICENSE)，署名见 [NOTICE](NOTICE)。
依赖和模型资产适用各自的许可证，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
