# 随机糖源觅食与持续训练

这是能在本机运行的地面觅食功能模型。它使用 FlyGym 身体、步态和接触反射，
加入感知、趋近、停止、伸出口器、摄入和饱足反馈。它没有实现完整果蝇大脑，
也不能保证每轮强化学习都会提高表现或生物真实性。

## 1. 运行觅食仿真

先完成 [安装](README.md#安装)，然后在 macOS 终端进入项目根目录并激活环境：

```bash
# 在项目根目录运行
source .venv/bin/activate
```

运行觅食检查：

```bash
python -m cyberfly forage-check
```

预期出现四组 `PASS`，包括真正运行物理仿真的走近、停止和摄入测试，
以及液滴的尺寸、体积和接触边界检查。

打开默认随机场景：

```bash
mjpython -m cyberfly forage --seed 0 --viewer --output outputs/my-forage-demo
```

macOS 交互窗口使用 `mjpython`。浅黄色圆片是糖源；绿点标出口器的功能接触点。
果蝇接近、停下取食，糖源被完全摄入后消失。模拟最多运行 15 秒，吃完全部糖源
也会提前结束。窗口随后关闭；模拟时间与实际等待时间不同。

参考基线测试中，seed=0 生成 4 个随机糖源，7.4 秒模拟时间内全部摄入，无跌倒或越界。
该行为来自工程化感觉控制器，不能用来证明神经网络已经学会觅食。
改成 `--seed 1`、`--seed 2` 可以换场景；并非所有随机场景都保证成功。

查看本次指标：

```bash
cat outputs/my-forage-demo/summary.json
```

完整成功回合应有 `complete: true`、`ate_all: true`、`fallen: false`。
手动提前关窗时 `complete` 为 false。`biological_similarity: null` 表示尚未测量
与真实实验的行为相似性。不是 0 分，也不是程序失败。

保存视频，训练和录制用普通 `python`：

```bash
python -m cyberfly forage --seed 0 --video outputs/my-foraging.mp4 --output outputs/my-foraging-video
open outputs/my-foraging.mp4
```

视频与 `--viewer` 共用固定场地相机：观察点 `[0, 0, 0]`、距离 30、
方位角 100°、俯仰角 −65°。相机不跟随果蝇身体，因此不会因行走起伏而抖动。
这匹配交互窗口刚打开时的默认视角；手动拖动、缩放窗口中的相机不会自动保存到视频。
窗口宽高比、界面面板和视频分辨率不同，边缘取景可能略有差异。

录像为 25 fps，每完成两个 0.02 秒控制步录一帧。完整 15 秒回合产生 375 帧，
播放时间为 15 秒。若回合在奇数控制步提前结束，会补录最终状态，播放时间比仿真
最多多 0.02 秒。`summary.json` 的 `video` 保存帧数、播放时间、时间差和每帧对应的
`frame_times_s`；精确分析使用这些仿真时间戳，不把视频帧序号当作采样时刻。

录制训练目录里当前通过验证的模型，并复现观看命令的 seed=12：

```bash
python -m cyberfly forage --model runs/my-forage --seed 12 --video outputs/forage-fixed-view.mp4 --output outputs/forage-fixed-view
open outputs/forage-fixed-view.mp4
```

如果训练仍在更新当前模型，两次运行可能加载不同的检查点。需要严格比较同一行为时，
可让两条命令都使用同一个具体的 `.zip` 检查点路径。

新报告的 `model_provenance` 保存实际加载文件的路径、SHA-256 和字节数。
加载与计算指纹使用同一份文件内容，因此即使训练晋升或文件被替换，报告仍标识
本次实际使用的模型。批量报告另存来源模型与批次快照各自的指纹；重新序列化后的
快照文件指纹可能不同，即便权重相同。文件路径本身不能证明两次使用了同一模型。

视频由上述命令生成，不随源码分发。每个演示输出目录还包含
`trajectory.csv`（轨迹、口器、摄入及能量）和 `events.json`（每次摄入的糖源编号、
时间、数量和口器位置）。重复使用相同演示目录会更新文件；需要保留时换目录名。

新生成的轨迹和指标采用版本 2，分别由 `trajectory_schema_version` 和
`metrics.schema_version` 标记。速度单位均为 mm/s：

- `trajectory.csv` 的 `speed_mm_s`：相邻两次位置之间的平面距离除以时间间隔；
  首行只是初始状态，其速度写为 0，不计入平均值。
- `filtered_speed_mm_s`：时间常数 0.1 秒的平面速度向量滤波后取模，供摄入和能量判断。
- `metrics.mean_speed_mm_s`：轨迹总长度除以回合时间，与生物比较中的同名特征一致。
- `metrics.mean_filtered_speed_mm_s`：各控制间隔的滤波速度均值，不含初始状态。

旧报告没有上述版本字段，旧的 `mean_speed_mm_s` 含义不同，不能与新指标混用。
历史文件保留；需要统一比较时，用保存的模型和种子重新生成报告。轨迹速度还会受
采样频率与实验位置滤波影响，生物比较仍须匹配实验的数据处理方法。

### 批量录制不同数量与布局

一次录制 2、5、8 个糖源，每种数量分别使用 seed=12、13、14，总计 9 段视频：

```bash
python -m cyberfly forage-batch --model runs/my-forage --food-counts 2 5 8 --seeds 12 13 14 --camera-distance 36 --output outputs/my-forage-tests
open outputs/my-forage-tests
```

`--food-counts` 中的每个数都是该组场景的确切糖源数量，范围为 1–12。
`--seeds` 控制随机位置；在相同数量和配置下，同一个种子可以复现布局，
换种子可以得到另一种布局。位置仍在起点周围默认 4–12 mm 的区域内。
每种数量都会与每个种子配对；例如改成 `--food-counts 1 3 6 10 --seeds 21 22`
会录制 8 段视频。使用新种子时不保证一定更难。

批次按顺序录制，保持固定相机。开始时将当前模型保存为批次目录的 `model.zip`，
整批使用这一个模型，避免正在进行的训练改变视频间的比较对象。
不会训练或修改原模型，也不会改变训练循环的验证设置。

`--camera-distance 36` 把镜头一次性调远，帮助容纳默认 4–12 mm 范围内的糖源，
在整段视频中保持不变，不会动态跟随或缩放。不传时保留原来的距离 30。
单段录制和 `--viewer` 也支持同一个参数，用相同值即可保持相同相机设置。

- `food_02_seed_12.mp4`：2 个糖源、seed=12 的视频，其他组合以相同方式命名。
- 每段视频的同名目录：`summary.json`、`trajectory.csv`、`events.json`。
- `index.json`：批次状态、模型来源、各段视频路径、逐场摄入和跌倒等指标。
- `model.zip`：该批使用的模型快照，可以用于后续复现。

批次输出目录必须是新目录或空目录，以免覆盖此前的视频。第二次使用时将
`--output` 改成例如 `outputs/my-forage-tests-02`。按 Control+C 可停止，已经完成的
视频会保留；当前未完成的视频可能只有部分内容，不作为完整回合统计。

只录制一段指定数量的场景：

```bash
python -m cyberfly forage --model runs/my-forage --food-count 8 --seed 21 --camera-distance 36 --video outputs/food8-seed21.mp4 --output outputs/food8-seed21
```

`--food-count` 是仅用于演示/测试的数量覆盖项，不更改检查点中的训练配置。
不传它时，继续使用模型保存的随机数量范围。若模型原本训练在 1–5 个糖源，
8 或 12 个属于超出训练数量范围的测试；终端和报告会明确标记
`outside_training_food_count_range: true`。这并不自动证明泛化成功或失败。
当前模型包含饱足反馈，糖源多时可能不会全部吃完；每段视频仍按模型保存的
回合时间上限运行，也可能因吃完或跌倒等条件提前结束。

### 使用更小的糖液滴，并重新测试

`sucrose-droplet` 预设用小型蔗糖液滴替代原来的抽象圆片：

- 原圆片直径 **1.6 mm**、高度 **0.24 mm**。
- 新液滴直径 **0.8 mm**、高度约 **0.298 mm**，扁椭球几何体积 **0.1 µL**。
- 体积参考[果蝇糖摄入与搜索实验](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2018.00280/full)
  使用的 0.1 µL 蔗糖液滴。直径、椭球形状与接触角没有从该实验测得，
  是本模型的几何选择；高度按 `V = 4πr²h/3` 计算，其中 h 为垂直半轴，1 mm³ = 1 µL。

这个预设表示从液滴表面摄入糖液，**不表示果蝇能吞下 0.8 mm 的固体糖晶体**。
食物是否适合摄食不能只由直径判断。当前仍没有液体流动、表面张力、唾液溶解或
吞咽流体力学；颜色是为了可视化。这里也没有复现实验的全部饥饿、照明和糖浓度条件。

感知与摄入使用新的半径和椭球形状，嘴部判定的垂直容差从旧斑块的 0.25 mm
收紧到 0.05 mm，并保留嘴部约 0.05 mm 的横向接触容差。它仍是功能性邻近判定，
不是由口器肌肉及液体接触力计算的真实吸食。

**摄入量仍用归一化食物单位**，每个食物初始为 1 单位。这里只校正几何与接触，
没有把摄入速度、能量或饱足标定为真实体积/热量；不能据模拟中吃完液滴的秒数
声称摄食速度符合实验。报告中的 `geometric_volume_ul` 只描述液滴的几何体积。

用当前模型观看新尺寸：

```bash
mjpython -m cyberfly forage --model runs/my-forage --food-profile sucrose-droplet --food-count 2 --seed 12 --camera-distance 36 --viewer --output outputs/my-droplet-view
```

批量录制新尺寸：

```bash
python -m cyberfly forage-batch --model runs/my-forage --food-profile sucrose-droplet --food-counts 2 5 8 --seeds 12 13 14 --camera-distance 36 --output outputs/my-droplet-tests
open outputs/my-droplet-tests
```

不传 `--food-profile` 时仍采用模型保存的配置。这个兼容规则防止旧训练在恢复时
被无声更换场景；原检查点和旧视频保留。`--food-profile legacy-patch` 可显式回到旧尺寸。
新报告同时保存 `food_geometry`、`training_config`、`test_overrides`，并用
`food_geometry_differs_from_training` 标记是否在不同于训练时的几何条件下测试。
批次 `model.zip` 保存原策略，复现这批测试时还要带上同样的 `--food-profile` 参数。

需要以后在新尺寸上训练时，新建训练目录并使用 `configs/forage-droplets.json`：

```bash
python -m cyberfly forage-loop --run runs/my-droplet-training --config configs/forage-droplets.json --rounds 3 --steps 2048 --eval-episodes 8
```

这会新建学习策略，不会续接其他几何配置下的旧策略。

尺寸检查及真实物理接近/摄入回归检查包含在 `python -m cyberfly forage-check` 中。
多种数量、布局和模型的评估请运行上面的批量录制命令，并检查各场景的报告。
这些检查验证功能实现，不能验证生物学摄食速度或饱足阈值。

## 2. 功能与模型边界

- **感觉控制器**：搜索、趋近、取食、饱足四种状态；只使用感觉读数，
  不把糖源的世界坐标直接传给控制器或学习策略。控制器自身是人工编写的规则。
- **人工神经网络**：每帧 90 维观察，保存最近 4 帧，经过 GRU 和 PPO 策略，
  输出左右行走与口器的三个修正量。默认每个修正幅度不超过 0.15；
  可用下方独立口器配置放开开闭决定。GRU 的记忆窗口约 80 ms，
  不保存整回合隐状态，也不是连接组或脉冲神经网络。
- **双眼功能感知**：每侧 24 个角度通道，模拟可见糖源的方向与表观大小。
  这是从场景几何计算的低分辨率对比信号，预先知道哪些物体是糖源；
  没有像素图像识别、遮挡、真实小眼光学或完整视叶回路。
- **味觉**：六足末端与口器的空间接触检测。嘴部接触糖源、身体直立、移动足够慢，
  并连续保持约 0.16 秒后才累计摄入；不能隔空或重复吃已耗尽的糖源。
- **口器动作**：为已有口器模型添加一个可控滑动关节。它能伸缩，但没有复现
  完整肌肉、吸泵、唾液、咀嚼或每口之间的真实收缩节律。
- **内部状态**：摄入增加能量，时间和运动消耗能量，高能量抑制继续进食。
  这些是无量纲工程参数，尚未标定为热量、真实饥饿或消化生理。
- **本体与接触反馈**：复用 FlyGym 的身体动力学、CPG、足部接触与反射修正；
  高层另外观察身体运动、姿态、步态、口器伸长和近期摄入信号。
- **嗅觉接口**：预留左右触角的简化挥发性食物信号。默认纯糖场景关闭它，
  不把纯蔗糖当成远距离气味源。打开开关也不会自动得到真实气流或嗅觉回路。

糖源每回合随机生成 1–5 个，位置在起点周围 4–12 mm 的区域。
每个糖源初始有 1 个归一化食物单位，采用营养斑块近似：视觉上像糖粒，
没有固体颗粒推动、溶解、液体吞咽物理或营养化学。

原来的行走模型有 33 维输入、2 维动作；新模型有 4×90 维观察、3 维动作，
因此不能直接加载旧的行走 PPO 检查点。这里复用的是原来的身体和低层行走控制，
使用独立的高层觅食网络。

### 口器自主控制配置

原配置将三维网络动作都乘以 `residual_scale`（默认 0.15），然后加在基线指令上。
基线的口器指令是 −1 或 +1，±0.15 的修正不能把它从关闭变成打开，或反过来。
已有模型仍采用这一规则，包括保存过自定义 `residual_scale` 的模型。

新预设 `configs/forage-mouth-control.json` 使用小型液滴，并将独立的
`mouth_residual_scale` 设为 2.0。左右腿仍按原来的 0.15 修正；口器指令为
`clip(基线口器指令 + 2 × 网络口器动作, −1, 1)`。这样网络既能打开原本关闭的
口器，也能关闭原本打开的口器。零动作仍保留基线行为，接触、速度、直立与饱足等
摄入条件继续生效；拥有控制权限不等于已经学会正确取食。

新建独立实验：

```bash
python -m cyberfly forage-loop --run runs/my-mouth-control --config configs/forage-mouth-control.json --rounds 3 --steps 2048 --eval-episodes 8
```

再次执行同一命令可继续实验。不要把此配置用于恢复旧训练目录；训练循环会拒绝
改变已保存的任务配置。省略 `mouth_residual_scale` 或设为 `null` 会沿用原来的共享
修正幅度；显式数值必须在 0–2 之间。新配置需要独立验证，不保证更好或更像真实果蝇。

配置加载时会拒绝 NaN、无穷大、错误类型及越界参数；糖源数量和版本必须是整数，
嗅觉开关必须是布尔值，回合长度必须是 0.02 秒的正整数倍。自定义摄入间隔仍按
控制步检测，非整数倍间隔会在达到该时长后的下一个控制步触发。

## 3. 先确认循环训练能够跑通

执行一个很短的流程测试：

```bash
python -m cyberfly forage-loop --run runs/my-forage-smoke --config configs/forage-smoke.json --rounds 1 --steps 128 --eval-episodes 2
```

这会训练、保存候选网络、在相同场景比较更新前后表现，并写出报告。
由于只有 2 个验证场景，少于要求的 5 个，这个测试**必然不会晋升候选模型**。
它只验证程序流程，不能证明学习有效。

`steps` 必须是 128 的正整数倍。每步是 0.02 秒控制时间，内部执行 200 个 MuJoCo
物理步。仿真和网络在 CPU 上运行；本任务默认没有使用 MPS 或 CUDA。

## 4. 正式训练，再开启持续循环

先运行 3 轮，每轮新增 2048 步，并用 8 个相同验证场景比较候选与当前模型：

```bash
caffeinate -i python -m cyberfly forage-loop --run runs/my-forage --config configs/forage.json --rounds 3 --steps 2048 --eval-episodes 8
```

`caffeinate -i` 在进程运行时防止机器因空闲睡眠。它不保证合盖后继续运行。
耗时包括物理仿真和两组验证，通常比纯训练更长，请看终端的轮次和进度。

确认运行正常后，使用同一目录开启持续循环：

```bash
caffeinate -i python -m cyberfly forage-loop --run runs/my-forage --rounds 0 --steps 2048 --eval-episodes 8
```

`--rounds 0` 表示持续运行，直到按 **Control+C**。这是本机训练进程，
没有创建定时自动任务。关闭终端或强制终止不能保证保存最近一次更新。

每轮流程为：从当前通过验证的模型开始 → 训练候选 → 成对验证 → 接受或拒绝。
候选被拒绝时不会替换当前模型，下一轮从当前模型重新尝试，
因此可能连续多轮没有更新；这不是保证单调进步的算法。

默认没有真实轨迹参考时，接受候选需要同时满足：

1. 至少 5 个相同种子的成对验证场景。
2. 每个验证场景的摄入比例均不退步，整体跌倒率、越界率不增加。
3. 任务分数改善的成对 bootstrap 95% 区间下界大于 0.002。

任务分数为 `摄入比例 − 0.02×耗时/回合上限 − 0.0005×路径长度(mm)`。
这是工程选择指标，不是生物学相似度。奖励主要来自真正发生的摄入，
另扣除时间、动作修正幅度及跌倒/越界代价。
有限样本、重复筛选和未见场景都可能带来误判，因此仍需独立测试。

文件结构：

- `state.json`：已完成轮次、当前模型路径、晋升次数和生物参考状态。
- `initial.zip`：初始网络；确定性输出为零修正，因此保留感觉控制器原行为。
- `round_0001_attempt_01/candidate.zip`：该次训练的候选模型。
- 每次尝试的 `evaluation.json`：逐回合指标、成对比较和拒绝原因。
- 每次尝试的 `status.json`：训练、完成、中断或失败状态。
- `tb/`：TensorBoard 日志。

查看当前采用哪个模型：

```bash
cat runs/my-forage/state.json
```

若 `champion` 仍是 `initial.zip`，表示尚没有候选通过门槛。
不要把“训练完成”或损失下降当作行为已经改善的证据。

## 5. 中断、恢复和观看当前模型

运行中按一次 Control+C，会尝试保存尚未验证的候选，当前模型保持不变。
再次执行同一个训练命令，程序会读取该候选的网络和优化器状态继续训练。
它不恢复完全相同的物理场景、随机数状态或尚未完成的 rollout。
正常完成后的同一命令会开启新的轮次。

候选每 256 个控制步还会定期保存，但只有标记为正常中断的候选才自动续接。
强制杀进程或断电后的定期检查点会保留，不能声称一定自动恢复全部进度。
每次尝试使用新目录；同一个训练目录不允许两个进程同时写入。
恢复时不能改变任务配置或真实数据参考，改变实验请使用新的 `--run` 目录。

直接传训练目录可以观看当前通过验证的模型，无需手动查找 zip：

```bash
mjpython -m cyberfly forage --model runs/my-forage --seed 12 --viewer --output outputs/my-forage-learned
```

模型默认使用保存时的配置，不能同时指定 `--config` 或 `--seconds`；
明确的测试参数 `--food-count` 和 `--food-profile` 可分别覆盖糖源数量与几何预设。
输出中的 `training_timesteps` 表示当前显示的模型接受过多少训练。
若没有候选晋升，此处仍为 0，这是有意保留可工作的初始控制器。

用没有参与默认模型筛选的种子做独立检查：

```bash
python -m cyberfly forage-evaluate --model runs/my-forage --episodes 10 --output outputs/my-forage-test.json
```

它使用 300000 开始的种子，对比固定感觉控制器和加载的模型；训练筛选使用
100000 开始的种子。不要反复根据独立测试结果调参后仍把这些种子称为未见测试集。

## 6. 公开研究如何指导模型

以下研究用于确定功能顺序和明确抽象边界，没有被转换成真实神经连接或实验数据：

- [Feeding initiation and the organization of feeding behavior](https://www.nature.com/articles/ncomms10678)：
  外部味觉检测、停止运动、口器伸展和摄入之间的功能关系。
- [flyPAD feeding measurements](https://www.nature.com/articles/ncomms5560)：
  摄食的细粒度时间结构；约 0.16 秒只用作当前接触累计时间的参考，
  并未重现完整摄食微结构或把归一化摄入量标定为实验体积。
- [Pharyngeal taste and sugar ingestion](https://www.nature.com/articles/ncomms7667)：
  咽部感受与持续摄入的关系；当前用摄入反馈近似，尚未建模真实咽部神经元。
- [Connectome-based whole-brain modeling](https://www.nature.com/articles/s41586-024-07763-9)：
  可作为后续连接组研究入口；本项目没有加载该连接组，也没有复现论文模型。
- [NeuroMechFly / FlyGym](https://neuromechfly.org/)：当前身体、接触动力学和步态基础。

这些论文不意味着已经得到与本模拟条件一致的真实轨迹。
在没有匹配轨迹时，所有报告保持 `biological_similarity: null`、
`biology.available: false`，只能评价任务表现。

已有真实数据时，`--reference` 可接入带来源与实验条件的 manifest。
CSV 必须包含 `episode_id,time_s,x_mm,y_mm,heading_rad,feeding`，至少 5 个独立实验；
`feeding` 必须为 0/1，并来自真实取食标注。坐标单位 mm，时间单位 s，角度 rad。
JSON manifest 的必需字段见 `cyberfly/behavior.py` 中 `BiologicalReference`：
`schema` 为 1，物种为 `Drosophila melanogaster`，包括 HTTPS 来源链接、性别、
实验协议说明、CSV 路径和完整 `matched_forage_config`。

参考条件需要核对性别、年龄、饥饿时长、照明、糖源成分和尺寸、场地、采样率等。
软件能检查格式及声明一致性，不能替代来源核验或证明实验条件真的匹配。
不要把本项目输出的模拟轨迹填成真实参考。

接入后比较五项轨迹描述：平均速度、转向率、停止比例、取食比例、取食段时长。
报告归一化 Wasserstein 分布距离，越低越接近这些统计特征，
并不代表“整只果蝇相似度百分比”。模型筛选还要求参考距离下降，
摄入和安全条件不退步；该启发式门槛不等于统计上确认生物真实性提高。
目前训练奖励没有直接加入该真实数据距离；它仅在模型筛选时使用。

## 7. 距离真实生物仍缺什么

当前没有实现完整大脑/VNC 连接组、真实视觉与嗅觉回路、神经调质、肌肉生理、
完整口器与消化、睡眠、梳理、繁殖等行为。身体里的翅膀网格不表示已经能够飞行。
当前任务仅包含地面觅食，没有飞行控制。

飞行需要单独加入翼拍动力学、空气动力模型、平衡棒反馈、起飞/降落策略并验证。
可以研究 [FlyBody 的飞行任务](https://github.com/TuragaLab/flybody/blob/main/flybody/tasks/flight_imitation.py)，
但应在独立环境核对其依赖和模型单位，不能直接在现有环境混装后声称完成。

后续提高真实性的优先顺序是：获取匹配觅食实验并校准传感与摄入参数，
再替换特定感觉或决策模块为有实验约束的神经回路，最后按明确实验目标扩展飞行等能力。
每个模块都需要单独验证；增加网络大小或训练轮数本身不保证更像真实果蝇。

## 8. 验证与复现

可复现的接口、物理和训练流程检查见 [验证指南](docs/VALIDATION.md)。
本仓库仅分发源码和配置，模型、日志、报告及视频由运行命令生成。
报告会记录当前配置和检查点信息；对外发布实验产物前应检查其中的本机路径等元数据。
