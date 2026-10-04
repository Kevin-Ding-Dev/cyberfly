# 验证与复现

以下命令均在项目根目录、已激活虚拟环境的终端执行。
源码仓库不包含预先生成的测试报告或模型；运行命令会生成本机产物。
已验证平台为 macOS ARM64 / Python 3.12，依赖版本见
`requirements-macos.lock.txt`。

## 接口与物理回归

```bash
python -m pip check
python -m unittest discover -s tests -v
python -m cyberfly check
python -m cyberfly forage-check
```

行走检查覆盖 Gymnasium/SB3 接口、随机种子复现、有限数值、回合结束及非法动作。
觅食检查覆盖感觉与摄入、食物耗尽和重置、实际物理接近与取食、小液滴尺寸和
接触边界，以及候选晋升与参考数据检查。

`tests/` 的回归测试覆盖模型晋升/文件替换时的报告来源、数值类型与有限性、
速度定义及真实轨迹一致性、录像帧数和提前结束，以及新口器配置与旧动作兼容性。
录像循环的单元测试使用模拟编码器，不替代下方实际 MP4 检查。

这些检查证明具体实现条件，不证明完整生物真实性，也不证明策略改善。

## 图形与相机

```bash
mjpython -m cyberfly forage --config configs/forage-droplets.json --seed 12 --camera-distance 36 --viewer --output outputs/view-check
python -m cyberfly forage --config configs/forage-droplets.json --seed 12 --camera-distance 36 --video outputs/video-check.mp4 --output outputs/video-check
open outputs/video-check.mp4
```

查看果蝇、糖源与口器是否正常渲染，相机是否保持静止，视频是否能播放。
两条命令使用相同配置、种子和相机设置；交互窗口手动拖动的相机不会传给录像。
视频与窗口宽高比不同可能改变边缘取景。

完整 15 秒录像应为 375 帧、25 fps。提前在奇数控制步结束时，播放时间可比仿真
多 0.02 秒，以保留最后状态；核对 `summary.json` 中的 `video` 与轨迹时间戳。
不要用旧的无版本速度摘要和新版本 2 的速度指标直接比较。

## 训练与检查点

```bash
python -m cyberfly forage-loop --run runs/pipeline-check --config configs/forage-smoke.json --rounds 1 --steps 128 --eval-episodes 2
```

检查候选模型和 `evaluation.json` 是否生成。两场验证不足晋升所需的五场，
因此这里预期拒绝晋升。此命令只验证训练流程，不用于证明行为改善。
重复测试时使用新的输出目录；完整训练与恢复流程见 [觅食指南](../FORAGING.md)。

验证独立口器控制的保存、训练和加载，可另用一个新目录，把配置改为
`configs/forage-mouth-control.json`，同样执行一轮 128 步与两场验证。
加载候选时应保留 `mouth_residual_scale: 2.0`；旧配置应继续使用共享修正幅度。
这类短测试只验证流程和参数更新，不构成学习成功或生物相似度改善的证据。

评估模型改进应使用相同配置、成对种子和独立测试集，并报告逐场摄入、跌倒、
越界与耗时。若声称接近生物行为，必须另外提供匹配实验条件的真实数据与来源。

## 发布源码前

- 检查待提交文件和 Git 作者/提交者身份，避免发布个人邮箱、绝对路径或密钥。
- 确认未包含虚拟环境、缓存、检查点、日志、视频或私人数据。
- 保留第三方版权与来源，核对新增资产的授权。
- 在不同目录复制源码并运行检查，确认不依赖原始工作区路径。

Git 忽略规则只防止未跟踪文件被通常的添加命令纳入；它不会清理已提交历史，
也不能替代发布前检查。
