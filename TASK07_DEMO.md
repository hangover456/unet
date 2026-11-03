# Task07 Pancreas 小样本完整流程

用途：向老师展示已经跑通原始 NIfTI 数据检查、预处理、训练、保存模型、整卷预测、评价和可视化。它是**小样本短训练流程验证**，不是正式收敛基线或论文性能复现。

## 本次配置

- 原始数据自动定位到 `~/Desktop/data/Task07_Pancreas/Task07_Pancreas`；也可传外层目录。
- 保留 0 背景、1 pancreas、2 tumour。另汇总 whole pancreas（1 和 2 的并集）指标。
- 固定病例划分在 `task07_demo/common.py`：8 训练、2 验证、2 测试，两个仓库完全一致。
- 2D 单 CT 通道，256×256，batch 2，适合本机 RTX 3050 4 GB。
- 默认训练 90 分钟；每轮 100 个训练 batch、10 个验证 batch。在训练 batch 边界检查时间，完成当前轮的验证和模型保存后退出。
- 数据检查/预处理和预测/评价时间**额外计算**，因此总耗时会超过 90 分钟。修改 `--minutes 60` 或 `--minutes 120` 调整训练预算。
- 全部训练从随机初始化开始，不加载旧权重。

## 本机直接运行

先在 U-Net 项目运行，结束后再运行 nnU-Net；不要同时占用 4 GB GPU。

```powershell
& 'C:\Users\30692\PycharmProjects\unet\RUN_TASK07_DEMO.cmd' --minutes 90
& 'C:\Users\30692\PycharmProjects\nnUNet\RUN_TASK07_DEMO.cmd' --minutes 90
```

也可以双击各项目根目录的 `RUN_TASK07_DEMO.cmd`，默认 90 分钟。入口会自动检查并选择 CUDA Python。U-Net 当前旧环境缺少 PyTorch，因此会使用相邻 nnUNet 项目的 `.venv`。本机已安装依赖，无需重新安装。

若更换目录或电脑，设置解释器后运行：

```powershell
$env:TASK07_PYTHON = '你的GPU环境\python.exe'
& .\RUN_TASK07_DEMO.cmd --data '数据路径' --work '输出路径' --minutes 90
```

新环境可用 Python 3.10+，先按 PyTorch 官方说明安装适合 GPU 的 CUDA 版 PyTorch，再在 nnUNet 仓库执行 `python -m pip install -e .`。两个入口可共用此环境。以仓库 `pyproject.toml` 的依赖约束为准。

## 两个实现的具体含义

U-Net 仓库保留原 Keras 代码，新增 `task07_demo/model.py` 的 PyTorch U-Net。四次下采样、跳跃连接、四次上采样、三类输出；起始通道由旧版 64 缩为 32，上采样采用 nearest + 3×3 卷积，使用 CE + foreground soft Dice。训练、验证和预测均使用 FP32；Adam 学习率为 `1e-4`，梯度范数上限为 1，非有限损失/梯度会在更新前报错。它是针对 Task07/4GB GPU 的适配实现，不兼容旧 Keras 权重。预处理采用固定 HU 窗 [-100,240] 和线性归一化，每例最多缓存 32 张前景切片与 16 张背景切片。仅训练/验证采样使用标签；预测遍历整卷全部切片，不用真值裁剪。

nnU-Net 使用仓库真实的 fingerprint、planner、preprocessor、数据增强、网络、损失与优化器。只新增 `nnUNetTrainerTask07Demo`，限制时间预算；2D plan 的 patch/batch 明确调整为 256×256/2。fingerprint、spacing 和归一化统计仅来自 8 个训练病例；另外 2 个验证病例用相同统计预处理，2 个测试病例不参与训练与规划。该配置不是默认五折/1000 epoch 的标准 nnU-Net 基准。

规划器可能打印 3D 配置的计算过程，本入口只预处理和训练 `2d`，不会启动 3D 模型训练。`--max-epochs` 是额外上限，通常先到达 `--minutes` 的时间预算。

两者使用相同病例和标签，但预处理与训练策略不同，短训练结果不能用于正式宣称某模型更优。

## 运行完成后的材料

默认输出到 `~/Desktop/data/Task07_DemoRuns/`，每次训练创建新时间戳目录，保留以前结果。

```text
Task07_DemoRuns/
  shared/dataset_audit.json      数据几何、标签、源文件哈希和患者划分
  shared/split.json
  unet/时间戳/
  nnunet/时间戳/
    config.json                 参数与用途
    environment.json            GPU、Python、PyTorch、源码指纹
    console.log                 全过程日志
    history.json                每轮损失和采样 patch Dice
    training_summary.json       实际训练时间、轮数、更新批次数、权重路径
    learning_curves.png
    predictions/*.nii.gz        验证/测试整卷预测，保持原始几何
    metrics.csv / metrics.json  按患者、类别和数据组统计的完整体积指标
    overlay_*.png               CT / 真值 / 预测图
    report_evidence.zip         交给助手生成报告的轻量材料包
```

U-Net 权重在运行目录下，nnU-Net 权重在该运行的 `results/` 子目录中；轻量 ZIP 不包含 CT、预测体积和大型 checkpoint。报告所需参数、日志、曲线、逐病例指标与示例图均在 ZIP 中。请把两个模型各自最新的 `report_evidence.zip` 发回，并注明训练过程中是否手动停止或报错。

采样 patch Dice 用于训练过程观察；`metrics.csv` 才是整卷评价，二者不可混用。肿瘤在小样本短训练下可能得到 0 Dice，应如实报告。图中示例切片根据真值最大胰腺面积选取，只用于可视化，整卷指标覆盖全部切片。

## 分阶段和故障处理

```powershell
& .\RUN_TASK07_DEMO.cmd --stage prepare
& .\RUN_TASK07_DEMO.cmd --stage train --minutes 90
& .\RUN_TASK07_DEMO.cmd --stage predict
& .\RUN_TASK07_DEMO.cmd --stage collect
```

`predict`/`collect` 默认读取该模型最新运行目录；也可加 `--run-dir '某次运行目录'`。若训练失败，不会冒充完成；错误状态和日志保存在运行目录。修复后可新建一次运行，已完成预处理可复用。不自动恢复中断的优化器状态。按 Ctrl+C 会保留已存在的 checkpoint，并尽可能保存 interrupted checkpoint。

若显存不足，关闭其他 GPU 程序再运行；本次已经按 batch 2/256 patch 做真实 CUDA 短测试。训练时保持接通电源、禁止系统睡眠。

### 旧版 U-Net 的 Non-finite training loss

初版 FP16/Adam `1e-3` 在真实训练第 239 个 batch 复现了前向溢出；同一批输入和权重改为 FP32 后损失有限。当前已改为 FP32 和较低学习率，并检查验证/预测输出、训练梯度。原始失败目录保留，重新执行同一启动命令会新建运行目录并复用已有预处理，无需删除数据或缓存，也不加载失败运行的权重。

训练设置另存 `training_settings.json`，每轮梯度范数记录在 `history.json`，以便核验实际精度和稳定性设置。

## 快速复查

```powershell
& .\RUN_TASK07_DEMO.cmd --minutes 0.05 --steps 2 --val-steps 1 --max-epochs 1 --work "$env:TEMP\Task07Smoke"
```

这是一次短测试，不是完整 90 分钟实验。代码检查测试可运行 `python -m unittest discover -s task07_demo -p test_contract.py`。
