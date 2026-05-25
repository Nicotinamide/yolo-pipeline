# YOLO Training Pipeline

配置驱动的 YOLO 检测/分割训练流水线。给定图片目录和 YOLO 格式标签，自动完成数据集准备、训练、验证集预测。

## 系统支持

| 平台 | 状态 | 说明 |
|------|------|------|
| Jetson Orin (aarch64) | ✅ | 默认推荐 |
| Linux x86_64 + NVIDIA GPU | ✅ | 推荐用于服务器/工作站 |
| Linux x86_64 + CPU | ✅ | 仅供调试，训练慢 |
| macOS (Apple Silicon) | ⚠️ | 仅 CPU 模式，没测过 MPS |
| Windows | ⚠️ | 没测过，env.sh 需要 WSL 或 Git Bash |

要求：Python 3.10+，可选 CUDA 11.8 / 12.x。

## 快速开始

### 1. 安装环境（推荐 uv）

```bash
# 安装 uv（如果还没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目并创建虚拟环境
git clone <repo-url> && cd yolo-pipeline
uv venv --python 3.10
source .venv/bin/activate
```

按平台选择 PyTorch：

```bash
# Jetson Orin (aarch64, JetPack 6, CUDA 12.6)
uv pip install -e ".[jetson]" --extra-index-url https://pypi.jetson-ai-lab.com/jp6/cu126

# x86_64 + CUDA 12.x（桌面/服务器）
uv pip install -e ".[cuda12]" --extra-index-url https://download.pytorch.org/whl/cu124

# x86_64 + CUDA 11.8
uv pip install -e ".[cuda12]" --extra-index-url https://download.pytorch.org/whl/cu118

# 仅 CPU（调试用）
uv pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
```

备选 conda：

```bash
conda create -n yolo python=3.10 -y && conda activate yolo
# x86_64 GPU
pip install -e "." && pip install torch --index-url https://download.pytorch.org/whl/cu124
# Jetson
pip install -e "." && pip install torch --index-url https://pypi.jetson-ai-lab.com/jp6/cu126
```

> **跨平台说明**：`env.sh` 和 `project_env.py` 会自动检测 `aarch64` / `x86_64` 架构，
> 并加载对应的 CUDA 库目录。无 CUDA 也能正常运行（跳过 GPU 初始化，使用 CPU）。

### 2. 训练

打开 `train.sh`，改顶部 5 个变量：

```bash
IMAGES="/path/to/your/images"     # 图片文件夹
LABELS="/path/to/your/labels"     # YOLO txt 标签文件夹
TASK="detect"                     # detect 或 segment
NAME=""                           # 输出目录名（留空=自动取 IMAGES 文件夹名）
PRETRAINED="weights/yolo26n.pt"   # 预训练权重
EPOCHS=150
```

如果项目里没有预训练权重，先下载：

```bash
bash weights/download.sh                              # 默认下载 yolo26n.pt
bash weights/download.sh yolo26s.pt yolo26n-seg.pt    # 指定下载
```

跑：

```bash
bash train.sh
```

会自动激活环境、准备数据集（80/20 划分 train/val）、训练、在 val 集上跑预测可视化。

输出位置：

```
datasets/<NAME>/                  # 标准 YOLO 数据集（含 data.yaml）
runs/<NAME>/weights/best.pt       # 最优模型
runs/<NAME>_predict/              # val 集预测可视化
```

### 3. 对比实验

复制 `train.sh`，改变量再跑：

```bash
cp train.sh train_v8.sh
# 编辑 train_v8.sh：PRETRAINED="weights/yolov8n.pt", NAME="my_run_v8"
bash train_v8.sh
```

各自保存到 `runs/my_run_v8/`，互不覆盖。

或者打开 `KEEP_HISTORY=1`，每次训练自动加时间戳，永远不覆盖。

### 4. 续训

训练中断后，把 `train.sh` 顶部的 `RESUME` 设为 last.pt 路径再跑：

```bash
RESUME="runs/my_run/weights/last.pt"
```

## 项目结构

```
yolo-pipeline/
├── env.sh                  # 自动检测项目根目录、CUDA、venv
├── train.sh                # 主训练脚本（改配置→运行）
├── pyproject.toml          # uv/pip/conda 依赖声明
├── weights/                # 预训练权重
│   ├── download.sh         # 一键下载常用权重
│   └── *.pt                # YOLO 预训练权重
├── pipeline/               # 共享流水线代码
│   ├── config.py           # 配置解析与智能默认值
│   ├── yolo_dataset.py     # 数据集准备
│   └── yolo_runner.py      # 训练/预测运行器
└── tools/                  # 常用工具脚本
    ├── train.py            # train.sh 的 Python 实现
    ├── predict.py          # 单图/视频推理
    ├── compare_models.py   # 多模型对比推理
    ├── compare_imgsz.py    # 不同分辨率对比
    ├── batch_predict.py    # 批量推理统计
    ├── model_info.py       # 模型信息查看
    └── export_model.py     # 导出 TensorRT/ONNX
```

## 数据格式

输入数据需要 YOLO 格式：

```
your_dataset/
├── images/                 # 图片（jpg/png/...，扁平结构）
│   ├── img_001.jpg
│   └── ...
├── labels/                 # YOLO txt 标签（同名）
│   ├── img_001.txt
│   └── ...
└── classes.txt             # 类别名（每行一个）
```

`classes.txt` 会被脚本自动检测（`--images` 同级或上级目录）。

YOLO 标签格式：
- 检测：`class_id x_center y_center width height`（坐标全部 0~1 归一化）
- 分割：`class_id x1 y1 x2 y2 x3 y3 ...`（多边形顶点）

## 推理工具

训练完用 `tools/` 下的脚本验证模型：

```bash
source env.sh

# 单图推理
python tools/predict.py --model runs/my_run/weights/best.pt --source image.png

# 多模型对比
python tools/compare_models.py --models a.pt b.pt c.pt --source image.png

# 不同分辨率对比
python tools/compare_imgsz.py --model best.pt --source image.png --sizes 640 960 1280

# 批量推理统计
python tools/batch_predict.py --model best.pt --source datasets/my_run/images/val

# 模型信息
python tools/model_info.py --model best.pt

# 导出 TensorRT (Jetson 部署)
python tools/export_model.py --model best.pt --format engine --imgsz 960
```

## 高级用法

需要更精细控制（自定义数据增强、自定义优化器超参等）时，直接调 `tools/train.py`：

```bash
python tools/train.py --images /path/to/imgs --labels /path/to/lbls \
    --task segment --pretrained yolo26s.pt --epochs 200 --imgsz 1280 --batch 4

# 续训
python tools/train.py --images /path/to/imgs --labels /path/to/lbls \
    --resume runs/my_run/weights/last.pt

# 已有 YOLO 数据集（含 data.yaml），跳过 prepare
python tools/train.py --data datasets/my_data/data.yaml --name my_run

# 只看解析后的配置，不实际跑
python tools/train.py --images ... --labels ... --dry-run
```

更进一步的超参（mosaic、mixup、cos_lr、lr0、weight_decay 等）走 Python API：

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="datasets/my_run/data.yaml", epochs=200, mosaic=1.0, mixup=0.3, cos_lr=True)
```

## 智能默认值

为了"无关紧要的就别填"，以下参数都有默认值：

| 参数 | 默认值 |
|------|--------|
| `device` | 自动选 GPU/CPU |
| `pretrained` | 自动找 `weights/yolo26n.pt` / `weights/yolov8n.pt` / `weights/yolo11n.pt` |
| `model` | 按 task 推导（detect→`weights/yolo26n.pt`，segment→`yolo26n-seg.yaml`）|
| `imgsz` | 960 |
| `batch` | 8 |
| `epochs` | 150 |
| `workers` | 4 |
| `cache` | true |
| `amp` | true |
| `patience` | 30 |
| `close_mosaic` | 15 |
| `val_ratio` | 0.2 |
| `seed` | 42 |
