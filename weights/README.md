# 预训练权重

存放 YOLO 预训练权重文件（.pt）。

## 下载

### 交互模式（推荐新手）

直接运行，跟随提示选择系列、任务、规模：

```bash
bash weights/download.sh
```

```
1. 选择 YOLO 系列  →  YOLO26 / YOLO11 / YOLOv8
2. 选择任务类型    →  检测 / 分割
3. 选择模型规模    →  n / s / m / l / x / 全部
4. 确认下载
```

### 命令行模式（推荐熟练用户）

```bash
bash weights/download.sh yolo26n.pt              # 单个
bash weights/download.sh yolo26s.pt yolo11n.pt   # 多个
bash weights/download.sh yolo26-detect           # YOLO26 检测全套 (n/s/m/l/x)
bash weights/download.sh yolo26-segment          # YOLO26 分割全套
bash weights/download.sh detect-all              # 所有系列的检测
bash weights/download.sh segment-all             # 所有系列的分割
bash weights/download.sh all                     # 全部
```

## 使用

在 `train.sh` 中通过相对路径引用：

```bash
PRETRAINED="weights/yolo26n.pt"        # YOLO26 nano（默认，速度优先）
PRETRAINED="weights/yolo26s.pt"        # YOLO26 small（精度更好）
PRETRAINED="weights/yolo26n-seg.pt"    # YOLO26 nano 分割
PRETRAINED="weights/yolo11n.pt"        # YOLO11 系列
```

也可以让脚本自动选择（不设 `PRETRAINED` 时），会按下面顺序找：
1. `weights/yolo26n.pt`
2. `weights/yolov8n.pt`
3. `weights/yolo11n.pt`

## 命名规则

模型规模从小到大：`n < s < m < l < x`，越大精度越高、速度越慢、显存占用越多。

后缀变体：
- 无后缀 — 检测（detect）
- `-seg` — 实例分割
- `-pose` — 关键点估计
- `-obb` — 旋转框
- `-cls` — 图片分类

## 来源

来自 Ultralytics 官方 GitHub release v8.4.0：
<https://github.com/ultralytics/assets/releases/tag/v8.4.0>
