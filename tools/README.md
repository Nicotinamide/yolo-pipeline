# 工具脚本 (tools/)

常用的训练、推理、对比、导出工具。所有脚本都支持相对路径（相对于项目根目录）。

## 训练

```bash
# 一行命令训练（推荐用 ../train.sh，更直观）
python tools/train.py --images /path/to/images --labels /path/to/labels --task detect

# 分割任务
python tools/train.py --images /path/to/images --labels /path/to/labels --task segment --name my_seg

# 自定义参数
python tools/train.py --images /path/to/images --labels /path/to/labels \
    --pretrained yolo26s.pt --epochs 200 --imgsz 1280 --batch 4

# 已有 YOLO 数据集（含 data.yaml），跳过 prepare
python tools/train.py --data datasets/my_data/data.yaml --name my_run

# 只准备数据集，不训练
python tools/train.py --images ... --labels ... --prepare-only

# 只看解析后的配置，不实际跑
python tools/train.py --images ... --labels ... --dry-run
```

## 推理

```bash
# 单图
python tools/predict.py --model runs/my_run/weights/best.pt --source image.png

# 视频
python tools/predict.py --model best.pt --source video.mp4

# 摄像头实时
python tools/predict.py --model best.pt --source 0 --show

# 调整阈值和分辨率
python tools/predict.py --model best.pt --source image.png --imgsz 1280 --conf 0.3
```

## 多模型对比

```bash
# 自动找项目根目录下所有 best*.pt
python tools/compare_models.py --source image.png

# 指定模型列表
python tools/compare_models.py --models a.pt b.pt c.pt --source image.png
```

## 分辨率对比

```bash
# 默认测试 640, 960, 1280, 1920
python tools/compare_imgsz.py --model best.pt --source image.png

# 自定义
python tools/compare_imgsz.py --model best.pt --source image.png --sizes 480 640 800 960 1280
```

## 批量推理与统计

```bash
# 对整个目录跑推理，输出统计
python tools/batch_predict.py --model best.pt --source /path/to/images

# 保存带标注的图片
python tools/batch_predict.py --model best.pt --source /path/to/images --save
```

## 模型信息

```bash
python tools/model_info.py --model best.pt
python tools/model_info.py --model best.pt --verbose
```

## 模型导出 (部署)

```bash
# TensorRT FP16 (Jetson 部署推荐)
python tools/export_model.py --model best.pt --format engine --imgsz 960

# ONNX
python tools/export_model.py --model best.pt --format onnx

# TensorRT FP32
python tools/export_model.py --model best.pt --format engine --no-half
```
