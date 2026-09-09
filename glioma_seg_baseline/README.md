# 赛道四脑胶质瘤病灶分割基线

这个目录是一版“先跑通”的赛道四分割工程。它以比赛公共模型中的 nnU-Net v2 为基线，先做两个独立分割模型：

- Core 模型：输入 T1CE/T1WI+C，输出核心区二值 mask。
- Total Abnormal 模型：输入 FLAIR，缺失时退到 T2，输出总异常区二值 mask。

这样设计的原因是比赛规则明确要求：

- 核心区 mask 必须和对应 T1 增强原始影像保持相同空间维度和 affine。
- 总异常区 mask 必须和对应 FLAIR/T2 原始影像保持相同空间维度和 affine。
- mask 体素值只能是 0 或 1。

第一版先保证数据、训练、推理、提交格式能跑通；后续再加入多模态配准、分类模型和集成。

## 目录内容

- `src/glioma_baseline/sequence_select.py`：根据文件名和序列描述识别 T1CE、FLAIR、T2。
- `src/glioma_baseline/prediction_json.py`：生成比赛要求的 `prediction.json` 骨架。
- `src/glioma_baseline/nnunet_runner.py`：调用 nnU-Net v2 命令行做预测。
- `src/glioma_baseline/service.py`：比赛推理服务，提供 `/health` 和 `/call`。
- `tools/brats_to_nnunet_single_modal.py`：用 BraTS 构造两个 nnU-Net 小数据集。
- `scripts/train_nnunet_baseline.sh`：训练 Core/Total 两套 nnU-Net。
- `scripts/start_service.sh`：启动比赛推理服务。

## 先用 BraTS 构造小样本

BraTS 常见结构：

```text
BraTS-GLI-00001-000/
  BraTS-GLI-00001-000-t1n.nii.gz
  BraTS-GLI-00001-000-t1c.nii.gz
  BraTS-GLI-00001-000-t2w.nii.gz
  BraTS-GLI-00001-000-t2f.nii.gz
  BraTS-GLI-00001-000-seg.nii.gz
```

转换命令：

```bash
python glioma_seg_baseline/tools/brats_to_nnunet_single_modal.py \
  --brats-root /path/to/BraTS \
  --out-root /2026aicompetition/workspace/nnUNet_raw \
  --num-cases 8
```

输出：

- `Dataset401_GliomaCoreT1CE`
- `Dataset402_GliomaTotalFLAIR`

标签映射：

- Core = BraTS TC = `seg == 1 or seg == 4`
- Total Abnormal = BraTS WT = `seg > 0`

## 训练

在云平台中，优先使用比赛提供的 nnU-Net：

```text
/2026aicompetition/public_models/MIC-DKFZ/nnUNet
```

训练前设置 nnU-Net 路径：

```bash
export nnUNet_raw=/2026aicompetition/workspace/nnUNet_raw
export nnUNet_preprocessed=/2026aicompetition/workspace/nnUNet_preprocessed
export nnUNet_results=/2026aicompetition/workspace/nnUNet_results
```

然后运行：

```bash
bash glioma_seg_baseline/scripts/train_nnunet_baseline.sh
```

默认会依次训练：

- Dataset 401：Core
- Dataset 402：Total Abnormal

## 推理服务

启动：

```bash
bash glioma_seg_baseline/scripts/start_service.sh
```

服务要求：

- `GET /health` 返回 200。
- `POST /call` 接收平台请求。
- 推理结果写入 `/2026aicompetition/workspace/answer/{evaluation_id}`。
- 每个病例输出：
  - `{AccessionNumber}/prediction.json`
  - `{AccessionNumber}/{SeriesUid}/{SeriesUid}.nii.gz`

## 重要限制

这是一版跑通基线，不是最终比赛模型：

- 分类字段目前是合规占位值，用来保证输出结构完整。
- 如果找不到训练好的 nnU-Net checkpoint，服务会输出全 0 mask，保证格式不炸，但没有分割能力。
- 真正上分需要完成 BraTS/自造数据训练，并补充分类模型。
