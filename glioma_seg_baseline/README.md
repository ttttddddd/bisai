# 赛道四脑胶质瘤病灶分割基线

这是面向比赛平台的“先跑通”分割工程，当前不依赖 BraTS，默认直接读取云平台训练标注数据和公共模型路径。

## 目标

赛道四的分割部分可以先拆成两个二值分割任务：

- Core：输入 T1CE/T1WI+C，输出核心区 mask。
- Total Abnormal：输入 FLAIR；如果 FLAIR 缺失，退到 T2，输出总异常区 mask。

第一版目标不是追求最高精度，而是先把平台数据读取、nnU-Net 训练、推理服务和结果输出跑通。

## 默认平台路径

代码默认使用以下路径：

```text
训练标注数据：/2026aicompetition/datasets/training/annotation
训练标签表：  /2026aicompetition/datasets/training/label
公共 nnU-Net：/2026aicompetition/public_models/MIC-DKFZ/nnUNet
nnU-Net raw： /2026aicompetition/workspace/nnUNet_raw
模型输出：    /2026aicompetition/workspace/nnUNet_results
提交结果：    /2026aicompetition/workspace/answer
日志目录：    /2026aicompetition/workspace/logs
```

这些默认值写在 `configs/config.yaml`，也可以用环境变量覆盖。

## 目录内容

- `configs/config.yaml`：平台路径、服务端口、nnU-Net 数据集编号。
- `tools/platform_to_nnunet.py`：把平台训练标注数据转换成 nnU-Net v2 数据集。
- `scripts/prepare_platform_nnunet_data.sh`：一键生成 401/402 两个训练数据集。
- `scripts/train_nnunet_baseline.sh`：安装/调用公共 nnU-Net，并训练两个分割模型。
- `scripts/validate_competition_dataset.py`：检查平台病例、序列和候选 T1CE/FLAIR/T2。
- `src/glioma_baseline/service.py`：比赛推理服务，提供 `/health` 和 `/call`。
- `src/glioma_baseline/sequence_select.py`：按序列描述/文件名选择 T1CE、FLAIR、T2。
- `src/glioma_baseline/prediction_json.py`：生成 `prediction.json`。

## 先检查平台数据

在云平台容器里进入仓库后运行：

```bash
python glioma_seg_baseline/scripts/validate_competition_dataset.py \
  --dataset-path /2026aicompetition/datasets/training/annotation \
  --output-csv /2026aicompetition/workspace/logs/td_segment_dataset_check.csv
```

如果要检查 NIfTI 文件是否能打开，加：

```bash
--validate-nifti
```

这个检查会告诉你每个病例下有多少序列，以及当前规则能否选出 T1CE 和 FLAIR/T2。

## 生成 nnU-Net 训练数据

```bash
bash glioma_seg_baseline/scripts/prepare_platform_nnunet_data.sh
```

默认输出：

```text
/2026aicompetition/workspace/nnUNet_raw/Dataset401_GliomaCoreT1CE
/2026aicompetition/workspace/nnUNet_raw/Dataset402_GliomaTotalFLAIR
```

转换逻辑：

- Core 数据集优先选择 T1CE/T1WI+C 序列。
- Total Abnormal 数据集优先选择 FLAIR，缺失时退到 T2。
- 如果平台标签表里能读到 `AccessionNumber`、`SeriesUid`、`Maskname`、`Task/SeriesLabel` 等字段，就按表格定位 mask。
- 如果标签表不可用，就在对应病例/序列目录里按文件名关键词兜底寻找 mask。
- 默认跳过没有找到 mask 的病例，避免把未知病例当阴性训练。

少量样本调试可设置：

```bash
MAX_CASES=8 bash glioma_seg_baseline/scripts/prepare_platform_nnunet_data.sh
```

只想测试流程但没有 mask 时，可以临时生成空 mask：

```bash
INCLUDE_EMPTY=1 MAX_CASES=2 bash glioma_seg_baseline/scripts/prepare_platform_nnunet_data.sh
```

注意：`INCLUDE_EMPTY=1` 只能用于检查流程，不能用于正式训练。

## 训练

```bash
bash glioma_seg_baseline/scripts/train_nnunet_baseline.sh
```

脚本会：

1. 设置 nnU-Net 工作目录。
2. 如果找不到 `nnUNetv2_*` 命令，就从 `/2026aicompetition/public_models/MIC-DKFZ/nnUNet` 安装。
3. 如果 401/402 数据集不存在，先自动生成。
4. 依次训练 Core 和 Total Abnormal 两个模型。

## 推理服务

启动：

```bash
bash glioma_seg_baseline/scripts/start_service.sh
```

服务接口：

- `GET /health`：返回服务状态。
- `POST /call`：接收平台请求，后台推理。

推理输出目录：

```text
/2026aicompetition/workspace/answer/{evaluation_id}
```

每个病例会输出：

```text
{AccessionNumber}/prediction.json
{AccessionNumber}/{SeriesUid}/{SeriesUid}_core.nii.gz
{AccessionNumber}/{SeriesUid}/{SeriesUid}_flair.nii.gz
```

如果模型 checkpoint 暂时不存在，服务会输出全 0 mask，保证提交格式先跑通；训练完成后会自动优先调用 nnU-Net 预测。

## 当前限制

- 这是单序列双模型基线：Core 用 T1CE，Total 用 FLAIR/T2；还不是最终多模态融合模型。
- 平台真实序列可能是 UID 命名，序列选择高度依赖标签表或描述字段。
- 分类/征象字段目前仍是占位值，后续需要由其他模块补齐。
