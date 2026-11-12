# 训练数据与重训说明

> 写给下一个人，也写给下一个 agent。改动数据前请先读完这份。
> 最后更新：2026-11-12

## 一、先记三条红线

1. **不删源数据。** `datasets/` 下除本文件提到的"生成物"外，全部是手工收集与标注的原图，只读。任何清理动作先确认有没有 `.jiantu_eval_build` 标记，没有标记的目录一律不动。
2. **不按帧随机划分。** 同一段视频的相邻帧必须整体落在同一侧，否则训练集和验证集看到的是同一个摆位的两张照片，指标虚高。已有 `scripts/prepare_round2.py` 强制场景键（成员_视频号）不跨集。
3. **干净测试集不能进训练。** `datasets/products_clean_test` 是评测专用，一旦被选进训练集，这套指标就作废，必须重建一套新的。

## 二、目录盘点（2026-11-12 实测）

### 源数据（只读，手工收集 / 组员拍摄）

| 目录 | 图片数 | 内容 |
| --- | --- | --- |
| `chair/` | 2333 | 椅子，多个公开数据集合并后统一成 YOLO 格式 |
| `door_handle/` | 2722 | 门把手（另一条产品线，现役模型未使用） |
| `indoor/` | 4344 | 室内混合场景 |
| `raw/Bottle.v1i.yolov8`、`raw/test2.v7-sgd`、`raw/SDGP-Chair` | 5689 | 三个原始 Roboflow 导出包，未改动 |
| `team_video_pilot_20261025/` | 145 | 组员手机拍摄的视频抽帧，按场景键命名（`ABC_001_a.jpg`），带 `build_manifest.json` 记录 train/val 视频号 |

### 生成物（脚本产出，可重建）

| 目录 | 规模 | 由谁生成 |
| --- | --- | --- |
| `products/` | train 389 / valid 49 / test 49 | `scripts/build_products_dataset.py` |
| `products_v2/` | train 2370 / valid 49 / test 49 | `scripts/build_products_v2_dataset.py` |
| `products_v3/` | train 5055 / valid 49 / test 49 | `scripts/build_products_v3_dataset.py`（**现行**） |
| `products_clean_test/` | 1452 张（椅子 837 / 瓶子 596 / 组员抽帧 19） | 仓库内 `model-training/indoor/eval/build_clean_test.py` |
| `products_clean_test_near`、`_far` | 由上面按框面积中位数切出的硬链接子集 | 同上 `evaluate.py` |
| `team_video_round2_<日期>` | 待生成 | `scripts/prepare_round2.py` |

`products_clean_test*` 三个目录带 `.jiantu_eval_build` 标记，只有带标记才会被重建脚本复用/覆盖；没有标记的同名目录会被改名留档，不会被删。

### 已知的坑：v2/v3 的 test 不干净

`products/file_map.csv` 记录了每张图来自哪个拍摄场次。四个场次（椅子+门、杯子+椅子、杯子+椅子+门、杯子+门）**全都横跨 train/valid/test**，相邻照片只差一点角度。所以 49 张 test 上测出的 mAP50 0.55 是虚高的，别再引用。

干净测试集的做法：从训练池没用到的公开数据取图，逐张与训练池比对 dHash，汉明距离 ≤4 视为近重复剔除，候选之间再按文件摘要去重。1452 张留下，1753 条被排除的逐条记在 `products_clean_test/excluded.csv`（已在训练集内 861、近重复 368、集合内重复 524）。

## 三、下次训练怎么更新数据

### 场景 A：只加新拍的视频帧（推荐，也是当前计划）

```bash
cd F:/归档/数据集与标注/YOLOv8
python scripts/prepare_round2.py \
    --sources datasets/team_video_pilot_20261025/train datasets/team_video_pilot_20261025/val 新帧目录 \
    --val-ratio 0.25 --name team_video_round2_20261120
```

- 脚本按 `成员_视频号` 前缀分组，同视频只进一边，并做 SHA-256 查重，跨集重复直接拒绝输出；
- 默认三类（bottle / chair / door）。只有当确实补标了 person 时才加 `--with-person`，否则拒绝启动，避免出现空类；
- 输出目录已存在就报错退出，不覆盖旧轮次（翻车了还能回到上一版数据）。

### 场景 B：补充公开数据（加近景/远景样本）

1. 新数据先放进 `raw/<数据集名>/`，保持 YOLO 目录结构（`images/` + `labels/`），**不要**直接塞进 `products_v3`；
2. 复制 `scripts/build_products_v3_dataset.py` 为 `build_products_v4_dataset.py`，改源目录列表与抽样量，输出改 `products_v4`；
3. 训练前先看类别分布：v3 的问题是椅子近景占比过高（中位框面积 0.331），而测试集多为远景（0.027），v3 新增的 2685 张近景图里 bottle 一个框都没加。**加数据要对着"测不好的那一档"加**，不是对着数量加。

### 训练与命名

```bash
python 开始训练脚本或 yolo 命令，务必不要覆盖旧 runs：
runs/detect/products_v4/      # 新一版
```

训练完立刻做三件事：
1. 在**新的**干净测试集上评测（旧的一旦被训练污染就作废，见 `build_clean_test.py` 重跑）；
2. 把 `runs/detect/products_v4/weights/best.pt` 拷进仓库 `model-training/indoor/products_v4/weights/`；
3. 更新 `server/indoor.py` 的 `MODELS` 默认项，或先用 `?mode=v4` 灰度，别直接换默认。

### 换模型后的验收清单

- `GET /api/status` 能看到新权重路径且 `present: true`；
- 同一张图用新旧两版各跑一次，框与置信度差异在预期内（见 `model-training/indoor/eval/REPORT.md` 的方法）；
- 干净测试集上重跑 `evaluate.py`，把数字写进 REPORT.md，**只写能站住的结论**：v3 相对 v2 的收益集中在近景召回（0.673→0.829），总 mAP 持平（0.2917→0.2887），远景两版都失效。不能写成"精度提升"。

## 四、各目录的"可删 / 不可删"

| 可以重建 | 不能动 |
| --- | --- |
| `products/`、`products_v2/`、`products_v3/`（由脚本生成） | `chair/`、`door_handle/`、`indoor/`、`raw/`、`team_video_pilot_20261025/` |
| `products_clean_test*`（带标记，可重建） | `products/file_map.csv`（场次映射，重建测试集的依据） |
| `runs/detect/*`（训练产物，可重训） | 仓库内 `model-training/indoor/*/weights/best.pt`（线上权重） |

回收站里若出现 `products_clean_test*`，多半是重建脚本删掉的旧副本，不必惊慌：源图一直在原目录，`manifest.csv` 里每张图的 `origin` 字段能直接指回去。
