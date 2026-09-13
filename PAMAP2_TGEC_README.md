# PAMAP2上でのTGEC独立蒸留実験

## 目的

この実験は、SHL-2023で学習した教師・生徒の重みをPAMAP2へ転移する実験ではない。
PatchEchoClassifier元論文の複数データセット評価と同じ方向性に合わせ、PAMAP2内で教師を学習し、同じPAMAP2内で生徒へ蒸留し、学習に含まれない被験者で評価する。

```text
PAMAP2 train subjects
  ├─ SENvT-B teacher training
  └─ PRC/APS/TG-Skip/TGEC training and distillation
                         ↓
                held-out PAMAP2 subject
```

SHL-2023用の`experiments/shl2023_senvt_kd/`は読み書きしない。PAMAP2の出力はすべて`experiments/pamap2_tgec/`へ保存する。

## 固定する実験条件

| 項目 | 設定 |
|---|---|
| データ | `dataset/pamap2_100hz_w496_s248_chest16g` |
| 入力 | 3軸加速度、496点 |
| ラベル | 12クラス |
| 使用被験者 | 101--108 |
| 除外被験者 | 109（24窓すべてがactivity 24） |
| 外側評価 | 1名をtestとする8-fold被験者独立評価 |
| validation | testとは別の1名を循環的に固定 |
| patch size | 16 |
| 入力パッチ数 | 31 |
| keep count | 16（`ceil(31 * 0.5)`） |
| Reservoir size | 1000 |
| 教師 | foldごとにPAMAP2で学習したSENvT-B |
| 生徒Seed | 0, 1, 2 |
| ファインチューニング | 別段階では実施しない |

ラベルは全foldで次の固定写像を使用する。

```text
1→0, 2→1, 3→2, 4→3, 5→4, 6→5,
7→6, 12→7, 13→8, 16→9, 17→10, 24→11
```

## 被験者分割

| Test | Validation | Train |
|---:|---:|---|
| 101 | 102 | 103--108 |
| 102 | 103 | 101, 104--108 |
| 103 | 104 | 101, 102, 105--108 |
| 104 | 105 | 101--103, 106--108 |
| 105 | 106 | 101--104, 107, 108 |
| 106 | 107 | 101--105, 108 |
| 107 | 108 | 101--106 |
| 108 | 101 | 102--107 |

分割は窓単位のランダム分割ではない。50%重複窓がtrainとtestへまたがらないよう、被験者単位で完全に分離する。

## 実験条件

| Method | Teacher | Route | Omitted patches | Loss |
|---|---|---|---|---|
| PRC-CE | なし | なし | 省略なし | CE |
| PRC-KD | PAMAP2 SENvT | なし | 省略なし | CE + logit KD |
| APS-CE | なし | 生徒router | 破棄 | CE |
| APS-KD | PAMAP2 SENvT | 生徒router | 破棄 | CE + logit KD |
| TG-Skip | PAMAP2 SENvT | route distillation | 破棄 | CE + logit KD + route loss |
| TGEC | PAMAP2 SENvT | route distillation | 1トークンへ凝縮 | CE + logit KD + route loss + content loss |

教師は学習時だけ使用する。held-out testおよび効率測定では教師を使用しない。

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `scripts/pamap2_tgec/config.sh` | データ、fold、Seed、学習条件、出力先 |
| `prepare_pamap2_loso_splits.py` | 固定被験者分割の生成と検証 |
| `01_prepare_splits.sh` | 分割生成の入口 |
| `02_train_teachers.sh` | foldごとのPAMAP2教師学習 |
| `03_run_students.sh` | 6条件のCE学習・蒸留 |
| `04_evaluate_students.sh` | held-out被験者評価と予測保存 |
| `aggregate_pamap2_results.py` | fold予測の結合とSeed集約 |
| `05_collect_results.sh` | 結果集約の入口 |
| `06_profile_efficiency.sh` | GPU/CPU latency、FLOPs、footprint等の測定 |
| `07_run_pipeline.sh` | Smokeまたは全実験を順番に実行 |

## 実行前確認

プロジェクト直下で実行する。

```bash
cd "/home/jovyan/work/srv11/蒸留/EchoClassifier"
conda activate mfm311

python -m py_compile \
  main.py datasets.py engine.py \
  scripts/pamap2_tgec/prepare_pamap2_loso_splits.py \
  scripts/pamap2_tgec/aggregate_pamap2_results.py

bash -n scripts/pamap2_tgec/*.sh
```

必要ファイルを確認する。

```bash
for f in X.npy Y.npy groups.npy meta.json; do
  test -f "dataset/pamap2_100hz_w496_s248_chest16g/$f" \
    && echo "[OK] $f" || echo "[MISSING] $f"
done

test -f dataset/SENvT-u4/1000k_task4/best.pth \
  && echo "[OK] SENvT pretrained checkpoint"
```

ここで使うSENvTチェックポイントは汎用事前学習の初期値であり、SHL-2023でファインチューニングした教師チェックポイントではない。

## Step 1: 分割生成

```bash
bash scripts/pamap2_tgec/01_prepare_splits.sh
```

出力:

```text
experiments/pamap2_tgec/data_splits/
├── pamap2_loso_splits.npz
└── pamap2_loso_splits.json
```

スクリプトはインデックス重複、被験者混入、trainのクラス欠落、Subject 109の混入を検査する。

## Step 2: Subject 101だけSmoke実験

次の一括コマンドは、Subject 101をtest、Subject 102をvalidationとし、教師と6条件を各1 epoch実行してtest評価まで行う。

```bash
bash scripts/pamap2_tgec/07_run_pipeline.sh smoke
```

個別に実行する場合:

```bash
FOLDS="101" SEEDS="0" \
bash scripts/pamap2_tgec/02_train_teachers.sh smoke 101

FOLDS="101" SEEDS="0" \
bash scripts/pamap2_tgec/03_run_students.sh smoke all 101

FOLDS="101" SEEDS="0" \
bash scripts/pamap2_tgec/04_evaluate_students.sh smoke all 101
```

Smokeで確認する値:

```text
sample_x shape: torch.Size([3, 496])
num_classes: 12
routing_patch_count: 31
routing_keep_count: 16
```

各評価フォルダには以下が生成される。

```text
evaluation_summary.json
predictions.npz
eval_stdout.log
```

Smoke結果は論文には使用しない。

## Step 3: 全8 foldの教師を学習

```bash
bash scripts/pamap2_tgec/02_train_teachers.sh full all
```

出力:

```text
experiments/pamap2_tgec/teacher_finetune/
├── test101/
├── test102/
├── ...
└── test108/
```

各foldの教師は、そのfoldのtest被験者とvalidation被験者を学習に使用しない。教師Seedは既定で0に固定し、同じfoldのすべての生徒条件で共有する。

中断後に再開する場合:

```bash
RESUME_PARTIAL=1 \
bash scripts/pamap2_tgec/02_train_teachers.sh full all
```

`summary.json`があるfoldは自動的にスキップされる。

## Step 4: 生徒6条件を学習

最初にSeed 0だけ実行して全foldの接続を確認できる。

```bash
SEEDS="0" \
bash scripts/pamap2_tgec/03_run_students.sh full all all
```

問題がなければSeed 1、2を実行する。

```bash
SEEDS="1 2" RESUME_PARTIAL=1 \
bash scripts/pamap2_tgec/03_run_students.sh full all all
```

一条件・一foldだけ実行する例:

```bash
SEEDS="0" \
bash scripts/pamap2_tgec/03_run_students.sh full tgec 101
```

全条件を最初から実行する場合:

```bash
bash scripts/pamap2_tgec/03_run_students.sh full all all
```

学習数は8 folds × 6 methods × 3 seeds = 144 runsである。

## Step 5: held-out被験者で評価

```bash
bash scripts/pamap2_tgec/04_evaluate_students.sh full all all
```

未完成のチェックポイントは`[SKIP missing]`として飛ばす。評価時は教師を読み込まない。

## Step 6: 結果集約

```bash
bash scripts/pamap2_tgec/05_collect_results.sh
```

出力:

```text
experiments/pamap2_tgec/results/
├── pamap2_fold_results.csv
├── pamap2_pooled_seed_results.csv
├── pamap2_seed_summary.csv
├── pamap2_seed_summary.md
└── pamap2_collection_status.json
```

主結果は、各Seedについて8foldの予測を結合して計算したpooled指標である。その後、Seed 0--2のpooled指標から平均±標準偏差を求める。被験者ごとの結果は`pamap2_fold_results.csv`に残す。

`pamap2_collection_status.json`の`complete`が全条件・全Seedで`true`であることを論文表作成前に確認する。

## Step 7: 推論効率測定

```bash
bash scripts/pamap2_tgec/06_profile_efficiency.sh
```

出力:

```text
experiments/pamap2_tgec/results/
├── pamap2_efficiency_gpu.json
└── pamap2_efficiency_cpu.json
```

測定項目には、trainable parameters、モデル状態サイズ（footprint）、latency中央値・平均、throughput、FLOPs、支配的MACs、GPU peak memoryが含まれる。CPUは既定で1 thread、batch sizeはGPU/CPUともに1である。

## 一括Full実行

Smokeと出力確認が終わった後だけ実行する。

```bash
bash scripts/pamap2_tgec/07_run_pipeline.sh full
```

長時間になるため、通常はStep 3--7を個別に実行し、教師とSeed 0の結果を確認してからSeed 1、2へ進む。

## 出力フォルダ

```text
experiments/pamap2_tgec/
├── data_splits/
├── teacher_finetune/
│   └── test101 ... test108/
├── student_distillation/
│   └── test101 ... test108/
│       ├── prc_ce_seed0 ... seed2/
│       ├── prc_kd_seed0 ... seed2/
│       ├── aps_ce_seed0 ... seed2/
│       ├── aps_kd_seed0 ... seed2/
│       ├── tg_skip_seed0 ... seed2/
│       └── tgec_seed0 ... seed2/
├── heldout_evaluation/
│   └── test101 ... test108/
└── results/
```

## 完了判定

| 状態 | 判定 |
|---|---|
| 学習完了 | `summary.json`がある |
| 学習途中 | `checkpoint.pth`はあるが`summary.json`がない |
| 評価完了 | `evaluation_summary.json`と`predictions.npz`がある |
| 未実行 | 対応フォルダまたはチェックポイントがない |

途中学習は`RESUME_PARTIAL=1`を指定して再開する。既存結果を意図せず上書きしないため、通常は`FORCE=1`を使用しない。

