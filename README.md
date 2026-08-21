# EchoAttnNet / Distillation Experiments (SHL-2023)

このREADMEは、SHL-2023に対する蒸留実験（MOMENT / SENvT / E-DTD / 2段階蒸留）の実行コマンドと保存先を集約し、久しぶりに開いても途中経過を追えるようにするための管理用メモである。

---

## 1. Environment（conda）

### 1.1 env作成 + environment.yml 出力

```bash
conda create -n mfm311 python=3.11 -y && conda activate mfm311 && \
conda install -y numpy=1.25.2 && \
pip install huggingface-hub==0.24.0 transformers==4.33.3 momentfm==0.1.4 && \
conda env export > environment.yml
```

### 1.2 Jupyter kernel

```bash
conda create -n mfm311 python=3.11 -y && conda activate mfm311 && \
conda install -y numpy=1.25.2 && \
pip install huggingface-hub==0.24.0 transformers==4.33.3 momentfm==0.1.4 && \
conda env export > environment.yml
conda activate mfm311
pip install ipykernel
python -m ipykernel install --user --name mfm311 --display-name "Python (mfm311)"
pip install pandas scikit-learn timm
```

---

## 2. Entrypoints

* `main.py`

  * timm互換の通常学習ループ（`train_one_epoch` / `evaluate`）
  * `DistillationLoss` によるKD（`--distillation-type` が `none` なら通常学習）

* `main_dist.py`（呼び出し名が `main_di.py` の可能性あり）

  * `--use-edtd` 指定時にE-DTDループ（`edtd_train`）へ分岐

---

## 3. 出力（runs）ルール

出力は基本 `./runs/...` に保存する。

生成物（想定）：

* `checkpoint.pth`
* `best_checkpoint.pth`
* `log.txt`（json lines）

---

## 4. Experiments

### 4.1 MOMENT：teacher moment-large → student moment-small

#### (A) 蒸留あり（soft KD / α=0.5, τ=2.0）

```bash
python main.py --data SHL2023 --student moment-small --model moment-large --distillation-type soft --distillation-alpha 0.5 --distillation-tau 2.0 --epochs 50 --batch-size 64 --output_dir ./runs/shl2023_moment_large2small_soft --mixup 0.8 --cutmix 1.0 --mixup-mode batch --flip_on
```

#### (B) ベースライン（蒸留なし・比較用）

```bash
python main.py --data SHL2023 --student moment-small --model moment-large --distillation-type none --epochs 50 --batch-size 64 --output_dir ./runs/shl2023_baseline_no_kd --ThreeAugment --mixup 0.8 --cutmix 1.0 --mixup-mode batch --flip_on
```

#### (C) 蒸留あり（安定寄り：α=0.4, τ=3.0）

```bash
python main.py --data SHL2023 --student moment-small --model moment-large --distillation-type soft --distillation-alpha 0.4 --distillation-tau 3.0 --epochs 80 --batch-size 64 --lr 3e-4 --warmup-epochs 5 --output_dir ./runs/shl2023_moment_large2small_soft_slow --ThreeAugment --mixup 0.8 --cutmix 1.0 --mixup-mode batch --flip_on
```

---

### 4.2 SENvT：teacher senvt-B → student senvt-XS / senvt-S

#### (A) senvt-B → senvt-XS（失敗ログ）

* 保存先：`./runs/shl2023_senvtB_to_XS`
* 現象：Epoch0の途中（例：670/2751）でLossがNaN

```bash
python main.py --data SHL2023 --input-size 496 --student senvt-XS --model senvt-B --distillation-type soft --distillation-alpha 0.2 --distillation-tau 2.0 --epochs 50 --batch-size 64 --lr 5e-4 --warmup-epochs 5 --output_dir ./runs/shl2023_senvtB_to_XS
```

#### (B) senvt-B → senvt-XS（NaN対策版：mixup/cutmixオフ、clip-grad）

* 保存先：`./runs/SHL2023_senvtB_to_XS`

```bash
python main.py --data SHL2023 --input-size 496 --student senvt-XS --model senvt-B --teacher-path dataset/SENvT-u4/1000k_task4/best.pth --distillation-type soft --distillation-alpha 0.5 --distillation-tau 2.0 --epochs 50 --batch-size 64 --lr 3e-4 --warmup-epochs 10 --mixup 0 --cutmix 0 --clip-grad 1.0 --output_dir ./runs/SHL2023_senvtB_to_XS
```

#### (C) senvt-B → senvt-S（soft KD）

* 保存先：`./runs/SHL2023_senvtB_to_S`

```bash
python main.py --data SHL2023 --input-size 496 --student senvt-S --model senvt-B --teacher-path dataset/SENvT-u4/1000k_task4/best.pth --distillation-type soft --distillation-alpha 0.5 --distillation-tau 2.0 --epochs 50 --batch-size 64 --lr 3e-4 --warmup-epochs 10 --mixup 0 --cutmix 0 --clip-grad 1.0 --output_dir ./runs/SHL2023_senvtB_to_S
```

---

## 5. Teacher tuning（MOMENT teacherの学習）

### MOMENT-1-large を SHL-2023 で微調整

```bash
python moment_train_shl2023.py --acc-path ../SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Acc.npy --lab-path ../SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Label.npy --epochs 50 --batch-size 128 --lr 1e-4
```

```bash
python moment_train_shl2023.py --acc-path ../SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Acc.npy --lab-path ../SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Label.npy --epochs 50 --batch-size 1024 --lr 3e-5 --finetune
```

---

## 6. E-DTD 実行

> `main_di.py` と書いているが、実体が `main_dist.py` の可能性がある。実際のファイル名に合わせて修正する。

* 保存先：`./runs/SHL2023_senvtB_to_S_EDTD`

```bash
python main_di.py --data SHL2023 --input-size 496 --student senvt-S --model senvt-B --teacher-path dataset/SENvT-u4/1000k_task4/best.pth --distillation-type soft --distillation-alpha 0.5 --distillation-tau 2.0 --epochs 50 --batch-size 64 --lr 3e-4 --warmup-epochs 10 --mixup 0 --cutmix 0 --clip-grad 1.0 --use-edtd --output_dir ./runs/SHL2023_senvtB_to_S_EDTD
```

---

## 7. 2段階目蒸留（Second stage distillation）

### 7.1 共通（パス変数）

```bash
FIRST=./runs/First
SECOND=./runs/Second
TEACHER_XS="$FIRST/SHL2023_senvtB_to_XS/best_checkpoint.pth"
TEACHER_S="$FIRST/SHL2023_senvtB_to_S/best_checkpoint.pth"

mkdir -p "$SECOND"
```

---

### 7.2 SENvT-XS → students

```bash
set -e

FIRST=./runs/First
SECOND=./runs/Second
TEACHER_XS="$FIRST/SHL2023_senvtB_to_XS/best_checkpoint.pth"

OUT_ROOT="$SECOND/XS"
mkdir -p "$OUT_ROOT"

COMMON="--data SHL2023 --input-size 496 \
        --model senvt-XS --teacher-path $TEACHER_XS \
        --distillation-type soft --distillation-alpha 0.7 --distillation-tau 2.5 \
        --epochs 100 --batch-size 128 \
        --patch_size 16 --reservoir_size 1000 \
        --mixup 0 --cutmix 0 --clip-grad 1.0"

students=(
  DeepConvLSTM100
  Resnet_L
)

for s in "${students[@]}"; do
  OUT_DIR="$OUT_ROOT/SHL2023_XS_to_${s}"
  echo "[XS] -> $s  ==>  $OUT_DIR"
  python main.py $COMMON --student "$s" --output_dir "$OUT_DIR"
done
```

---

### 7.3 SENvT-S → students

```bash
set -e

FIRST=./runs/First
SECOND=./runs/Second
TEACHER_S="$FIRST/SHL2023_senvtB_to_S/best_checkpoint.pth"

OUT_ROOT="$SECOND/S"
mkdir -p "$OUT_ROOT"

COMMON="--data SHL2023 --input-size 496 \
        --model senvt-S --teacher-path $TEACHER_S \
        --distillation-type soft --distillation-alpha 0.7 --distillation-tau 2.5 \
        --epochs 100 --batch-size 128 \
        --patch_size 16 --reservoir_size 1000 \
        --mixup 0 --cutmix 0 --clip-grad 1.0"

students=(
  DeepConvLSTM100
  Resnet_S
  Resnet_M
  Resnet_L
  Transformer
  MLPMixer
  PRC
  PESAC
)

for s in "${students[@]}"; do
  OUT_DIR="$OUT_ROOT/SHL2023_S_to_${s}"
  echo "[S] -> $s  ==>  $OUT_DIR"
  python main.py $COMMON --student "$s" --output_dir "$OUT_DIR"
done
```

---

### 7.4 SENvT-B → students（直接蒸留）

```bash
set -e

TEACHER_B="dataset/SENvT-u4/1000k_task4/best.pth"
OUT_ROOT=./runs/B_to_students

COMMON="--data SHL2023 --input-size 496 \
        --model senvt-B --teacher-path $TEACHER_B \
        --distillation-type soft --distillation-alpha 0.7 --distillation-tau 2.5 \
        --epochs 100 --batch-size 128 \
        --patch_size 16 --reservoir_size 1000 \
        --mixup 0 --cutmix 0 --clip-grad 1.0"

students=(
  MLPMixer
  PRC
  PESAC
)

for s in "${students[@]}"; do
  OUT_DIR="$OUT_ROOT/SHL2023_B_to_${s}"
  echo "[B] -> $s  ==>  $OUT_DIR"
  python main.py $COMMON --student "$s" --output_dir "$OUT_DIR"
done
```

---

## 8. Known issues / Notes

* `senvt-B -> senvt-XS` で epoch0中に NaN が出たことがある

  * 対策：mixup/cutmix off + clip-grad + warmup増加

* `main_di.py` / `main_dist.py` のファイル名は要確認

  * README内のコマンドは実体ファイル名に合わせて修正する

---

## 9. TODO（再開時チェックリスト）

* [ ] `./runs/*/log.txt` を見て、各runの best acc / best epoch を抜き出して表にする
* [ ] `First` / `Second` のディレクトリ構造が現在の `runs/` と一致しているか確認
* [ ] E-DTDルート（`--use-edtd`）が動く状態か（main_dist側のインデント崩れ等）を確認
* [ ] NaNが出る条件（mixup/cutmix, lr, tau, alpha）を整理し、再現性チェック
