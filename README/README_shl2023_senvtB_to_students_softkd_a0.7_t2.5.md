# SHL2023 ベースライン実験: SENvT-B -> Students（Soft KD, alpha=0.7, tau=2.5）

## 目的

この実験は、SHL2023 において SENvT-B を教師モデルとし、複数の学生モデルに対して一段階の知識蒸留（one-stage knowledge distillation）を行うためのベースライン実験である。

* データセット: SHL2023
* 教師モデル: SENvT-B（`dataset/SENvT-u4/1000k_task4/best.pth`）
* 蒸留方式: Soft KD
* alpha: 0.7
* tau: 2.5
* エポック数: 100
* バッチサイズ: 128
* patch_size: 16
* reservoir_size: 1000

## 出力先ディレクトリ

```bash
./runs_baseline/SENvT_B->students_soft_alpha0.7_tau_2.5_patch16_res1000
```

## 学生モデル一覧

* DeepConvLSTM25
* DeepConvLSTM50
* DeepConvLSTM100
* Resnet_S
* Resnet_M
* Resnet_L
* Transformer
* MLPMixer
* PRC
* PESAC

## 実行コマンド（一括実行）

```bash
set -e

TEACHER_B="dataset/SENvT-u4/1000k_task4/best.pth"
OUT_ROOT="./runs_baseline/SENvT_B_to_students_soft_alpha0.7_tau_2.5_patch16_res1000"

COMMON="--data SHL2023 --input-size 496 \
        --model senvt-B --teacher-path $TEACHER_B \
        --distillation-type soft --distillation-alpha 0.7 --distillation-tau 2.5 \
        --epochs 100 --batch-size 128 \
        --patch_size 16 --reservoir_size 1000 \
        --mixup 0 --cutmix 0 --clip-grad 1.0"

students=(
  DeepConvLSTM25
  DeepConvLSTM50
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
  OUT_DIR="$OUT_ROOT/SHL2023_B_to_${s}"
  echo "[B] -> $s  ==>  $OUT_DIR"
  python main.py $COMMON --student "$s" --output_dir "$OUT_DIR"
done
```

## 注意事項

* このベースラインでは `mixup=0`, `cutmix=0` とし、データ拡張を無効化している。
* 勾配クリッピングは `1.0` に設定している。
* 公平な比較のため、すべての学生モデルで同一の KD ハイパーパラメータ（alpha, tau）を使用している。
