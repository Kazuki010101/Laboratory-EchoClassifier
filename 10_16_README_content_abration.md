# TGECのContent Loss除去アブレーション：実装・実験ガイド

> 対象: `Kazuki010101/Laboratory-EchoClassifier` の `main` ブランチ（2026-10-16 確認）  
> 目的: **要約トークンの構造をそのまま残し、Content Distillation の学習上の寄与のみを切り分ける**。  
> この資料は変更案です。GitHub上のファイルは変更していません。

## 1. 最初に結論

**`models/TeacherGuidedEvidenceCondensation.py` と `loss_func.py` の変更は不要。** 既存の `--tgec-content-weight 0` を指定すれば、要約トークンは生成される一方で、`L_content` は計算・加算されない。

元のTGEC:

```text
student=TGEC_PRC, distillation-type=soft
route-weight=1, content-weight=1
L = 0.3 L_CE + 0.7 L_KD + 1 L_route + 1 L_content
```

今回のアブレーション（TGEC-NoContent）:

```text
student=TGEC_PRC, distillation-type=soft
route-weight=1, content-weight=0
L = 0.3 L_CE + 0.7 L_KD + 1 L_route
```

**`--distillation-type none` にしてはいけない。** それではLogit KDもRoute Distillationも失われ、Content Lossだけを外した比較にならない。

## 2. GitHub実装で確認した根拠

| ファイル | 現行実装の確認 | 対応 |
|---|---|---|
| `models/TeacherGuidedEvidenceCondensation.py` | `mode="tgec"` なら常に省略パッチの平均とRMSを計算し、`summary_encoder`で1個の要約トークンを生成 | **変更不要** |
| `loss_func.py` | `and self.content_weight > 0` の条件を満たすときだけContent Lossを計算 | **変更不要** |
| `main.py` | `--tgec-content-weight` を受け取る学習引数が、既存シェルから渡されている | **既存引数を使用** |
| `scripts/shl2023_senvt_kd/20_run_prc_condensation_study.sh` | `tgec)` 分岐で `CONTENT_W=1` に設定し、出力名は `tgec_seed${SEED}` | **新規条件は別スクリプト推奨** |
| `scripts/shl2023_senvt_kd/17_finetune_checkpoint_user23.sh` | `--finetune` で重みだけ読み込み、`--distillation-type none` でUser2/3をFine-tuning | **そのまま再利用可能** |
| `scripts/shl2023_senvt_kd/24_eval_finetuned_models_user23_test.sh` | 対象4+2条件の名前が固定 | **新条件は専用評価コマンドが必要** |
| `scripts/shl2023_senvt_kd/collect_prc_results.py` | 正規表現と手法名一覧が6条件に固定 | **新条件の自動集計には修正が必要** |

## 3. 修正してはいけない箇所

以下を変更すると、アブレーションの意味が変わる。

- `mode="tgec"` を `teacher_skip` に変更しない。そうすると要約トークン自体が消える。
- `summary_encoder` を削除しない。パラメータ数、モデルサイズ、推論処理も変化する。
- `--tgec-route-weight 1` を0にしない。Route蒸留の条件を保つ。
- patch size = 16、keep ratio = 0.5、teacher layer = 1、reservoir size = 1000、alpha = 0.7、tau = 2.5 を変えない。
- **学習済みTGECチェックポイントをそのまま評価するだけでは不十分。** Content LossなしでUser1から再学習する。
- **既存の `tgec_seed0/1/2` などに上書きしない。** 新しい出力名を使用する。

## 4. 推奨: 新規スクリプトを追加してUser1→Fine-tuning→独立Testを実行

新しく次のファイルを作る:

`scripts/shl2023_senvt_kd/28_run_tgec_no_content.sh`

内容（リポジトリの `config.sh` と `17_finetune_checkpoint_user23.sh` を流用）：

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="${1:-smoke}"
case "$MODE" in
  smoke) USER1_EPOCHS=1; USER1_WARMUP=0; FT_MODE=smoke; SUFFIX=_smoke ;;
  full) USER1_EPOCHS="$STUDENT_EPOCHS"; USER1_WARMUP="$STUDENT_WARMUP_EPOCHS"; FT_MODE=full; SUFFIX= ;;
  *) echo "Usage: bash $0 smoke|full" >&2; exit 2 ;;
esac

for f in "$SHL_DISTILL_ACC" "$SHL_DISTILL_LABEL" "$TEACHER_B_CHECKPOINT" \
         "$SHL_USER23_ROOT/Hips_Acc.npy" "$SHL_USER23_ROOT/Hips_Label.npy" "$USER23_SPLIT"; do
  [[ -f "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }
done

cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for seed in ${SEEDS:-0 1 2}; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  tag="tgec_no_content_seed${seed}"
  user1_out="$STUDENT_DISTILLATION_ROOT/prc_condensation/${tag}${SUFFIX}"
  ft_out="$STUDENT_FINETUNE_ROOT/user23/${tag}${SUFFIX}"
  eval_out="$HELDOUT_EVALUATION_ROOT/user23_test/${tag}${SUFFIX}"

  # Phase A: User1 distillation training
  if [[ ! -f "$user1_out/summary.json" ]]; then
    [[ ! -e "$user1_out/checkpoint.pth" ]] || {
      echo "Partial checkpoint found: $user1_out; handle resume separately" >&2; exit 1;
    }
    mkdir -p "$user1_out"
    python main.py --data SHL2023 --input-size "$INPUT_SIZE" \
      --epochs "$USER1_EPOCHS" --batch-size "$STUDENT_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" --device "$DEVICE" --seed "$seed" \
      --model senvt-B --student TGEC_PRC --teacher-path "$TEACHER_B_CHECKPOINT" \
      --distillation-type soft --distillation-alpha "$DISTILLATION_ALPHA" \
      --distillation-tau "$DISTILLATION_TAU" \
      --patch_size 16 --reservoir_size 1000 --patch_keep_ratio 0.5 \
      --tgec-teacher-layer 1 --tgec-route-weight 1 \
      --tgec-content-weight 0 \
      --mixup 0 --cutmix 0 --smoothing 0.1 \
      --clip-grad 1 --opt adamw --weight-decay "$WEIGHT_DECAY" \
      --lr "$STUDENT_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE" \
      --warmup-epochs "$USER1_WARMUP" --no-model-ema --unscale-lr \
      --output_dir "$user1_out" 2>&1 | tee "$user1_out/train_stdout.log"
  fi
  [[ -f "$user1_out/best_checkpoint.pth" ]] || { echo "Missing User1 best checkpoint" >&2; exit 1; }

  # Phase B: downstream Fine-tuning (reuses existing helper).
  # The helper sources config.sh and sets SEED=0 by default, so for seeds 1/2
  # this helper cannot be called unchanged. Run main.py directly instead.
  if [[ ! -f "$ft_out/summary.json" ]]; then
    [[ ! -e "$ft_out/checkpoint.pth" ]] || { echo "Partial fine-tune: $ft_out" >&2; exit 1; }
    mkdir -p "$ft_out"
    if [[ "$MODE" == smoke ]]; then ft_epochs=1; ft_warmup=0;
    else ft_epochs="$DOWNSTREAM_EPOCHS"; ft_warmup="$DOWNSTREAM_WARMUP_EPOCHS"; fi
    python main.py --data SHL2023_user23_finetune \
      --data-path "$SHL_USER23_ROOT" --split-indices "$USER23_SPLIT" \
      --finetune "$user1_out/best_checkpoint.pth" \
      --input-size "$INPUT_SIZE" --epochs "$ft_epochs" \
      --batch-size "$DOWNSTREAM_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
      --device "$DEVICE" --seed "$seed" --student TGEC_PRC \
      --distillation-type none --patch_size 16 --reservoir_size 1000 \
      --reservoir_rank 64 --patch_keep_ratio 0.5 --input_rank 16 \
      --mixup 0 --cutmix 0 --smoothing 0.1 --clip-grad 1 \
      --opt adamw --weight-decay "$WEIGHT_DECAY" \
      --lr "$DOWNSTREAM_LEARNING_RATE" --min-lr "$MIN_LEARNING_RATE" \
      --warmup-epochs "$ft_warmup" --no-model-ema --unscale-lr \
      --output_dir "$ft_out" 2>&1 | tee "$ft_out/train_stdout.log"
  fi
  [[ -f "$ft_out/best_checkpoint.pth" ]] || { echo "Missing fine-tuned best checkpoint" >&2; exit 1; }

  # Phase C: held-out User2/3 Test
  if [[ ! -f "$eval_out/evaluation_summary.json" ]]; then
    mkdir -p "$eval_out"
    python main.py --eval --data SHL2023_user23_test \
      --data-path "$SHL_USER23_ROOT" --split-indices "$USER23_SPLIT" \
      --input-size "$INPUT_SIZE" --batch-size "$DOWNSTREAM_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" --device "$DEVICE" \
      --seed "$seed" --student TGEC_PRC \
      --resume "$ft_out/best_checkpoint.pth" --distillation-type none \
      --patch_size 16 --reservoir_size 1000 --reservoir_rank 64 \
      --patch_keep_ratio 0.5 --input_rank 16 \
      --mixup 0 --cutmix 0 --no-model-ema \
      --output_dir "$eval_out" 2>&1 | tee "$eval_out/eval_stdout.log"
  fi
  echo "[OK] $tag"
done
```

### 実行順

```bash
# 1. 構文検査
bash -n scripts/shl2023_senvt_kd/28_run_tgec_no_content.sh

# 2. まずseed 0で短い動作確認
SEEDS="0" bash scripts/shl2023_senvt_kd/28_run_tgec_no_content.sh smoke

# 3. 正式実験: seed 0,1,2を新たに学習
SEEDS="0 1 2" bash scripts/shl2023_senvt_kd/28_run_tgec_no_content.sh full
```

注意: 既存の `config.sh` は `SEED=0` を固定しているが、この新規スクリプトは `--seed "$seed"` を明示している。User2/3の分割ファイルは既存の固定分割を使う（再生成しない）。既存ファイルが途中停止状態にあるときは、安全のため停止する仕様なので、内容を確認してから別途再開する。

## 5. 正しくContent Lossだけ外れているか確認

`loss_func.py` では次の条件がある。

```python
if (
    guided_distillation_enabled
    and aux.get("summary_token") is not None
    and self.content_weight > 0
):
    # content loss calculation
```

今回 `content_weight=0` なので、このブロックには入らない。一方、`guided_distillation_enabled=True` であり、Route Loss、Logit KDは引き続き計算される。`PatchReservoir.forward()` の `mode="tgec"` 分岐には影響しないため、Summary Encoderは残る。

Smoke終了後、以下をチェックする。

```bash
find experiments/shl2023_senvt_kd/student_distillation/prc_condensation \
  -path '*tgec_no_content*' -name summary.json -print

# 実験設定記録が保存されているか
python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/shl2023_senvt_kd/student_distillation/prc_condensation/tgec_no_content_seed0_smoke/summary.json')
d = json.loads(p.read_text())
print('distillation_type:', d.get('distillation_type'))
print('route weight:', d.get('tgec_route_weight'))
print('content weight:', d.get('tgec_content_weight'))
PY
```

期待: `soft / 1 / 0`。学習実行ログにRoute Lossの記録があるか確認する。`summary.json` の設定欄のキーがバージョンにより異なる場合は、保存済みの実行引数・学習ログも確認する。

## 6. 結果集計は既存の6条件集約器だけでは不可

現行の `collect_prc_results.py` の `TAG_PATTERN` は `tgec_no_content_seedN` を認識しない。したがって、**単に新実験を走らせても `prc_seed_summary.csv` に追加されない**。

最初は専用の集計スクリプトで、元TGECとNoContentの評価JSONを比較するのが安全。以下を `scripts/shl2023_senvt_kd/29_compare_tgec_no_content.py` として保存する。

```python
from pathlib import Path
import json
import statistics

root = Path('experiments/shl2023_senvt_kd/heldout_evaluation/user23_test')
for method in ['tgec', 'tgec_no_content']:
    acc, f1 = [], []
    for seed in (0, 1, 2):
        path = root / f'{method}_seed{seed}' / 'evaluation_summary.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        d = json.loads(path.read_text())
        a = float(d['acc1'])
        f = float(d['f1_macro'])
        acc.append(a * 100 if abs(a) <= 1 else a)
        f1.append(f * 100 if abs(f) <= 1 else f)
    print(method)
    print('  acc seeds:', acc)
    print('  acc mean±sd:', statistics.mean(acc), statistics.stdev(acc))
    print('  f1  seeds:', f1)
    print('  f1  mean±sd:', statistics.mean(f1), statistics.stdev(f1))
```

実行:

```bash
python scripts/shl2023_senvt_kd/29_compare_tgec_no_content.py
```

公表済みのTGEC精度（88.15±0.46%）と比較し、数値は新たに得られた実験結果をそのまま使用する。数字を先回りして記述しない。

## 7. 何が分かり、何は分からないか

| 比較 | 分かること | 分からないこと |
|---|---|---|
| TGEC vs TGEC-NoContent | **要約機構を保ったままContent Lossを加えたことの効果** | 要約機構そのものの効果 |
| TGEC-NoContent vs TG-Skip | Content Lossなしで要約トークンを追加した効果（ただしパラメータ数と状態更新数も変わる） | 完全に同じ計算量下での効果 |
| TGEC vs TG-Skip | 凝縮機構とContent Lossを合わせた効果 | 両機構の個別寄与 |

学習済みTGECのSummary Encoderはそのまま存在するため、推論時のモデル構造・状態更新数はTGECと同じ **19回**。同じアーキテクチャなので、推論計算量の理論的条件も同等（ただし実測の遅延値は追加確認が必要）。

重要: Content Lossを外した条件だけでは「平均とRMSによる窓統計の寄与」と「非選択情報の凝縮」の説明を完全には分離できない。査読(4)全体に対応するには、必要に応じて **全パッチ＋全パッチ統計要約トークン** の比較も追加する。

## 8. 論文へ追加する場合の文章（結果取得後）

### 実験条件（事前に追記可能）

> 内容蒸留の寄与を検証するため、TGECから内容蒸留損失のみを除いた条件（TGEC w/o Content Distillation）を追加した。この条件では、パッチ選択器、経路蒸留、非選択パッチの統計量に基づく要約トークン生成機構およびモデル構造をTGECと共通とし、内容蒸留損失の重みのみを0とした。その他の学習条件と評価データ分割も共通とした。

### 結果・考察（実測後に埋める）

> TGEC w/o Content DistillationのAccuracyは［実測値］%、macro F1は［実測値］%であり、TGECとの差はそれぞれ［実測値］ポイント、［実測値］ポイントであった。この比較は、非選択パッチの要約機構を共通とした条件における内容蒸留損失の影響を示す。

**値が悪化しても上昇しても、そのまま報告する。** 優位差が小さい場合は「内容蒸留が単独で有効」と断言しない。

## 9. 最低限のチェックリスト

- [ ] 元のTGECコード・結果を保持
- [ ] `--student TGEC_PRC` のまま
- [ ] `--tgec-route-weight 1`、`--tgec-content-weight 0`
- [ ] `--distillation-type soft` でUser1学習
- [ ] seed 0 のSmokeが完了
- [ ] seed 0/1/2 のUser1 Full学習が完了
- [ ] 各SeedのUser1ベストチェックポイントからUser2/3 Fine-tuning
- [ ] User2/3 Testは既存の固定分割を再使用
- [ ] 3 Seedの精度平均・標本標準偏差を算出
- [ ] TGEC / TGEC-NoContent / TG-Skipを比較
- [ ] 論文中で「凝縮」と「内容蒸留」の寄与を混同しない

---

### 参照した現行コード

- [TeacherGuidedEvidenceCondensation.py](https://github.com/Kazuki010101/Laboratory-EchoClassifier/blob/main/models/TeacherGuidedEvidenceCondensation.py)
- [loss_func.py](https://github.com/Kazuki010101/Laboratory-EchoClassifier/blob/main/loss_func.py)
- [20_run_prc_condensation_study.sh](https://github.com/Kazuki010101/Laboratory-EchoClassifier/blob/main/scripts/shl2023_senvt_kd/20_run_prc_condensation_study.sh)
- [25_run_additional_prc_experiments.sh](https://github.com/Kazuki010101/Laboratory-EchoClassifier/blob/main/scripts/shl2023_senvt_kd/25_run_additional_prc_experiments.sh)
