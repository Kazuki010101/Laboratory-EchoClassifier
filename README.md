# EchoAttnNet

## 10/17

### 岡橋くんの提案をもとに、ChatGPTに書かせたpythonスクリプトを掲載
未実行のため、バグがある可能性が大きい

以下、実行例のイメージ
```
# 教師・生徒ロード
# moment_teacher = MOMENTPipeline.from_pretrained("path/to/moment")
# student_moment = StudentMomentModel()

best_moment = train_with_validation(
    train_loader, val_loader,
    teacher=moment_teacher,
    student=student_moment,
    config=config,
    distill_target="moment",
    save_dir="./results/moment_kd"
)


# 例: SENvT のみ蒸留
# senvt_teacher = SENvT()   # pretrained weights
# student_senvt = StudentSENvT()

best_senvt = train_with_validation(
    train_loader, val_loader,
    teacher=senvt_teacher,
    student=student_senvt,
    config=config,
    distill_target="senvt",
    save_dir="./results/senvt_kd"
)

```

### Moment, SENvTの事前学習済みモデルの使用法を確認

Momentの使用には以下のライブラリが必要
```
pip install momentfm
```

両モデルの具体的な使用法とデータの入力確認、モデルのアーキテクチャはmoment_senvt.ipynbを確認


### Reservoirモデルのさらなる軽量化（取組中）

