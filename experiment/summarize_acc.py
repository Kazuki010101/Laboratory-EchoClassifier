import os
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def safe_load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # jsonが壊れてる/空/権限など
        return None


def find_test_dirs(root: Path):
    # test101, test102 ... のみ拾う（必要なら条件を変えてOK）
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("test")]
    # test101, test102 ... を自然順に並べたいので数値でソート
    def key_fn(d: Path):
        s = d.name.replace("test", "")
        return int(s) if s.isdigit() else 10**9
    return sorted(dirs, key=key_fn)


def collect_acc1(root: Path, json_name: str = "test_best.json", key: str = "test_acc1"):
    """
    return:
      df: index=test101.., columns=model_name, values=test_acc1(float)
    """
    test_dirs = find_test_dirs(root)

    rows = {}
    all_models = set()

    for tdir in test_dirs:
        test_name = tdir.name
        rows[test_name] = {}

        # 各モデルフォルダを走査
        model_dirs = [d for d in tdir.iterdir() if d.is_dir()]
        for mdir in model_dirs:
            model_name = mdir.name
            all_models.add(model_name)

            jp = mdir / json_name
            if not jp.exists():
                rows[test_name][model_name] = None
                continue

            data = safe_load_json(jp)
            if not isinstance(data, dict):
                rows[test_name][model_name] = None
                continue

            val = data.get(key, None)
            # 数値化できるならfloatに
            try:
                val = float(val) if val is not None else None
            except Exception:
                val = None

            rows[test_name][model_name] = val

    # DataFrame化（列順はモデル名アルファベット順）
    model_list = sorted(all_models)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=model_list)

    # もし列順を固定したいなら（例：あなたの9モデル順に並べたい）
    # desired = ["DeepConvLSTM25","DeepConvLSTM50","DeepConvLSTM100","MLPMixer","PESAC","PRC","Resnet_L","Resnet_M","Resnet_S"]
    # df = df.reindex(columns=desired)

    return df


def plot_heatmap(df: pd.DataFrame, outpath: Path = None, title: str = "test_acc1 (heatmap)"):
    # matplotlibのみで簡易ヒートマップ
    plt.figure(figsize=(max(8, 0.8 * df.shape[1]), max(3, 0.6 * df.shape[0])))
    im = plt.imshow(df.values, aspect="auto")  # 色は指定しない（デフォルト）
    plt.colorbar(im, label="test_acc1")

    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(range(df.shape[0]), df.index)
    plt.title(title)

    # セルに値を文字で載せたい場合（見やすさ優先で小さめ）
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            if pd.notna(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()


def plot_models_bar_mean(df: pd.DataFrame, outpath: Path = None, title: str = "Mean test_acc1 by model"):
    means = df.mean(axis=0, skipna=True).sort_values(ascending=False)
    plt.figure(figsize=(max(8, 0.8 * len(means)), 4))
    plt.bar(means.index, means.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean(test_acc1)")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()
    
def compute_model_mean_std(df: pd.DataFrame):
    """
    return:
      summary_df: index=model_name, columns=[mean, std]
    """
    summary_df = pd.DataFrame({
        "mean": df.mean(axis=0, skipna=True),
        "std": df.std(axis=0, skipna=True)  # 標本標準偏差 (ddof=1)
    })
    return summary_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="S_to_students_pamap2 のパス")
    ap.add_argument("--json-name", type=str, default="test_best.json")
    ap.add_argument("--key", type=str, default="test_acc1")
    ap.add_argument("--out-csv", type=str, default="acc1_table.csv")
    ap.add_argument("--out-xlsx", type=str, default="acc1_table.xlsx")
    ap.add_argument("--out-heatmap", type=str, default="acc1_heatmap.png")
    ap.add_argument("--out-bar", type=str, default="acc1_model_mean.png")
    args = ap.parse_args()

    root = Path(args.root)
    df = collect_acc1(root, json_name=args.json_name, key=args.key)

    # 表を保存
    df.to_csv(args.out_csv, index=True)
    df.to_excel(args.out_xlsx, index=True)

    # 表を見やすく表示（小数点）
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(df.round(4))

    # 可視化
    plot_heatmap(df, outpath=Path(args.out_heatmap), title=f"{args.key} (heatmap)")
    plot_models_bar_mean(df, outpath=Path(args.out_bar), title=f"Mean {args.key} by model")

    summary = compute_model_mean_std(df)

    print("\n[Model-wise mean ± std]")
    print(summary.round(3))

    # 保存
    summary.to_csv("model_mean_std.csv")
    summary.to_excel("model_mean_std.xlsx")

    # 論文用（mean ± std 表記）
    summary_pm = summary.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}",
        axis=1
    )

    print("\n[Model-wise mean ± std (paper format)]")
    print(summary_pm)

if __name__ == "__main__":
    main()
