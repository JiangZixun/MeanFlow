import os

import matplotlib.pyplot as plt
import numpy as np


def build_data():
    clean = {
        "label": "clean",
        "Coarse MSE": 44.55,
        "Fine-grained MSE": 53.87,
        "Coarse MAE": 3.296,
        "Fine-grained MAE": 4.129,
        "Coarse PSNR": 24.42,
        "Fine-grained PSNR": 23.79,
        "Coarse SSIM": 0.701,
        "Fine-grained SSIM": 0.672,
        "Coarse FVD": 130.40,
        "Fine-grained FVD": 67.56,
    }

    mild = [
        clean,
        {
            "label": "blur",
            "Coarse MSE": 51.36,
            "Fine-grained MSE": 63.83,
            "Coarse MAE": 3.52,
            "Fine-grained MAE": 4.50,
            "Coarse PSNR": 23.92,
            "Fine-grained PSNR": 23.13,
            "Coarse SSIM": 0.675,
            "Fine-grained SSIM": 0.634,
            "Coarse FVD": 271.01,
            "Fine-grained FVD": 133.48,
        },
        {
            "label": "noise",
            "Coarse MSE": 50.48,
            "Fine-grained MSE": 57.29,
            "Coarse MAE": 3.88,
            "Fine-grained MAE": 4.35,
            "Coarse PSNR": 23.91,
            "Fine-grained PSNR": 23.51,
            "Coarse SSIM": 0.542,
            "Fine-grained SSIM": 0.617,
            "Coarse FVD": 124.31,
            "Fine-grained FVD": 112.69,
        },
        {
            "label": "deform",
            "Coarse MSE": 51.80,
            "Fine-grained MSE": 82.40,
            "Coarse MAE": 3.60,
            "Fine-grained MAE": 5.19,
            "Coarse PSNR": 23.72,
            "Fine-grained PSNR": 21.98,
            "Coarse SSIM": 0.665,
            "Fine-grained SSIM": 0.584,
            "Coarse FVD": 161.80,
            "Fine-grained FVD": 164.22,
        },
        {
            "label": "bias",
            "Coarse MSE": 65.34,
            "Fine-grained MSE": 71.87,
            "Coarse MAE": 4.91,
            "Fine-grained MAE": 5.00,
            "Coarse PSNR": 23.20,
            "Fine-grained PSNR": 22.99,
            "Coarse SSIM": 0.690,
            "Fine-grained SSIM": 0.667,
            "Coarse FVD": 207.08,
            "Fine-grained FVD": 79.25,
        },
        {
            "label": "scale",
            "Coarse MSE": 51.19,
            "Fine-grained MSE": 62.11,
            "Coarse MAE": 4.58,
            "Fine-grained MAE": 5.18,
            "Coarse PSNR": 23.81,
            "Fine-grained PSNR": 23.08,
            "Coarse SSIM": 0.701,
            "Fine-grained SSIM": 0.671,
            "Coarse FVD": 198.73,
            "Fine-grained FVD": 87.87,
        },
        {
            "label": "dropout",
            "Coarse MSE": 591.38,
            "Fine-grained MSE": 281.27,
            "Coarse MAE": 9.22,
            "Fine-grained MAE": 10.98,
            "Coarse PSNR": 14.16,
            "Fine-grained PSNR": 16.92,
            "Coarse SSIM": 0.135,
            "Fine-grained SSIM": 0.175,
            "Coarse FVD": 736.72,
            "Fine-grained FVD": 815.06,
        },
        {
            "label": "all",
            "Coarse MSE": 609.32,
            "Fine-grained MSE": 319.31,
            "Coarse MAE": 10.24,
            "Fine-grained MAE": 11.70,
            "Coarse PSNR": 14.03,
            "Fine-grained PSNR": 16.39,
            "Coarse SSIM": 0.104,
            "Fine-grained SSIM": 0.158,
            "Coarse FVD": 736.70,
            "Fine-grained FVD": 784.45,
        },
    ]

    strong = [
        clean,
        {
            "label": "blur",
            "Coarse MSE": 61.02,
            "Fine-grained MSE": 71.83,
            "Coarse MAE": 3.82,
            "Fine-grained MAE": 4.71,
            "Coarse PSNR": 23.31,
            "Fine-grained PSNR": 22.67,
            "Coarse SSIM": 0.663,
            "Fine-grained SSIM": 0.629,
            "Coarse FVD": 443.06,
            "Fine-grained FVD": 196.28,
        },
        {
            "label": "noise",
            "Coarse MSE": 79.35,
            "Fine-grained MSE": 81.62,
            "Coarse MAE": 5.60,
            "Fine-grained MAE": 5.52,
            "Coarse PSNR": 22.17,
            "Fine-grained PSNR": 22.05,
            "Coarse SSIM": 0.269,
            "Fine-grained SSIM": 0.434,
            "Coarse FVD": 297.71,
            "Fine-grained FVD": 353.94,
        },
        {
            "label": "deform",
            "Coarse MSE": 70.32,
            "Fine-grained MSE": 118.01,
            "Coarse MAE": 4.26,
            "Fine-grained MAE": 6.26,
            "Coarse PSNR": 22.40,
            "Fine-grained PSNR": 20.40,
            "Coarse SSIM": 0.638,
            "Fine-grained SSIM": 0.527,
            "Coarse FVD": 283.46,
            "Fine-grained FVD": 300.06,
        },
        {
            "label": "bias",
            "Coarse MSE": 154.33,
            "Fine-grained MSE": 167.37,
            "Coarse MAE": 9.26,
            "Fine-grained MAE": 9.64,
            "Coarse PSNR": 19.72,
            "Fine-grained PSNR": 19.78,
            "Coarse SSIM": 0.665,
            "Fine-grained SSIM": 0.644,
            "Coarse FVD": 315.36,
            "Fine-grained FVD": 138.87,
        },
        {
            "label": "scale",
            "Coarse MSE": 147.92,
            "Fine-grained MSE": 149.63,
            "Coarse MAE": 9.56,
            "Fine-grained MAE": 9.37,
            "Coarse PSNR": 19.80,
            "Fine-grained PSNR": 19.66,
            "Coarse SSIM": 0.694,
            "Fine-grained SSIM": 0.658,
            "Coarse FVD": 266.32,
            "Fine-grained FVD": 134.18,
        },
        {
            "label": "dropout",
            "Coarse MSE": 1410.94,
            "Fine-grained MSE": 561.67,
            "Coarse MAE": 18.14,
            "Fine-grained MAE": 16.28,
            "Coarse PSNR": 10.44,
            "Fine-grained PSNR": 14.08,
            "Coarse SSIM": 0.0878,
            "Fine-grained SSIM": 0.131,
            "Coarse FVD": 1162.56,
            "Fine-grained FVD": 1056.89,
        },
        {
            "label": "all",
            "Coarse MSE": 1477.67,
            "Fine-grained MSE": 706.61,
            "Coarse MAE": 21.07,
            "Fine-grained MAE": 18.50,
            "Coarse PSNR": 10.21,
            "Fine-grained PSNR": 13.10,
            "Coarse SSIM": 0.0416,
            "Fine-grained SSIM": 0.116,
            "Coarse FVD": 1175.85,
            "Fine-grained FVD": 1048.49,
        },
    ]
    return mild, strong


def plot_row(ax_row, row_data, row_title, metrics, labels, colors):
    x = np.arange(len(labels))
    width = 0.34
    metric_titles = {
        "MSE": "MSE ↓",
        "MAE": "MAE ↓",
        "PSNR": "PSNR ↑",
        "SSIM": "SSIM ↑",
        "FVD": "FVD ↓",
    }

    for col, metric in enumerate(metrics):
        ax = ax_row[col]
        coarse_key = f"Coarse {metric}"
        fine_key = f"Fine-grained {metric}"
        coarse_vals = [item[coarse_key] for item in row_data]
        fine_vals = [item[fine_key] for item in row_data]

        coarse_bars = ax.bar(
            x - width / 2,
            coarse_vals,
            width=width,
            color=colors["coarse"],
            label="Before refine",
            edgecolor="white",
            linewidth=0.7,
        )
        fine_bars = ax.bar(
            x + width / 2,
            fine_vals,
            width=width,
            color=colors["fine"],
            label="After refine",
            edgecolor="white",
            linewidth=0.7,
        )

        ax.set_title(metric_titles.get(metric, metric), fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if col == 0:
            ax.set_ylabel(row_title, fontsize=13)

        if metric == "MSE":
            ax.set_yscale("log")
            ax.set_ylim(35, 2500)
        elif metric == "MAE":
            ax.set_yscale("log")
            ax.set_ylim(2.5, 30)
        elif metric == "FVD":
            ax.set_yscale("log")
            ax.set_ylim(50, 1800)
        elif metric == "PSNR":
            ax.set_ylim(8, 26)
        elif metric == "SSIM":
            ax.set_ylim(0.0, 0.78)

        if metric in {"MSE", "MAE", "FVD"}:
            label_fmt = "{:.1f}"
        elif metric == "PSNR":
            label_fmt = "{:.2f}"
        else:
            label_fmt = "{:.3f}"

        ax.bar_label(
            coarse_bars,
            labels=[label_fmt.format(v) for v in coarse_vals],
            padding=2,
            fontsize=7,
            rotation=90,
        )
        ax.bar_label(
            fine_bars,
            labels=[label_fmt.format(v) for v in fine_vals],
            padding=2,
            fontsize=7,
            rotation=90,
        )

    ax_row[-1].legend(loc="upper right", frameon=False, fontsize=10)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mild, _ = build_data()
    labels = ["clean", "blur", "noise", "deform", "bias", "scale", "dropout", "all"]
    metrics = ["MSE", "MAE", "PSNR", "SSIM", "FVD"]
    colors = {"coarse": "#ff7f0e", "fine": "#1f77b4"}

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "xtick.major.pad": 6,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.8), constrained_layout=True)
    plot_row(axes, mild, "", metrics, labels, colors)

    fig.suptitle(
        "Sensitivity to Physical Evolution Errors and Correction by PAFM",
        fontsize=17,
        y=1.06,
    )

    png_path = os.path.join(out_dir, "rebuttal_mild_1x5_metrics.png")
    pdf_path = os.path.join(out_dir, "rebuttal_mild_1x5_metrics.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
