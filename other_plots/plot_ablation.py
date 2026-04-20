import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
def _setup_chinese_font() -> None:
	# 优先按字体文件路径注册，确保在 Linux 环境可稳定使用中文字体。
	font_files = [
		"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
		"/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
	]
	selected_font = None
	for font_path in font_files:
		if os.path.exists(font_path):
			font_manager.fontManager.addfont(font_path)
			selected_font = font_manager.FontProperties(fname=font_path).get_name()
			break

	if selected_font is None:
		selected_font = "Noto Sans CJK SC"

	plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
	plt.rcParams["axes.unicode_minus"] = False


def plot_ablation_metrics(save_dir: str = "other_plots/ablation_figs") -> None:
	# 指标顺序: CLIPScore, PickScore, FID, IS
	# 这里按“优化后”与“优化前”分别对应两组数值。
	scores_after = {
		"CLIPScore": 0.313,
		"PickScore": 0.2169,
		"FID": 16.505,
		"IS": 30.552,
	}
	scores_before = {
		"CLIPScore": 0.312,
		"PickScore": 0.215,
		"FID": 17.49,
		"IS": 30.109,
	}

	os.makedirs(save_dir, exist_ok=True)

	# 统一放大字号，不加图标题。
	plt.rcParams.update(
		{
			"font.size": 16,
			"axes.labelsize": 18,
			"xtick.labelsize": 16,
			"ytick.labelsize": 16,
			"legend.fontsize": 15,
		}
	)

	_setup_chinese_font()
	methods = ["优化前", "优化后"]
	x_label = "方法"
	colors = ["#9FB7CF", "#E9958A"]

	for metric in ["CLIPScore", "PickScore", "FID", "IS"]:
		values = [scores_before[metric], scores_after[metric]]

		fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=180)
		ax.bar(methods, values, color=colors, width=0.62)

		ax.set_xlabel(x_label)
		ax.set_ylabel(metric)

		v_min = min(values)
		v_max = max(values)
		span = v_max - v_min
		if span <= 0:
			span = max(abs(v_min), 1.0) * 0.02
		pad = span * 0.8
		ax.set_ylim(v_min - pad, v_max + pad)

		ax.grid(axis="y", linestyle="--", alpha=0.25)

		fig.tight_layout()
		fig.savefig(os.path.join(save_dir, f"ablation_{metric}.png"), bbox_inches="tight")
		plt.close(fig)


if __name__ == "__main__":
	plot_ablation_metrics()




