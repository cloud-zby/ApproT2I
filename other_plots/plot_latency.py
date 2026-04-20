import os
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_chinese_font() -> None:
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


def plot_latency_scatter(save_path: str = "other_plots/latency_clip_scatter.png") -> None:
	methods = ["Ours", "ReCon", "ReDi", "NIRVANA", "无缓存"]
	clip_scores = [0.3132, 0.3028, 0.3050, 0.2899, 0.3146]
	latency = [13.09, 12.30, 12.21, 12.16, 17.06]

	_setup_chinese_font()
	plt.rcParams.update(
		{
			"font.size": 15,
			"axes.labelsize": 18,
			"xtick.labelsize": 14,
			"ytick.labelsize": 14,
		}
	)

	fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=180)

	colors = ["#E67E73", "#79A8D8", "#8AC09A", "#C3A3D8", "#F1B56D"]
	for method, x, y, color in zip(methods, latency, clip_scores, colors):
		ax.scatter(x, y, s=230, marker="o", color=color, edgecolors="#2B2B2B", linewidths=1.8, zorder=3)
		ax.annotate(method, (x, y), xytext=(7, 6), textcoords="offset points", fontsize=13)

	x_min, x_max = min(latency), max(latency)
	y_min, y_max = min(clip_scores), max(clip_scores)
	ax.set_xlim(x_min - (x_max - x_min) * 0.10, x_max + (x_max - x_min) * 0.10)
	ax.set_ylim(y_min - (y_max - y_min) * 0.15, y_max + (y_max - y_min) * 0.15)

	ax.set_xlabel("时延")
	ax.set_ylabel("CLIP分数")
	ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

	fig.tight_layout()
	fig.savefig(save_path, bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	plot_latency_scatter()
















