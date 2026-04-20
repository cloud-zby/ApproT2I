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


def plot_cache_curve(save_path: str = "other_plots/cache_clip_line.png") -> None:
	cache_capacity = ["128G", "256G", "512G", "1T", "2T"]
	clip_scores = [0.3012, 0.3078, 0.3126, 0.3132]

	if len(cache_capacity) != len(clip_scores):
		n = min(len(cache_capacity), len(clip_scores))
		print(f"[提示] 数据长度不一致，将使用前 {n} 个点绘图。")
		cache_capacity = cache_capacity[:n]
		clip_scores = clip_scores[:n]

	_setup_chinese_font()
	plt.rcParams.update(
		{
			"font.size": 15,
			"axes.labelsize": 18,
			"xtick.labelsize": 14,
			"ytick.labelsize": 14,
		}
	)

	fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
	ax.plot(
		cache_capacity,
		clip_scores,
		marker="o",
		markersize=9,
		linewidth=3,
		color="#4A78B8",
		markerfacecolor="#E98973",
		markeredgecolor="#2E2E2E",
		markeredgewidth=1.2,
	)

	y_min = min(clip_scores)
	y_max = max(clip_scores)
	span = y_max - y_min
	if span <= 0:
		span = max(abs(y_min), 1.0) * 0.02
	pad = span * 0.20
	ax.set_ylim(y_min - pad, y_max + pad)

	ax.set_xlabel("缓存容量")
	ax.set_ylabel("CLIP分数")
	ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)

	fig.tight_layout()
	fig.savefig(save_path, bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	plot_cache_curve()




