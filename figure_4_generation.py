import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from PIL import Image
import numpy as np

outdir = Path("./review")
outdir.mkdir(parents=True, exist_ok=True)

modalities = ["MRI encoder", "CoWS-Rad", "Synthetic PET", "UNICAL"]
oasis = np.array([0.744, 0.731, 0.791, 0.868])
adni  = np.array([0.718, 0.697, 0.762, 0.836])
delta = adni - oasis
y = np.arange(len(modalities))

# Colorblind-safe choices (Okabe-Ito inspired) plus redundant marker shapes.
oasis_color = "#0072B2"   # blue
adni_color  = "#D55E00"   # vermillion
line_color  = "#666666"

# ---------- Panel A ----------
fig, ax = plt.subplots(figsize=(6.7, 4.6), facecolor="white")
ax.set_facecolor("white")

for i in range(len(modalities)):
    ax.plot([adni[i], oasis[i]], [y[i], y[i]],
            color=line_color, linewidth=1.6, zorder=1)
    ax.scatter(oasis[i], y[i], marker="o", s=72,
               facecolor="white", edgecolor=oasis_color, linewidth=2.0, zorder=3)
    ax.scatter(adni[i], y[i], marker="s", s=64,
               facecolor=adni_color, edgecolor=adni_color, linewidth=1.5, zorder=3)

    # Non-overlapping value labels using point offsets.
    ax.annotate(f"{oasis[i]:.3f}", (oasis[i], y[i]),
                xytext=(0, 15), textcoords="offset points",
                ha="center", va="top", fontsize=9)
    ax.annotate(f"{adni[i]:.3f}", (adni[i], y[i]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

ax.set_yticks(y, modalities)
ax.invert_yaxis()
ax.set_xlabel("AUROC")
ax.set_xlim(0.68, 0.885)
ax.set_title("(a) Modality-level external transfer", loc="left",
             fontsize=12.5, fontweight="bold", pad=10)

ax.grid(axis="x", alpha=0.18, linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    Line2D([0], [0], marker="o", markersize=7, linestyle="None",
           markerfacecolor="white", markeredgecolor=oasis_color,
           markeredgewidth=1.8, label="OASIS-3"),
    Line2D([0], [0], marker="s", markersize=7, linestyle="None",
           markerfacecolor=adni_color, markeredgecolor=adni_color,
           label="ADNI3"),
]
ax.legend(handles=legend_handles, frameon=False, ncol=2,
          loc="upper center", bbox_to_anchor=(0.62, -0.12))

fig.tight_layout(rect=[0, 0.06, 1, 1])
panel_a = outdir / "Fig4_panelA_transfer_v2.png"
panel_a_svg = outdir / "Fig4_panelA_transfer_v2.svg"
fig.savefig(panel_a, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(panel_a_svg, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Panel B ----------
fig, ax = plt.subplots(figsize=(6.7, 4.6), facecolor="white")
ax.set_facecolor("white")

marker_shapes = ["o", "s", "^", "D"]
marker_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

ax.axvline(0, color="#444444", linestyle="--", linewidth=1.2, zorder=0)

for i in range(len(modalities)):
    ax.plot([delta[i], 0], [y[i], y[i]],
            color="#777777", linewidth=1.6, zorder=1)
    ax.scatter(delta[i], y[i], marker=marker_shapes[i], s=72,
               facecolor="white", edgecolor=marker_colors[i],
               linewidth=2.0, zorder=3)
    ax.annotate(f"{delta[i]:+.3f}", (delta[i], y[i]),
                xytext=(0, 11), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

ax.set_yticks(y, modalities)
ax.invert_yaxis()
ax.set_xlabel("ΔAUROC (ADNI3 − OASIS-3)")
ax.set_xlim(-0.041, 0.0045)
ax.set_title("(b) External AUROC change", loc="left",
             fontsize=12.5, fontweight="bold", pad=10)

ax.grid(axis="x", alpha=0.18, linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
panel_b = outdir / "Fig4_panelB_delta_v2.png"
panel_b_svg = outdir / "Fig4_panelB_delta_v2.svg"
fig.savefig(panel_b, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(panel_b_svg, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Combine panels ----------
im_a = Image.open(panel_a).convert("RGB")
im_b = Image.open(panel_b).convert("RGB")

target_h = max(im_a.height, im_b.height)
def pad_to_height(im, h):
    if im.height == h:
        return im
    canvas = Image.new("RGB", (im.width, h), "white")
    canvas.paste(im, (0, (h - im.height)//2))
    return canvas

im_a = pad_to_height(im_a, target_h)
im_b = pad_to_height(im_b, target_h)

gap = 36
canvas = Image.new("RGB", (im_a.width + gap + im_b.width, target_h), "white")
canvas.paste(im_a, (0, 0))
canvas.paste(im_b, (im_a.width + gap, 0))

preview = outdir / "Fig4_external_transfer_v2_preview.png"
tiff = outdir / "Fig4_external_transfer_v2_600dpi.tiff"
canvas.save(preview)
canvas.save(tiff, dpi=(600, 600), compression="tiff_lzw")

print("Created:")
print(preview)
print(tiff)
print(panel_a_svg)
print(panel_b_svg)
