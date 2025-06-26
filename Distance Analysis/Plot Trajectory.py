
import os

json_path = ("/Users/naokihiratani/Documents/HMM_RIS/155_neurons_810b1e07-009e-4ebe-930a-915e4cd8ece4.json")
save_dir = "/Users/naokihiratani/Desktop/ICe_trajectory"
os.makedirs(save_dir, exist_ok=True)



import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import re
from matplotlib import cm



filename = os.path.basename(json_path)
match = re.search(r'(\d+)_neurons', filename)
neuron_count = int(match.group(1)) if match else "unknown"

rt_range = (0.1, 0.6)
n_sample = 3
# === 加载 JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

session_id = data["session_id"]
results_by_bin = data["results_by_bin_size"]
bin_keys_sorted = sorted(results_by_bin.keys(), key=lambda x: float(x))
largest_bin = bin_keys_sorted[-1]

pdf_path = os.path.join(save_dir, f"{neuron_count}_neurons_{session_id}_combined.pdf")

with PdfPages(pdf_path) as pdf:
    endpoint_plots = []
    for bin_label in bin_keys_sorted:
        content = results_by_bin[bin_label]
        axis_labels = content["global_vectors"]["axis_labels"]
        global_vecs = content["global_vectors"]
        traj = content["projections"]
        dim = len(axis_labels)

        rt_vals = sorted({float(rt) for label in ["left", "right"] for rt in traj[label]})
        rt_vals = [rt for rt in rt_vals if rt_range[0] <= rt <= rt_range[1]]

        ncols, nrows = 3, int(np.ceil(len(rt_vals) / 3))

        for mode in ['early', 'late']:
            fig = plt.figure(figsize=(ncols * 4, nrows * 4))
            for i, rt_val in enumerate(rt_vals):
                ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d' if dim == 3 else None)
                plotted_legend = set()
                true_n = {}

                for label, color, light1, light2 in zip(
                    ["left", "right"], ["red", "blue"], ["mistyrose", "lightblue"], ["indianred", "cornflowerblue"]
                ):
                    rt_str = str(rt_val)
                    indiv_data = traj.get(label, {}).get(rt_str, [])
                    total_n = len(indiv_data)
                    true_n[label] = total_n

                    first_n = indiv_data[:n_sample]
                    last_n = indiv_data[-n_sample:] if total_n >= n_sample else []

                    subsampled = first_n if mode == 'early' else last_n
                    true = len(subsampled)
                    light_color = light1 if mode == 'early' else light2

                    for traj_pts in subsampled:
                        traj_pts = [pt[:dim] for pt in traj_pts]
                        X = [pt[0] for pt in traj_pts]
                        Y = [pt[1] for pt in traj_pts]
                        if dim == 3:
                            Z = [pt[2] for pt in traj_pts]
                            line = ax.plot(X, Y, Z, color=light_color, alpha=0.5)
                        else:
                            line = ax.plot(X, Y, color=light_color, alpha=0.5)
                        leg_label = f"{mode} {label} (sample = {true})"
                        if leg_label not in plotted_legend:
                            line[0].set_label(leg_label)
                            plotted_legend.add(leg_label)

                    if total_n > 0:
                        min_len = min(len(traj) for traj in indiv_data)
                        traj_aligned = [np.array([pt[:dim] for pt in traj[:min_len]]) for traj in indiv_data]
                        traj_mean = np.mean(traj_aligned, axis=0)
                        X = traj_mean[:, 0]
                        Y = traj_mean[:, 1]
                        if dim == 3:
                            Z = traj_mean[:, 2]
                            line = ax.plot(X, Y, Z, color=color, linewidth=2.5)
                            ax.scatter(X[0], Y[0], Z[0], s=60, facecolors='none', edgecolors=color)
                            ax.scatter(X[-1], Y[-1], Z[-1], s=60, color=color)
                        else:
                            line = ax.plot(X, Y, color=color, linewidth=2.5)
                            ax.scatter(X[0], Y[0], s=60, facecolors='none', edgecolors=color)
                            ax.scatter(X[-1], Y[-1], s=60, color=color)
                        if f"avg {label}" not in plotted_legend:
                            line[0].set_label(f"avg {label}")
                            plotted_legend.add(f"avg {label}")

                ax.set_title(f"RT={rt_val:.2f}\nleft n={true_n.get('left', 0)}, right n={true_n.get('right', 0)}")
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                if dim == 3:
                    ax.set_zlabel(axis_labels[2])
                ax.legend(fontsize=6)

            plt.suptitle(f"{bin_label} - {mode.title()} Trials\nSession {session_id} (n={neuron_count})", fontsize=14)
            plt.tight_layout()
            pdf.savefig()
            plt.close()


        count_data = []
        for label in ['left', 'right']:
            for rt_str, trials in traj.get(label, {}).items():
                rt_val = float(rt_str)
                if rt_range[0] <= rt_val <= rt_range[1]:
                    count_data.append({'rt': rt_val, 'label': label, 'count': len(trials)})
        if count_data:
            df = pd.DataFrame(count_data)
            df_pivot = df.pivot_table(index='rt', columns='label', values='count', fill_value=0)
            df_pivot = df_pivot.sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            df_pivot.plot(kind='bar', ax=ax)
            ax.set_ylabel("# of trials")
            ax.set_title(f"RT Bin Trial Count Histogram ({bin_label})")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # === Collect endpoint plots (for last pages) ===
        if bin_label == largest_bin:
            endpoint_plots.append((traj, global_vecs, axis_labels, dim, rt_vals))

    # === Final: Endpoint plots ===
    for traj, global_vecs, axis_labels, dim, rt_vals in endpoint_plots:
        for mode, title in zip(['label', 'rt'], ["By Label", "By RT bin"]):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)
            cmap = cm.get_cmap('tab10' if mode == 'label' else 'viridis')
            norm = plt.Normalize(min(rt_vals), max(rt_vals))

            for label in ['left', 'right']:
                for rt_str, trials in traj.get(label, {}).items():
                    color = cmap(norm(float(rt_str))) if mode == 'rt' else ('blue' if label == 'left' else 'red')
                    for traj_pts in trials:
                        pt = traj_pts[-1][:dim]
                        if dim == 3:
                            ax.scatter(pt[0], pt[1], pt[2], color=color, alpha=0.5)
                        else:
                            ax.scatter(pt[0], pt[1], color=color, alpha=0.5)
            for label, color in zip(['hold', 'left', 'right'], ['black', 'blue', 'red']):
                vec = global_vecs[label][:dim]
                if dim == 3:
                    ax.scatter(*vec, color=color, marker='x', s=60, label=f'global {label}')
                else:
                    ax.scatter(vec[0], vec[1], color=color, marker='x', s=60, label=f'global {label}')
            ax.set_title(f"Endpoints {title} ({bin_label})")
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            if dim == 3:
                ax.set_zlabel(axis_labels[2])
            ax.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
