import os, sys, math, random, re, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datasets import load_dataset, Dataset
from collections import defaultdict
from vllm import LLM, SamplingParams
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
from tqdm import tqdm
import choix
from z_reddit import create_prompt_grm_answer, extract_verdict, multi_gpu_inference, process_batch_on_gpu, create_prompt_grm_thinking
from zzzzz import compute_newcomer_elo_no_ties
from matplotlib.patches import Patch
import colorsys

def create_pivoted_dataset(dataset, step_shift, prefix):
    df = pd.DataFrame(dataset)
    
    pivoted_data = defaultdict(dict)
    for _, row in df.iterrows():
        key = (row['question'])
        pivoted_data[key].update({
            'question': row['question'],
            f'{prefix}_{row["step"] + step_shift}': row['current']
        })
    
    result_df = pd.DataFrame(list(pivoted_data.values()))
    step_cols = sorted([c for c in result_df.columns if c.startswith(prefix)], key=get_checkpoint)
    result_df = result_df[['question'] + step_cols]
    print(f"Shape: {result_df.shape}")
    return result_df

def sample_prompts(dataset, tokenizer, n, seed=42):
    rng = random.Random(seed)
    m = len(dataset)
    players = [k for k in dataset[0].keys() if k != "question"]

    prompts_thinking, prompts_answer = [], []
    for _ in tqdm(range(n)):
        qid = rng.randrange(m)
        row = dataset[qid]
        pa, pb = rng.sample(players, 2)
        ra, rb = row.get(pa), row.get(pb)
        if ra is None or rb is None: continue
        pt = create_prompt_grm_thinking(row["question"], ra, rb, tokenizer)
        pa_ = create_prompt_grm_answer(row["question"], ra, rb, tokenizer)
        meta = {"qid": qid, "player_a": pa, "player_b": pb}
        prompts_thinking.append({"prompt": pt, **meta})
        prompts_answer.append({"prompt": pa_, **meta})
    return prompts_thinking, prompts_answer

def extract_judgments(outputs):
    judgments = []
    for output in outputs:
        verdict = extract_verdict(output.outputs[0].text)
        if verdict == -1:
            judgments.append("A")
        elif verdict == 1:
            judgments.append("B")
        else:
            judgments.append("Tie")
    return judgments

def attach_judgments(prompts, judgments):
    judged = []
    for meta, j in zip(prompts, judgments):
        judged.append({
            **meta,
            "judgment": j  # A, B, or Tie
        })
    return judged


def to_pairwise_results(judged):
    results = []
    for row in judged:
        if row["judgment"] == "A":
            winner, loser = row["player_a"], row["player_b"]
        elif row["judgment"] == "B":
            winner, loser = row["player_b"], row["player_a"]
        else:
            continue
        results.append((winner, loser))
    return results
    
def push_results_to_hf(results: list, repo_id: str, split: str):
    data = [{"winner": w, "loser": l} for w, l in results]
    ds = Dataset.from_list(data)
    ds.push_to_hub(repo_id, split=split)


def compute_bt_ratings(results, players=None, reg=1.0):
    if players is None:
        players = sorted(set([p for pair in results for p in pair]))
    id_map = {p: i for i, p in enumerate(players)}
    data = [(id_map[w], id_map[l]) for (w, l) in results]
    scores = choix.opt_pairwise(n_items=len(players), data=data, alpha=reg)
    return {player: score for player, score in zip(players, scores)}


def bt_to_shifted_elo(ratings, baseline, scale_base=400/np.log(10)):
    elo = {p: s * scale_base for p, s in ratings.items()}
    rescaled_val = elo[baseline]
    shifted = {p: val - rescaled_val for p, val in elo.items()}
    return shifted


def get_checkpoint(player):
    try:
        return int(player.split('_')[-1])
    except:
        return -1


def create_color_map(labels):
    keys = list(labels.keys())
    color_map = {}
    
    for i, key in enumerate(keys):
        hue = i * 0.618033988749895 % 1
        saturation = 0.7
        value = 0.95
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        color_map[key] = hex_color
    
    return color_map

def create_color_list(num_colors):
    color_list = []
    
    for i in range(num_colors):
        hue = i * 0.618033988749895 % 1
        saturation = 0.7
        value = 0.95
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        color_list.append(hex_color)
    
    return color_list

def plot_ratings(ratings, file_name=None, save_path=None, label_mapping=None, color_mapping=None):
    items = sorted(ratings.items(), key=lambda x: x[1])
    players, scores = zip(*items)
    def get_color(player):
        for prefix in color_mapping:
            if player.startswith(prefix):
                return color_mapping[prefix]
        return '#B0B0B0'  # Default gray for unmatched players
    bar_colors = [get_color(p) for p in players]
    pd.DataFrame({'player': players, 'rating': scores}).to_csv(f"{save_path}/{file_name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(range(len(players)), scores, tick_label=players, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_ylabel("Rating", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', labelrotation=90, labelsize=4)
    max_score = max(scores) if scores else 0
    offset = max_score * 0.01 if max_score > 0 else 0.01
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + offset, f"{v:.0f}", ha='center', va='bottom', fontsize=3)
    ax.set_title(file_name, fontsize=12, pad=8)
    ax.tick_params(axis='y', labelsize=7)
    legend_labels = [
        Patch(facecolor=color_mapping[key], edgecolor='black', label=label_mapping[key])
        for key in label_mapping.keys()
    ]
    ax.legend(handles=legend_labels, loc='best', fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(f"{save_path}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {save_path}/{file_name}.png")


def plot_proxy_vs_gold(pairs, file_name, save_path, min_checkpoint=0, max_checkpoint=1000, min_x_value=0, max_x_value=10000):
    #colors = create_color_list(len(pairs) // 2)
    colors = create_color_list(len(pairs))
    all_x, all_y = [], []
    plt.figure(figsize=(10,10))
    ax = plt.gca()

    for i, (proxy_csv, prefix, label) in enumerate(pairs):
        proxy_df = pd.read_csv(proxy_csv)
        gold_df = pd.read_csv(f"{save_path}/gold.csv")
        proxy_df = proxy_df[proxy_df['player'].str.startswith(prefix)]
        proxy_df = proxy_df[proxy_df['player'].apply(get_checkpoint).between(min_checkpoint, max_checkpoint)]
        gold_df = gold_df[gold_df['player'].str.startswith(prefix)]
        gold_df = gold_df[gold_df['player'].apply(get_checkpoint).between(min_checkpoint, max_checkpoint)]
        merged = pd.merge(proxy_df, gold_df, on='player', suffixes=('_proxy', '_gold'))
        merged = merged[merged['rating_proxy'] >= min_x_value]
        merged = merged[merged['rating_proxy'] <= max_x_value]
        x, y = merged['rating_proxy'].values, merged['rating_gold'].values
        all_x.append(x)
        all_y.append(y)
        plt.scatter(x, y, label=label, color=colors[i], alpha=0.7, s=30)
        print(prefix)
        if x.size >= 3 and y.size >= 3:
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_smooth, poly(x_smooth), '-', linewidth=2, alpha=0.8, color=colors[i])
        #if "Ans" in label:
        #    plt.plot(x_smooth, poly(x_smooth), '-', linewidth=2, alpha=0.8, color=colors[i // 2], label=label)
        #else:
        #    plt.plot(x_smooth, poly(x_smooth), '--', linewidth=2, alpha=0.8, color=colors[i // 2], label=label)

    all_x = np.concatenate(all_x) if all_x else np.array([0,1])
    all_y = np.concatenate(all_y) if all_y else np.array([0,1])
    min_val, max_val = min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    plt.plot([min_val, max_val], [min_val, max_val], 'k:', lw=1, label='45° line')
    
    plt.xlabel('Proxy Rating')
    plt.ylabel('Gold Rating')
    plt.title('Proxy vs Gold Ratings')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_max_gold(pairs, file_name, save_path, top_n=5):
    colors = create_color_list(max(color_id for _, _, _, (color_id, _) in pairs) + 10)
    plt.figure(figsize=(15,8))
    ax = plt.gca()

    avg_top_ratings = []
    labels = []
    style_info = []  # Will store (color_id, is_thinking) tuples
    for _, prefix, label, (color_id, is_thinking) in pairs:
        df = pd.read_csv(f"{save_path}/gold.csv")
        df = df[df['player'].str.startswith(prefix)]
        if len(df) > 0:
            # Get average of top N ratings
            top_n_ratings = sorted(df['rating'].values, reverse=True)[:top_n]
            avg_rating = sum(top_n_ratings) / len(top_n_ratings)
            avg_top_ratings.append(avg_rating)
            labels.append(label)
            style_info.append((color_id, is_thinking))

    # Sort by average rating
    sorted_data = sorted(zip(avg_top_ratings, labels, style_info), reverse=True)
    avg_top_ratings, labels, style_info = zip(*sorted_data)
    
    # Create bars with colors based on color_id and hatching based on is_thinking
    bars = plt.bar(range(len(avg_top_ratings)), avg_top_ratings, 
                  color=[colors[cid+1] for cid, _ in style_info],  # -1 because color_ids start at 1
                  edgecolor="black", linewidth=0.5, alpha=0.9)
    
    # Apply hatching for thinking bars
    for bar, (_, is_thinking) in zip(bars, style_info):
        if is_thinking:
            bar.set_hatch('//')
    
    plt.ylabel(f"Average of Top {top_n} Ratings", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Average of Top {top_n} Gold Ratings by Model', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_steps_vs_rating(pairs, file_name, save_path):
    colors = create_color_list(len(pairs))
    plt.figure(figsize=(15,8))
    ax = plt.gca()

    for i, (proxy_csv, prefix, label) in enumerate(pairs):
        # Read and process gold data
        gold_df = pd.read_csv(f"{save_path}/gold.csv")
        gold_df = gold_df[gold_df['player'].str.startswith(prefix)]
        gold_steps = []
        gold_ratings = []
        for _, row in gold_df.iterrows():
            try:
                step = int(row['player'].split('_')[-1])
                gold_steps.append(step)
                gold_ratings.append(row['rating'])
            except:
                continue
        
        # Sort by steps
        gold_data = sorted(zip(gold_steps, gold_ratings))
        if gold_data:
            gold_steps, gold_ratings = zip(*gold_data)
            plt.plot(gold_steps, gold_ratings, '-', color=colors[i], label=f"{label} (Gold)", linewidth=2)

        # Read and process proxy data
        proxy_df = pd.read_csv(proxy_csv)
        proxy_df = proxy_df[proxy_df['player'].str.startswith(prefix)]
        proxy_steps = []
        proxy_ratings = []
        for _, row in proxy_df.iterrows():
            try:
                step = int(row['player'].split('_')[-1])
                proxy_steps.append(step)
                proxy_ratings.append(row['rating'])
            except:
                continue
        
        # Sort by steps
        proxy_data = sorted(zip(proxy_steps, proxy_ratings))
        if proxy_data:
            proxy_steps, proxy_ratings = zip(*proxy_data)
            plt.plot(proxy_steps, proxy_ratings, '--', color=colors[i], label=f"{label} (Proxy)", linewidth=2)

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Rating", fontsize=12)
    plt.title("Rating Progression Over Training Steps", fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def _combine_kl_by_prefix(df, mapping_by_prefix):
    combined = {}
    for (prefix, label), exp_list in mapping_by_prefix.items():
        step_to_vals = defaultdict(list)
        for (exp_name, step_offset) in exp_list:
            if exp_name not in df.columns:
                continue
            steps = df['Step'].values + step_offset
            kls = df[exp_name].values
            for s, k in zip(steps, kls):
                # Ignore NaNs/infs so we don't propagate them into the averages
                try:
                    kf = float(k)
                except Exception:
                    continue
                if np.isfinite(kf):
                    step_to_vals[int(s)].append(kf)
        if not step_to_vals:
            continue
        # Keep only steps that have at least one finite kl value
        valid_items = [(s, np.mean(vals)) for s, vals in step_to_vals.items() if len(vals) > 0]
        if not valid_items:
            continue
        valid_items.sort(key=lambda x: x[0])
        steps_sorted = np.array([s for s, _ in valid_items], dtype=int)
        kls_avg = np.array([v for _, v in valid_items], dtype=float)
        combined[prefix] = (label, steps_sorted, kls_avg)
    return combined

def _plot_steps_vs_kl_series(series_by_prefix, filename, smooth=False):
    prefixes = list(series_by_prefix.keys())
    colors = create_color_list(len(prefixes))
    plt.figure(figsize=(15,8))
    ax = plt.gca()
    csv_rows = []
    for i, prefix in enumerate(prefixes):
        label, steps, kls = series_by_prefix[prefix]
        # Drop NaNs/infs if any remain
        if kls.size == 0:
            continue
        mask = np.isfinite(kls)
        steps = steps[mask]
        y = kls[mask]
        if y.size == 0:
            continue
        if smooth:
            try:
                n = y.size
                if n >= 7:
                    window = min(51, n if n % 2 == 1 else n-1)
                    if window < 7:
                        window = 7
                    if window % 2 == 0:
                        window -= 1
                    polyorder = 3 if window > 5 else 2
                    polyorder = min(polyorder, window-1)
                    y = signal.savgol_filter(y, window_length=window, polyorder=polyorder)
            except Exception:
                y = kls
        plt.plot(steps, y, '-', color=colors[i], linewidth=2, label=label)
        for s, val in zip(steps, y):
            csv_rows.append({'step': int(s), 'kl': float(val), 'label': label, 'prefix': prefix})
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("KL Divergence", fontsize=12)
    plt.title("Steps vs KL Divergence (Smoothed)" if smooth else "Steps vs KL Divergence", fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"z_plots/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    pd.DataFrame(csv_rows).to_csv(f"z_plots/{filename}.csv", index=False)

def generate_steps_vs_kl_outputs(wandb_file, mapping_by_prefix):
    df = pd.read_csv(wandb_file)
    series_by_prefix = _combine_kl_by_prefix(df, mapping_by_prefix)
    # Raw combined (no smoothing)
    _plot_steps_vs_kl_series(series_by_prefix, filename="steps_vs_kl", smooth=False)
    # Smoothed combined
    _plot_steps_vs_kl_series(series_by_prefix, filename="steps_vs_kl_est", smooth=True)

def plot_kl_vs_rating(pairs, file_name, save_path, kl_csv):
    colors = create_color_list(len(pairs))
    plt.figure(figsize=(15,8))
    ax = plt.gca()

    # Read KL data from existing CSV
    kl_df = pd.read_csv(kl_csv)
    
    # Collect data for CSV
    csv_data = []
    
    # Plot each pair's KL vs Rating points and fit curves
    for i, (proxy_csv, prefix, label) in enumerate(pairs):
        # Get KL values for this prefix from the CSV
        prefix_kl = kl_df[kl_df['prefix'] == prefix]
        if len(prefix_kl) == 0:
            continue
            
        steps = prefix_kl['step'].values
        kl_values = prefix_kl['kl'].values
        
        # Collect points for gold data
        gold_kls = []
        gold_ratings = []
        
        # Read and process gold data
        gold_df = pd.read_csv(f"{save_path}/gold.csv")
        gold_df = gold_df[gold_df['player'].str.startswith(prefix)]
        for _, row in gold_df.iterrows():
            try:
                step = int(row['player'].split('_')[-1])
                kl = kl_values[step]
                rating = row['rating']
                plt.scatter(kl, rating, color=colors[i], marker='o', label=f"{label} (Gold)" if len(gold_kls) == 0 else "", alpha=0.6)
                gold_kls.append(kl)
                gold_ratings.append(rating)
                csv_data.append({
                    'kl': kl,
                    'rating': rating,
                    'source': 'gold',
                    'label': label,
                    'prefix': prefix
                })
            except:
                continue

        # Fit quadratic curve for gold data if we have enough points
        if len(gold_kls) > 3:
            gold_kls = np.array(gold_kls)
            gold_ratings = np.array(gold_ratings)
            coeffs = np.polyfit(gold_kls, gold_ratings, 2)
            x_fit = np.linspace(min(gold_kls), max(gold_kls), 100)
            y_fit = coeffs[0] * x_fit**2 + coeffs[1] * x_fit + coeffs[2]
            plt.plot(x_fit, y_fit, '-', color=colors[i], label=f"{label} (Gold Fit)" if len(gold_kls) > 0 else "", linewidth=2)

        # Collect points for proxy data
        proxy_kls = []
        proxy_ratings = []

        # Read and process proxy data
        proxy_df = pd.read_csv(proxy_csv)
        proxy_df = proxy_df[proxy_df['player'].str.startswith(prefix)]
        for _, row in proxy_df.iterrows():
            try:
                step = int(row['player'].split('_')[-1])
                kl = kl_values[step]
                rating = row['rating']
                plt.scatter(kl, rating, color=colors[i], marker='x', label=f"{label} (Proxy)" if len(proxy_kls) == 0 else "", alpha=0.6)
                proxy_kls.append(kl)
                proxy_ratings.append(rating)
                csv_data.append({
                    'kl': kl,
                    'rating': rating,
                    'source': 'proxy',
                    'label': label,
                    'prefix': prefix
                })
            except:
                continue
                
        # Fit quadratic curve for proxy data if we have enough points
        if len(proxy_kls) > 3:
            proxy_kls = np.array(proxy_kls)
            proxy_ratings = np.array(proxy_ratings)
            coeffs = np.polyfit(proxy_kls, proxy_ratings, 2)
            x_fit = np.linspace(min(proxy_kls), max(proxy_kls), 100)
            y_fit = coeffs[0] * x_fit**2 + coeffs[1] * x_fit + coeffs[2]
            plt.plot(x_fit, y_fit, '--', color=colors[i], label=f"{label} (Proxy Fit)" if len(proxy_kls) > 0 else "", linewidth=2)

    plt.xlabel("KL Divergence", fontsize=12)
    plt.ylabel("Rating", fontsize=12)
    plt.title("KL Divergence vs Rating", fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save CSV
    pd.DataFrame(csv_data).to_csv(f"{save_path}/{file_name}.csv", index=False)







if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python z_elo_calculation.py [1|2|3]")
        sys.exit(1)
    
    branch = sys.argv[1]

    ##############################################################################
    ##############################################################################
    ##############################################################################
    plot_name = "_grm_of_training" #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO  #TODO
    save_path = f"z_plots/{plot_name}" 
    os.makedirs(save_path, exist_ok=True)
    ##############################################################################
    ##############################################################################
    ##############################################################################
    
    if branch == "1":
        ##############################################################################
        ##############################################################################
        selected_indices = [29,31,32,33,34,35,36,37] #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO 
        ##############################################################################
        ##############################################################################

        all_dataset_mappings = [
            (0,  "RLAIF/dpo_answer_reddit_judge_full_1e-6_0.02_8B_4B",                          "ans_8b_4b",    0,   "Policy 8B, Judge 4B Answer"),
            (1,  "RLAIF/dpo_thinking_reddit_judge_full_1e-6_0.02_8B_4B",                        "think_8b_4b",  0,   "Policy 8B, Judge 4B Thinking"),
            (2,  "RLAIF/dpo_answer_reddit_judge_full_1e-6_0.02_4B_4B",                          "ans_4b_4b",    0,   "Policy 4B, Judge 4B Answer (full)"),
            (3,  "RLAIF/dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_4B",          "think_4b_4b",  0,   "Policy 4B, Judge 4B Thinking (pos-bias)"),
            (4,  "RLAIF/dpo_answer_reddit_judge_1e-6_0.02_1.7B_4B",                              "ans_1.7b_4b",  0,   "Policy 1.7B, Judge 4B Answer"),
            (5,  "RLAIF/dpo_answer_reddit_judge_extra_1e-6_0.02_1.7B_4B",                        "ans_1.7b_4b",  270, "Policy 1.7B, Judge 4B Answer (extra)"),
            (6,  "RLAIF/dpo_thinking_reddit_judge_position_bias_1e-6_0.02_1.7B_4B",              "think_1.7b_4b",0,   "Policy 1.7B, Judge 4B Thinking"),
            (7,  "RLAIF/dpo_thinking_reddit_judge_position_bias_extra_1e-6_0.02_1.7B_4B",        "think_1.7b_4b",270, "Policy 1.7B, Judge 4B Thinking (extra)"),
            (8,  "RLAIF/dpo_answer_reddit_judge_1e-6_0.02_0.6B_4B",                               "ans_0.6b_4b",  0,   "Policy 0.6B, Judge 4B Answer"),
            (9,  "RLAIF/dpo_thinking_reddit_judge_position_bias_1e-6_0.02_0.6B_4B",               "think_0.6b_4b",0,   "Policy 0.6B, Judge 4B Thinking"),
            (10, "RLAIF/dpo_answer_reddit_judge_1e-6_0.02_4B_0.6B",                               "ans_4b_0.6b",  0,   "Policy 4B, Judge 0.6B Answer"),
            (11, "RLAIF/dpo_answer_reddit_judge_extra_1e-6_0.02_4B_0.6B",                         "ans_4b_0.6b",  270, "Policy 4B, Judge 0.6B Answer (extra)"),
            (12, "RLAIF/dpo_thinking_reddit_judge_position_bias_1e-6_0.02_4B_0.6B",               "think_4b_0.6b",0,   "Policy 4B, Judge 0.6B Thinking"),
            (13, "RLAIF/dpo_thinking_reddit_judge_position_bias_extra_1e-6_0.02_4B_0.6B",         "think_4b_0.6b",270, "Policy 4B, Judge 0.6B Thinking (extra)"),
            (14, "RLAIF/dpo_answer_reddit_judge_full_1e-6_0.02_4B_1.7B",                          "ans_4b_1.7b",  0,   "Policy 4B, Judge 1.7B Answer"),
            (15, "RLAIF/dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_1.7B",          "think_4b_1.7b",0,   "Policy 4B, Judge 1.7B Thinking"),
            (16, "RLAIF/dpo_answer_reddit_judge_full_1e-6_0.02_4B_4B",                            "ans_4b_4b",    0,   "Policy 4B, Judge 4B Answer (full)"),
            (17, "RLAIF/dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_4B",            "think_4b_4b",  0,   "Policy 4B, Judge 4B Thinking (full)"),
            (18, "RLAIF/dpo_answer_reddit_judge_1e-6_0.02_4B_8B",                                 "ans_4b_8b",    0,   "Policy 4B, Judge 8B Answer"),
            (19, "RLAIF/dpo_thinking_reddit_judge_position_bias_1e-6_0.02_4B_8B",                 "think_4b_8b",  0,   "Policy 4B, Judge 8B Thinking"),
            (20, "RLAIF/dpo_answer_reddit_judge_1e-6_0.02_4B_14B",                                "ans_4b_14b",   0,   "Policy 4B, Judge 14B Answer"),
            (21, "RLAIF/dpo_thinking_reddit_judge_position_bias_1e-6_0.02_4B_14B",                "think_4b_14b", 0,   "Policy 4B, Judge 14B Thinking"),
            (22, "RLAIF/dpo_answer_reddit_judge_1e-6_0.1_4B_4B",                                  "ans_0.1",      0,   "Policy 4B, Judge 4B Answer, β=0.1"),
            (23, "RLAIF/dpo_answer_reddit_judge_1e-6_0.02_4B_4B",                                 "ans_0.02",     0,   "Policy 4B, Judge 4B Answer, β=0.02"),
            (24, "RLAIF/dpo_answer_reddit_judge_1e-6_0.01_4B_4B",                                 "ans_0.01",     0,   "Policy 4B, Judge 4B Answer, β=0.01"),
            (25, "RLAIF/dpo_answer_reddit_judge_1e-6_0.005_4B_4B",                                "ans_0.005",    0,   "Policy 4B, Judge 4B Answer, β=0.005"),
            (26, "RLAIF/dpo_answer_reddit_judge_1e-6_0.002_4B_4B",                                "ans_0.002",    0,   "Policy 4B, Judge 4B Answer, β=0.002"),
            (27, "RLAIF/dpo_answer_reddit_offtheshelf_1e-6_0.02_4B_4B",                           "ans_off_4b_4b",0,   "Policy 4B, Judge 4B Answer, Offtheshelf"),
            (28, "RLAIF/dpo_answer_reddit_offtheshelf_extra_1e-6_0.02_4B_4B",                     "ans_off_4b_4b",270, "Policy 4B, Judge 4B Answer, Offtheshelf (extra)"),
            (29, "RLAIF/dpo_thinking_reddit_offtheshelf_1e-6_0.02_4B_4B",                         "think_off_4b_4b",0, "Policy 4B, Judge 4B Thinking, Offtheshelf"),
            (30, "RLAIF/dpo_thinking_reddit_offtheshelf_extra_1e-6_0.02_4B_4B",                   "think_off_4b_4b",270,"Policy 4B, Judge 4B Thinking, Offtheshelf (extra)"),
            (31, "RLAIF/dpo_answer_reddit_judge_full_1e-6_0.02_4B_4B",                            "ans_4b_4b",    0,   "Policy 4B, Judge 4B Answer, Trained"),
            (32, "RLAIF/dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_4B",            "think_4b_4b",  0,   "Policy 4B, Judge 4B Thinking, Trained"),
            (33, "RLAIF/dpo_thinking_reddit_judge_last_minute_50_1e-6_0.02_4B_4B",                "think_50_4b_4b",  0,   "Policy 4B, Judge 4B Thinking, Trained 50 steps"),
            (34, "RLAIF/dpo_thinking_reddit_judge_last_minute_100_1e-6_0.02_4B_4B",               "think_100_4b_4b",  0,   "Policy 4B, Judge 4B Thinking, Trained 100 steps"),
            (35, "RLAIF/dpo_thinking_reddit_judge_last_minute_150_1e-6_0.02_4B_4B",               "think_150_4b_4b",  0,   "Policy 4B, Judge 4B Thinking, Trained 150 steps"),
            (36, "RLAIF/dpo_thinking_reddit_judge_last_minute_200_1e-6_0.02_4B_4B",               "think_200_4b_4b",  0,   "Policy 4B, Judge 4B Thinking, Trained 200 steps"),
            (37, "RLAIF/dpo_thinking_reddit_judge_last_minute_250_1e-6_0.02_4B_4B",               "think_250_4b_4b",  0,   "Policy 4B, Judge 4B Thinking, Trained 250 steps"),
        ]

        df_list = []
        for idx, hf_repo, prefix, step_shift, desc in all_dataset_mappings:
            if idx in selected_indices:
                ds = load_dataset(hf_repo, split="train")
                df_list.append(create_pivoted_dataset(ds, step_shift=step_shift, prefix=prefix))
        
        merged_df = df_list[0]
        for df in df_list[1:]:
            merged_df = merged_df.merge(df, on="question", how="outer", copy=False)

        dataset = Dataset.from_pandas(merged_df)
        dataset.push_to_hub(f"RLAIF/reddit_evaluation_{plot_name}", split="train")
        
        ##############################################################################

    elif branch == "2":
        ##############################################################################
        ##############################################################################
        ##############################################################################
        N = 15000 #TODO
        baseline = "ans_4b_4b_0" #TODO
        judges = ["4b_off_thinking", "4b_answer", "4b_thinking", "4b_thinking_step_50", "4b_thinking_step_100", "4b_thinking_step_150", "4b_thinking_step_200","4b_thinking_step_250", "gold"] #TODO
        label_indices = [24, 25, 26, 27, 28, 29, 30, 31] #TODO
        ##############################################################################
        ##############################################################################
        ##############################################################################

        models = {
            "0.6b_answer": "z_models/sft_answer_reddit_gold_grm_1e-5_0.6B",
            "1.7b_answer": "z_models/sft_answer_reddit_gold_grm_1e-5_1.7B",
            "4b_answer": "z_models/sft_answer_reddit_gold_grm_1e-5_4B",
            "8b_answer": "z_models/sft_answer_reddit_gold_grm_1e-5_8B",
            "14b_answer": "z_models/sft_answer_reddit_gold_grm_1e-5_14B",
            "0.6b_thinking": "z_models/grpo_reddit_judge_position_bias_16_64_8_3e-5_1e-6_0.6B/global_step_800",
            "1.7b_thinking": "z_models/grpo_reddit_judge_position_bias_16_64_8_3e-5_1e-6_1.7B/global_step_800",
            "4b_thinking": "z_models/grpo_reddit_judge_position_bias_16_64_8_3e-5_1e-6_4B",
            "8b_thinking": "z_models/grpo_reddit_judge_position_bias_16_64_8_3e-5_1e-6_8B/global_step_1100",
            "14b_thinking": "z_models/grpo_reddit_judge_position_bias_16_64_8_3e-5_1e-6_14B/global_step_800",
            "gold": "z_models/sft_answer_reddit_human_grm_1e-5_32B",
            "4b_off_answer": "Qwen/Qwen3-4B",
            
            "4b_off_thinking": "Qwen/Qwen3-4B",
            "4b_thinking_step_50": "z_models/grpo_reddit_judge_last_minute_16_64_8_3e-5_1e-6_4B/global_step_50",
            "4b_thinking_step_100": "z_models/grpo_reddit_judge_last_minute_16_64_8_3e-5_1e-6_4B/global_step_100",
            "4b_thinking_step_150": "z_models/grpo_reddit_judge_last_minute_16_64_8_3e-5_1e-6_4B/global_step_150",
            "4b_thinking_step_200": "z_models/grpo_reddit_judge_last_minute_16_64_8_3e-5_1e-6_4B/global_step_200",
            "4b_thinking_step_250": "z_models/grpo_reddit_judge_last_minute_16_64_8_3e-5_1e-6_4B/global_step_250",
        }

        all_label_mappings = [
            (0, "ans_8b_4b", "Policy 8B, Judge 4B Answer"),
            (1, "think_8b_4b", "Policy 8B, Judge 4B Thinking"),
            (2, "ans_4b_4b", "Policy 4B, Judge 4B Answer"),
            (3, "think_4b_4b", "Policy 4B, Judge 4B Thinking"),
            (4, "ans_1.7b_4b", "Policy 1.7B, Judge 4B Answer"),
            (5, "think_1.7b_4b", "Policy 1.7B, Judge 4B Thinking"),
            (6, "ans_0.6b_4b", "Policy 0.6B, Judge 4B Answer"),
            (7, "think_0.6b_4b", "Policy 0.6B, Judge 4B Thinking"),

            (8, "ans_4b_0.6b", "Policy 4B, Judge 0.6B Answer"),
            (9, "think_4b_0.6b", "Policy 4B, Judge 0.6B Thinking"),
            (10, "ans_4b_1.7b", "Policy 4B, Judge 1.7B Answer"),
            (11, "think_4b_1.7b", "Policy 4B, Judge 1.7B Thinking"),
            (12, "ans_4b_4b", "Policy 4B, Judge 4B Answer"),
            (13, "think_4b_4b", "Policy 4B, Judge 4B Thinking"),
            (14, "ans_4b_8b", "Policy 4B, Judge 8B Answer"),
            (15, "think_4b_8b", "Policy 4B, Judge 8B Thinking"),
            (16, "ans_4b_14b", "Policy 4B, Judge 14B Answer"),
            (17, "think_4b_14b", "Policy 4B, Judge 14B Thinking"),

            (18, "ans_0.002", "Policy 4B, Judge 4B Answer, ß=0.002"),
            (19, "ans_0.005", "Policy 4B, Judge 4B Answer, ß=0.005"),
            (20, "ans_0.01", "Policy 4B, Judge 4B Answer, ß=0.01"),
            (21, "ans_0.02", "Policy 4B, Judge 4B Answer, ß=0.02"),
            (22, "ans_0.1", "Policy 4B, Judge 4B Answer, ß=0.1"),

            (23, "ans_off_4b_4b", "Policy 4B, Judge 4B Answer, Offtheshelf"),

            (24, "think_off_4b_4b", "Policy 4B, Judge 4B Thinking, Offtheshelf"),
            (25, "ans_4b_4b", "Policy 4B, Judge 4B Answer, Trained"),
            (26, "think_4b_4b", "Policy 4B, Judge 4B Thinking, Trained"),

            (27, "think_50_4b_4b", "Policy 4B, Judge 4B Thinking, Trained 50 steps"),
            (28, "think_100_4b_4b", "Policy 4B, Judge 4B Thinking, Trained 100 steps"),
            (29, "think_150_4b_4b", "Policy 4B, Judge 4B Thinking, Trained 150 steps"),
            (30, "think_200_4b_4b", "Policy 4B, Judge 4B Thinking, Trained 200 steps"),
            (31, "think_250_4b_4b", "Policy 4B, Judge 4B Thinking, Trained 250 steps"),
        ]

        modes = {
            "0.6b_answer": "answer",
            "1.7b_answer": "answer",
            "4b_answer": "answer",
            "8b_answer": "answer",
            "14b_answer": "answer",
            "0.6b_thinking": "thinking",
            "1.7b_thinking": "thinking_2",
            "4b_thinking": "thinking",
            "8b_thinking": "thinking",
            "14b_thinking": "thinking",
            "gold": "answer",
            "4b_off_answer": "answer",
            "4b_off_thinking": "thinking",
            "4b_thinking_step_50": "thinking",
            "4b_thinking_step_100": "thinking",
            "4b_thinking_step_150": "thinking",
            "4b_thinking_step_200": "thinking",
            "4b_thinking_step_250": "thinking",
        }

        sampling_params = {
            "answer": SamplingParams(max_tokens=16, n=1, temperature=0.0),
            "thinking": SamplingParams(max_tokens=2000, n=1, temperature=0.0),
            "thinking_2": SamplingParams(max_tokens=2000, n=1, temperature=1.0)
        }
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        dataset = load_dataset(f"RLAIF/reddit_evaluation_{plot_name}", split="train")
        results_repo= f"RLAIF/reddit_results_{plot_name}"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        prompts_thinking, prompts_answer = sample_prompts(dataset, tokenizer, n=N, seed=42)
        label_mapping = {}
        for i in label_indices:
            _, name, label = all_label_mappings[i]
            label_mapping[name] = label
        color_mapping = create_color_map(label_mapping)
        ##############################################################################
        ##############################################################################
        ##############################################################################

        for name in judges:
            model, mode = models[name], modes[name]
            params = sampling_params[mode]
            prompts = prompts_thinking if mode == "thinking" or mode == "thinking_2" else prompts_answer
            try:
                ds = load_dataset(results_repo, split=name)
                results = [(row['winner'], row['loser']) for row in ds]
            except Exception as e:
                results = []
            outputs = multi_gpu_inference(prompts, model, params)
            judgments = extract_judgments(outputs)
            judged = attach_judgments(prompts, judgments)
            new_results = to_pairwise_results(judged)
            results.extend(new_results)
            push_results_to_hf(results, repo_id=results_repo, split=name)
            ratings = bt_to_shifted_elo(compute_bt_ratings(results, reg=0.01), baseline)
            plot_ratings(ratings, file_name=name, save_path=save_path, label_mapping=label_mapping, color_mapping=color_mapping)

    elif branch == "3":
        pairs = [
            #(f"{save_path}/0.6b_answer.csv",  "ans_4b_0.6b",        "GenRM-0.6B-Ans; Qwen3-4B"), #TODO
            #(f"{save_path}/0.6b_thinking.csv",  "think_4b_0.6b",        "GenRM-0.6B-Think; Qwen3-4B"), #TODO
            #(f"{save_path}/4b_answer.csv",   "ans_0.6b_4b",        "GenRM-4B-Ans; Qwen3-0.6B"), #TODO
            #(f"{save_path}/4b_thinking.csv",   "think_0.6b_4b",        "GenRM-4B-Think; Qwen3-0.6B"), #TODO
            #(f"{save_path}/1.7b_answer.csv",   "ans_4b_1.7b",        "GenRM-1.7B-Ans; Qwen3-4B"), #TODO    
            #(f"{save_path}/1.7b_thinking.csv",   "think_4b_1.7b",        "GenRM-1.7B-Think; Qwen3-4B"), #TODO   
            #(f"{save_path}/4b_answer.csv",   "ans_1.7b_4b",        "GenRM-4B-Ans; Qwen3-1.7B"), #TODO
            #(f"{save_path}/4b_thinking.csv",   "think_1.7b_4b",        "GenRM-4B-Think; Qwen3-1.7B"), #TODO
            (f"{save_path}/4b_answer.csv",   "ans_4b_4b",        "GenRM-4B-Ans; Qwen3-4B"), #TODO
            (f"{save_path}/4b_thinking.csv",   "think_4b_4b",        "GenRM-4B-Think; Qwen3-4B"), #TODO
            #(f"{save_path}/8b_answer.csv",  "ans_4b_8b",        "GenRM-8B-Ans; Qwen3-4B"), #TODO
            #(f"{save_path}/8b_thinking.csv",  "think_4b_8b",        "GenRM-8B-Think; Qwen3-4B"), #TODO
            #(f"{save_path}/4b_answer.csv",   "ans_8b_4b",        "GenRM-4B-Ans; Qwen3-8B"), #TODO
            #(f"{save_path}/4b_thinking.csv",   "think_8b_4b",        "GenRM-4B-Think; Qwen3-8B"), #TODO
            #(f"{save_path}/14b_answer.csv",   "ans_4b_14b",        "GenRM-14B-Ans; Qwen3-4B"), #TODO
            #(f"{save_path}/14b_thinking.csv",   "think_4b_14b",        "GenRM-14B-Think; Qwen3-4B"), #TODO
            (f"{save_path}/4b_thinking_step_50.csv",   "think_50_4b_4b",        "GenRM-4B-Think-50; Qwen3-4B"), #TODO
            (f"{save_path}/4b_thinking_step_100.csv",   "think_100_4b_4b",        "GenRM-4B-Think-100; Qwen3-4B"), #TODO
            (f"{save_path}/4b_thinking_step_150.csv",   "think_150_4b_4b",        "GenRM-4B-Think-150; Qwen3-4B"), #TODO
            (f"{save_path}/4b_thinking_step_200.csv",   "think_200_4b_4b",        "GenRM-4B-Think-200; Qwen3-4B"), #TODO
            (f"{save_path}/4b_thinking_step_250.csv",   "think_250_4b_4b",        "GenRM-4B-Think-250; Qwen3-4B"), #TODO
            (f"{save_path}/4b_off_thinking.csv",   "think_off_4b_4b",        "Baseline-4B-Think; Qwen3-4B"), #TODO
        ]
        plot_proxy_vs_gold(
            pairs, save_path=save_path, file_name="proxy_vs_gold", #TODO
            min_checkpoint=0, max_checkpoint=260, #TODO
            min_x_value=-100, max_x_value=3500 #TODO
        )   














        
    elif branch == "4":
        # Use same pairs as branch 3 but replace all csv files with gold.csv
        pairs = [
            # 4B judge pairs
            (f"{save_path}/gold.csv",  "ans_4b_0.6b",        "GenRM-0.6B-Ans; Qwen3-4B", (1, False)),
            (f"{save_path}/gold.csv",  "think_4b_0.6b",        "GenRM-0.6B-Think; Qwen3-4B", (1, True)),
            (f"{save_path}/gold.csv",   "ans_4b_1.7b",        "GenRM-1.7B-Ans; Qwen3-4B", (2, False)),
            (f"{save_path}/gold.csv",   "think_4b_1.7b",        "GenRM-1.7B-Think; Qwen3-4B", (2, True)),
            (f"{save_path}/gold.csv",   "ans_4b_4b",        "GenRM-4B-Ans; Qwen3-4B", (3, False)),
            (f"{save_path}/gold.csv",   "think_4b_4b",        "GenRM-4B-Think; Qwen3-4B", (3, True)),
            (f"{save_path}/gold.csv",  "ans_4b_8b",        "GenRM-8B-Ans; Qwen3-4B", (4, False)),
            (f"{save_path}/gold.csv",  "think_4b_8b",        "GenRM-8B-Think; Qwen3-4B", (4, True)),
            (f"{save_path}/gold.csv",   "ans_4b_14b",        "GenRM-14B-Ans; Qwen3-4B", (5, False)),
            (f"{save_path}/gold.csv",   "think_4b_14b",        "GenRM-14B-Think; Qwen3-4B", (5, True)),
            
            # Other judge sizes
            (f"{save_path}/gold.csv",   "ans_0.6b_4b",        "GenRM-4B-Ans; Qwen3-0.6B", (6, False)),
            (f"{save_path}/gold.csv",   "think_0.6b_4b",        "GenRM-4B-Think; Qwen3-0.6B", (6, True)),
            (f"{save_path}/gold.csv",   "ans_1.7b_4b",        "GenRM-4B-Ans; Qwen3-1.7B", (7, False)),
            (f"{save_path}/gold.csv",   "think_1.7b_4b",        "GenRM-4B-Think; Qwen3-1.7B", (7, True)),
            (f"{save_path}/gold.csv",   "ans_8b_4b",        "GenRM-4B-Ans; Qwen3-8B", (8, False)),
            (f"{save_path}/gold.csv",   "think_8b_4b",        "GenRM-4B-Think; Qwen3-8B", (8, True)),
        ]
        plot_max_gold(pairs, file_name="top_5_improved", save_path=save_path, top_n=5)

    elif branch == "5":
        # Use pairs from branch 3 to plot rating progression over steps
        pairs = [
            #(f"{save_path}/0.6b_answer.csv",  "ans_4b_0.6b",        "GenRM-0.6B-Ans; Qwen3-4B"),
            #(f"{save_path}/0.6b_thinking.csv",  "think_4b_0.6b",        "GenRM-0.6B-Think; Qwen3-4B"),
            #(f"{save_path}/4b_answer.csv",   "ans_0.6b_4b",        "GenRM-4B-Ans; Qwen3-0.6B"),
            (f"{save_path}/4b_thinking.csv",   "think_0.6b_4b",        "GenRM-4B-Think; Qwen3-0.6B"),
            #(f"{save_path}/1.7b_answer.csv",   "ans_4b_1.7b",        "GenRM-1.7B-Ans; Qwen3-4B"),
            #(f"{save_path}/1.7b_thinking.csv",   "think_4b_1.7b",        "GenRM-1.7B-Think; Qwen3-4B"),
            #(f"{save_path}/4b_answer.csv",   "ans_1.7b_4b",        "GenRM-4B-Ans; Qwen3-1.7B"),
            (f"{save_path}/4b_thinking.csv",   "think_1.7b_4b",        "GenRM-4B-Think; Qwen3-1.7B"),
            #(f"{save_path}/4b_answer.csv",   "ans_4b_4b",        "GenRM-4B-Ans; Qwen3-4B"),
            (f"{save_path}/4b_thinking.csv",   "think_4b_4b",        "GenRM-4B-Think; Qwen3-4B"),
            #(f"{save_path}/8b_answer.csv",  "ans_4b_8b",        "GenRM-8B-Ans; Qwen3-4B"),
            #(f"{save_path}/8b_thinking.csv",  "think_4b_8b",        "GenRM-8B-Think; Qwen3-4B"),
            #(f"{save_path}/4b_answer.csv",   "ans_8b_4b",        "GenRM-4B-Ans; Qwen3-8B"),
            (f"{save_path}/4b_thinking.csv",   "think_8b_4b",        "GenRM-4B-Think; Qwen3-8B"),
            #(f"{save_path}/14b_answer.csv",   "ans_4b_14b",        "GenRM-14B-Ans; Qwen3-4B"),
            #(f"{save_path}/14b_thinking.csv",   "think_4b_14b",        "GenRM-14B-Think; Qwen3-4B"),
        ]
        plot_steps_vs_rating(pairs, file_name="steps_policy_sizes_think", save_path=save_path)
    elif branch == "6":
        wandb_file = "wandb_export_2025-09-24T21_11_03.458-07_00.csv"
        # Original flat mapping (kept for reference of experiments list)
        all_experimnts_mappings = [
            (0,  "dpo_answer_reddit_judge_full_1e-6_0.02_8B_4B - kl_divergence",                            "ans_8b_4b",    0,   "Policy 8B, Judge 4B Answer"),
            (1,  "dpo_thinking_reddit_judge_full_1e-6_0.02_8B_4B - kl_divergence",                          "think_8b_4b",  0,   "Policy 8B, Judge 4B Thinking"),
            (2,  "dpo_answer_reddit_judge_full_1e-6_0.02_4B_4B - kl_divergence",                            "ans_4b_4b",    0,   "Policy 4B, Judge 4B Answer"),
            (3,  "dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_4B - kl_divergence",            "think_4b_4b",  0,   "Policy 4B, Judge 4B Thinking"),
            (4,  "dpo_answer_reddit_judge_1e-6_0.02_1.7B_4B - kl_divergence",                               "ans_1.7b_4b",  0,   "Policy 1.7B, Judge 4B Answer"),
            (5,  "dpo_answer_reddit_judge_extra_1e-6_0.02_1.7B_4B - kl_divergence",                         "ans_1.7b_4b",  260, "Policy 1.7B, Judge 4B Answer"),
            (6,  "dpo_thinking_reddit_judge_position_bias_1e-6_0.02_1.7B_4B - kl_divergence",               "think_1.7b_4b",0,   "Policy 1.7B, Judge 4B Thinking"),
            (7,  "dpo_thinking_reddit_judge_position_bias_extra_1e-6_0.02_1.7B_4B - kl_divergence",         "think_1.7b_4b",260, "Policy 1.7B, Judge 4B Thinking"),
            (8,  "dpo_answer_reddit_judge_1e-6_0.02_0.6B_4B - kl_divergence",                               "ans_0.6b_4b",  0,   "Policy 0.6B, Judge 4B Answer"),
            (9,  "dpo_thinking_reddit_judge_position_bias_1e-6_0.02_0.6B_4B - kl_divergence",               "think_0.6b_4b",0,   "Policy 0.6B, Judge 4B Thinking"),
            (10, "dpo_answer_reddit_judge_1e-6_0.02_4B_0.6B - kl_divergence",                               "ans_4b_0.6b",  0,   "Policy 4B, Judge 0.6B Answer"),
            (11, "dpo_answer_reddit_judge_extra_1e-6_0.02_4B_0.6B - kl_divergence",                         "ans_4b_0.6b",  260, "Policy 4B, Judge 0.6B Answer"),
            (12, "dpo_thinking_reddit_judge_position_bias_1e-6_0.02_4B_0.6B - kl_divergence",               "think_4b_0.6b",0,   "Policy 4B, Judge 0.6B Thinking"),
            (13, "dpo_thinking_reddit_judge_position_bias_extra_1e-6_0.02_4B_0.6B - kl_divergence",         "think_4b_0.6b",260, "Policy 4B, Judge 0.6B Thinking"),
            (14, "dpo_answer_reddit_judge_full_1e-6_0.02_4B_1.7B - kl_divergence",                          "ans_4b_1.7b",  0,   "Policy 4B, Judge 1.7B Answer"),
            (15, "dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_1.7B - kl_divergence",          "think_4b_1.7b",0,   "Policy 4B, Judge 1.7B Thinking"),
            (16, "dpo_answer_reddit_judge_full_1e-6_0.02_4B_4B - kl_divergence",                            "ans_4b_4b",    0,   "Policy 4B, Judge 4B Answer"),
            (17, "dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_4B - kl_divergence",            "think_4b_4b",  0,   "Policy 4B, Judge 4B Thinking"),
            (18, "dpo_answer_reddit_judge_1e-6_0.02_4B_8B - kl_divergence",                                 "ans_4b_8b",    0,   "Policy 4B, Judge 8B Answer"),
            (19, "dpo_thinking_reddit_judge_position_bias_1e-6_0.02_4B_8B - kl_divergence",                 "think_4b_8b",  0,   "Policy 4B, Judge 8B Thinking"),
            (20, "dpo_answer_reddit_judge_1e-6_0.02_4B_14B - kl_divergence",                                "ans_4b_14b",   0,   "Policy 4B, Judge 14B Answer"),
            (21, "dpo_thinking_reddit_judge_position_bias_1e-6_0.02_4B_14B - kl_divergence",                "think_4b_14b", 0,   "Policy 4B, Judge 14B Thinking"),
            (22, "dpo_answer_reddit_judge_1e-6_0.1_4B_4B - kl_divergence",                                  "ans_0.1",      0,   "Policy 4B, Judge 4B Answer, β=0.1"),
            (23, "dpo_answer_reddit_judge_1e-6_0.02_4B_4B - kl_divergence",                                 "ans_0.02",     0,   "Policy 4B, Judge 4B Answer, β=0.02"),
            (24, "dpo_answer_reddit_judge_1e-6_0.01_4B_4B - kl_divergence",                                 "ans_0.01",     0,   "Policy 4B, Judge 4B Answer, β=0.01"),
            (25, "dpo_answer_reddit_judge_1e-6_0.005_4B_4B - kl_divergence",                                "ans_0.005",    0,   "Policy 4B, Judge 4B Answer, β=0.005"),
            (26, "dpo_answer_reddit_judge_1e-6_0.002_4B_4B - kl_divergence",                                "ans_0.002",    0,   "Policy 4B, Judge 4B Answer, β=0.002"),
            (27, "dpo_answer_reddit_offtheshelf_1e-6_0.02_4B_4B - kl_divergence",                           "ans_off_4b_4b",0,   "Policy 4B, Judge 4B Answer, Offtheshelf"),
            (28, "dpo_answer_reddit_offtheshelf_extra_1e-6_0.02_4B_4B - kl_divergence",                     "ans_off_4b_4b",260, "Policy 4B, Judge 4B Answer, Offtheshelf"),
            (29, "dpo_thinking_reddit_offtheshelf_1e-6_0.02_4B_4B - kl_divergence",                         "think_off_4b_4b",0, "Policy 4B, Judge 4B Thinking, Offtheshelf"),
            (30, "dpo_thinking_reddit_offtheshelf_extra_1e-6_0.02_4B_4B - kl_divergence",                   "think_off_4b_4b",260,"Policy 4B, Judge 4B Thinking, Offtheshelf"),
            (31, "dpo_answer_reddit_judge_full_1e-6_0.02_4B_4B - kl_divergence",                            "ans_4b_4b",    0,   "Policy 4B, Judge 4B Answer"),
            (32, "dpo_thinking_reddit_judge_position_bias_full_1e-6_0.02_4B_4B - kl_divergence",            "think_4b_4b",  0,   "Policy 4B, Judge 4B Thinking"),
        ]

        # Inverted mapping: keys are (prefix, label), values are list of (exp_name, step_offset)
        mapping_by_prefix = defaultdict(list)
        for _, exp_name, prefix, step_offset, label in all_experimnts_mappings:
            mapping_by_prefix[(prefix, label)].append((exp_name, step_offset))
        
        # Generate both RAW and SMOOTHED per-prefix combined outputs (no params required)
        generate_steps_vs_kl_outputs(wandb_file, mapping_by_prefix)
    
    elif branch == "7":
        pairs = [
            #(f"{save_path}/0.6b_answer.csv",  "ans_4b_0.6b",        "GenRM-0.6B-Ans; Qwen3-4B"),
            #(f"{save_path}/0.6b_thinking.csv",  "think_4b_0.6b",        "GenRM-0.6B-Think; Qwen3-4B"),
            #(f"{save_path}/4b_answer.csv",   "ans_0.6b_4b",        "GenRM-4B-Ans; Qwen3-0.6B"),
            #(f"{save_path}/4b_thinking.csv",   "think_0.6b_4b",        "GenRM-4B-Think; Qwen3-0.6B"),
            #(f"{save_path}/1.7b_answer.csv",   "ans_4b_1.7b",        "GenRM-1.7B-Ans; Qwen3-4B"),
            #(f"{save_path}/1.7b_thinking.csv",   "think_4b_1.7b",        "GenRM-1.7B-Think; Qwen3-4B"),
            #(f"{save_path}/4b_answer.csv",   "ans_1.7b_4b",        "GenRM-4B-Ans; Qwen3-1.7B"),
            #(f"{save_path}/4b_thinking.csv",   "think_1.7b_4b",        "GenRM-4B-Think; Qwen3-1.7B"),
            #(f"{save_path}/4b_answer.csv",   "ans_4b_4b",        "GenRM-4B-Ans; Qwen3-4B"),
            #(f"{save_path}/4b_thinking.csv",   "think_4b_4b",        "GenRM-4B-Think; Qwen3-4B"),
            #(f"{save_path}/8b_answer.csv",  "ans_4b_8b",        "GenRM-8B-Ans; Qwen3-4B"),
            #(f"{save_path}/8b_thinking.csv",  "think_4b_8b",        "GenRM-8B-Think; Qwen3-4B"),
            #(f"{save_path}/4b_answer.csv",   "ans_8b_4b",        "GenRM-4B-Ans; Qwen3-8B"),
            #(f"{save_path}/4b_thinking.csv",   "think_8b_4b",        "GenRM-4B-Think; Qwen3-8B"),
            #(f"{save_path}/14b_answer.csv",   "ans_4b_14b",        "GenRM-14B-Ans; Qwen3-4B"),
            #(f"{save_path}/14b_thinking.csv",   "think_4b_14b",        "GenRM-14B-Think; Qwen3-4B"),
            (f"{save_path}/4b_answer.csv",   "ans_0.1",        "ß=0.1; GenRM-4B-Ans; Qwen3-4B"),
            (f"{save_path}/4b_answer.csv",   "ans_0.02",        "ß=0.02; GenRM-4B-Ans; Qwen3-4B"),
            (f"{save_path}/4b_answer.csv",   "ans_0.01",        "ß=0.01; GenRM-4B-Ans; Qwen3-4B"),
            (f"{save_path}/4b_answer.csv",   "ans_0.005",        "ß=0.005; GenRM-4B-Ans; Qwen3-4B"),
            (f"{save_path}/4b_answer.csv",   "ans_0.002",        "ß=0.002; GenRM-4B-Ans; Qwen3-4B"),
        ]

        #kl_csv = f"{save_path}/steps_vs_kl.csv"
        kl_csv_est = f"z_plots/steps_vs_kl_est.csv"
        #plot_kl_vs_rating(pairs, file_name="kl_vs_rating", save_path=save_path, kl_csv=kl_csv)
        plot_kl_vs_rating(pairs, file_name="kl_betas", save_path=save_path, kl_csv=kl_csv_est)
    
    #elif branch == "8":

    
