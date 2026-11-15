"""
Graph plotting utilities for xG analysis.
Add these functions to dashboard/app.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def plot_xg_timeline(home_shots, away_shots, home_team, away_team):
    """
    Plot cumulative xG over match time (simulated 15-min windows).
    Shows momentum and buildup.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0C1410")
    ax.set_facecolor("#0C1410")

    # Sort shots by x position (rough proxy for match time)
    h_sorted = sorted(home_shots, key=lambda s: s["x"])
    a_sorted = sorted(away_shots, key=lambda s: s["x"])

    # Cumulative xG
    h_cum = np.cumsum([s["xg"] for s in h_sorted])
    a_cum = np.cumsum([s["xg"] for s in a_sorted])

    h_idx = np.arange(len(h_cum))
    a_idx = np.arange(len(a_cum))

    ax.plot(h_idx, h_cum, marker="o", linewidth=2.5, markersize=6,
            color="#C9A84C", label=home_team, alpha=0.85)
    ax.plot(a_idx, a_cum, marker="s", linewidth=2.5, markersize=6,
            color="#1D9E75", label=away_team, alpha=0.85)

    ax.set_xlabel("Shot number (match progression)", color="#3A5A3E", fontsize=10)
    ax.set_ylabel("Cumulative xG", color="#3A5A3E", fontsize=10)
    ax.set_title(f"{home_team} vs {away_team} — xG Accumulation", 
                 color="#C9A84C", fontsize=12, fontweight="bold", pad=16)

    ax.tick_params(colors="#4A6A50", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1A2A1E")
    ax.spines["bottom"].set_color("#1A2A1E")

    ax.grid(True, alpha=0.1, linestyle="--", color="#1A2A1E")
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor="#6A8A6E")

    plt.tight_layout()
    return fig


def plot_shot_quality_scatter(home_shots, away_shots, home_team, away_team):
    """
    Scatter plot: distance vs angle, sized by xG, colored by team.
    Shows shot quality clustering.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0C1410")
    ax.set_facecolor("#0C1410")

    h_dist = [s["x"] for s in home_shots]
    h_y    = [s["y"] for s in home_shots]
    h_xg   = [s["xg"] for s in home_shots]

    a_dist = [s["x"] for s in away_shots]
    a_y    = [s["y"] for s in away_shots]
    a_xg   = [s["xg"] for s in away_shots]

    # Distance proxy: distance from goal (120 - x)
    h_dist_from_goal = [120 - d for d in h_dist]
    a_dist_from_goal = [120 - d for d in a_dist]

    # Angle proxy: lateral distance from centre
    h_centrality = [abs(y - 40) for y in h_y]
    a_centrality = [abs(y - 40) for y in a_y]

    ax.scatter(h_dist_from_goal, h_centrality,
               s=[xg * 400 for xg in h_xg],
               c="#C9A84C", alpha=0.6, edgecolors="#C9A84C",
               linewidth=1, label=home_team)

    ax.scatter(a_dist_from_goal, a_centrality,
               s=[xg * 400 for xg in a_xg],
               c="#1D9E75", alpha=0.6, edgecolors="#1D9E75",
               linewidth=1, label=away_team)

    ax.set_xlabel("Distance from goal (yards)", color="#3A5A3E", fontsize=10)
    ax.set_ylabel("Lateral distance from centre (yards)", color="#3A5A3E", fontsize=10)
    ax.set_title(f"{home_team} vs {away_team} — Shot Quality Map\n(bubble size = xG value)",
                 color="#C9A84C", fontsize=12, fontweight="bold", pad=16)

    ax.tick_params(colors="#4A6A50", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1A2A1E")
    ax.spines["bottom"].set_color("#1A2A1E")

    ax.grid(True, alpha=0.1, linestyle="--", color="#1A2A1E")
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor="#6A8A6E")

    plt.tight_layout()
    return fig


def plot_team_radar(home_team, away_team, home_xg, away_xg, home_shots, away_shots):
    """
    Radar chart comparing team playing styles.
    5 metrics: Attack intensity, Pressing (shot count), 
    Shot efficiency, Centrality, Danger conversion.
    """
    from math import pi

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#0C1410")
    ax.set_facecolor("#0C1410")

    categories = [
        "Attack\nIntensity",
        "Shot\nEfficiency",
        "Pressing\n(Volume)",
        "Chance\nQuality",
        "Dangerous\nZone %"
    ]

    # Calculate metrics
    h_attack = min(len(home_shots) / 12 * 100, 100)  # normalize to 12 shots = 100
    a_attack = min(len(away_shots) / 12 * 100, 100)

    h_efficiency = (home_xg / max(len(home_shots), 1)) * 100  # xG per shot normalized
    a_efficiency = (away_xg / max(len(away_shots), 1)) * 100

    h_pressing = len(home_shots) * 8  # raw shot count
    a_pressing = len(away_shots) * 8

    h_quality = home_xg * 20  # xG normalized
    a_quality = away_xg * 20

    h_dangerous = sum(1 for s in home_shots if s["xg"] > 0.15) / max(len(home_shots), 1) * 100
    a_dangerous = sum(1 for s in away_shots if s["xg"] > 0.15) / max(len(away_shots), 1) * 100

    h_values = [h_attack, h_efficiency, h_pressing, h_quality, h_dangerous]
    a_values = [a_attack, a_efficiency, a_pressing, a_quality, a_dangerous]

    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    h_values += h_values[:1]
    a_values += a_values[:1]
    angles += angles[:1]

    ax.plot(angles, h_values, "o-", linewidth=2.5, color="#C9A84C",
            label=home_team, markersize=7)
    ax.fill(angles, h_values, alpha=0.15, color="#C9A84C")

    ax.plot(angles, a_values, "s-", linewidth=2.5, color="#1D9E75",
            label=away_team, markersize=7)
    ax.fill(angles, a_values, alpha=0.15, color="#1D9E75")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#3A5A3E", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#2A4A2E", fontsize=8)
    ax.grid(True, color="#1A2A1E", linestyle="--", alpha=0.5)

    ax.set_title(f"{home_team} vs {away_team} — Playing Style Comparison",
                 color="#C9A84C", fontsize=12, fontweight="bold", pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              frameon=False, fontsize=10, labelcolor="#6A8A6E")

    plt.tight_layout()
    return fig


def plot_shot_type_breakdown(home_shots, away_shots, home_team, away_team):
    """
    Bar chart: breakdown of shots by type (goal, saved, blocked, off target).
    Shows finishing quality.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0C1410")
    ax.set_facecolor("#0C1410")

    h_goals   = sum(1 for s in home_shots if s["goal"])
    h_miss    = len(home_shots) - h_goals
    a_goals   = sum(1 for s in away_shots if s["goal"])
    a_miss    = len(away_shots) - a_goals

    teams = [home_team, away_team]
    goals = [h_goals, a_goals]
    misses = [h_miss, a_miss]

    x = np.arange(len(teams))
    width = 0.5

    bars1 = ax.bar(x - width/2, goals, width, label="Goals",
                   color="#C9A84C", edgecolor="#C9A84C", linewidth=1.5)
    bars2 = ax.bar(x + width/2, misses, width, label="Shots (no goal)",
                   color="#2A4A2E", edgecolor="#3A5A3E", linewidth=1)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{int(height)}", ha="center", va="bottom",
                    color="#C9A84C", fontweight="bold", fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{int(height)}", ha="center", va="bottom",
                    color="#3A5A3E", fontweight="bold", fontsize=10)

    ax.set_ylabel("Count", color="#3A5A3E", fontsize=10)
    ax.set_title(f"{home_team} vs {away_team} — Shot Breakdown (Simulated)",
                 color="#C9A84C", fontsize=12, fontweight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(teams, color="#F0F0F0", fontsize=11, fontweight="bold")

    ax.tick_params(colors="#4A6A50", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1A2A1E")
    ax.spines["bottom"].set_color("#1A2A1E")

    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor="#6A8A6E")
    ax.set_ylim(0, max(len(home_shots), len(away_shots)) * 1.15)

    plt.tight_layout()
    return fig
