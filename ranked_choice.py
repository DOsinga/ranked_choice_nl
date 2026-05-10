#!/usr/bin/env python3
"""
Simulate Dutch Tweede Kamer elections under:
  1. First Past the Post (FPTP)
  2. Ranked Choice Voting (Instant Runoff)

Uses actual per-gemeente (municipality) results. For ranked choice, voters are
placed on a left-right spectrum calibrated to match actual vote shares, then
rank parties by proximity.
"""

import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

# ── Party positions on 0-10 left-right scale (Chapel Hill Expert Survey 2024) ──
# Only include parties with meaningful vote share
PARTIES = {
    "SP (Socialistische Partij)":                   1.17,
    "Partij voor de Dieren":                        1.50,
    "BIJ1":                                         1.00,
    "GROENLINKS / Partij van de Arbeid (PvdA)":     2.75,
    "DENK":                                         3.89,
    "Volt":                                         4.55,
    "ChristenUnie":                                 4.58,
    "D66":                                          4.67,
    "50PLUS":                                       5.00,
    "CDA":                                          5.75,
    "Nieuw Sociaal Contract":                       6.58,
    "Nieuw Sociaal Contract (NSC)":                 6.58,
    "BBB":                                          7.75,
    "VVD":                                          7.67,
    "Staatkundig Gereformeerde Partij (SGP)":       8.00,
    "JA21":                                         8.25,
    "PVV (Partij voor de Vrijheid)":                9.08,
    "Forum voor Democratie":                        9.83,
}

SHORT_NAMES = {
    "SP (Socialistische Partij)":                   "SP",
    "Partij voor de Dieren":                        "PvdD",
    "BIJ1":                                         "BIJ1",
    "GROENLINKS / Partij van de Arbeid (PvdA)":     "GL-PvdA",
    "DENK":                                         "DENK",
    "Volt":                                         "Volt",
    "ChristenUnie":                                 "CU",
    "D66":                                          "D66",
    "50PLUS":                                       "50+",
    "CDA":                                          "CDA",
    "Nieuw Sociaal Contract":                       "NSC",
    "Nieuw Sociaal Contract (NSC)":                 "NSC",
    "BBB":                                          "BBB",
    "VVD":                                          "VVD",
    "Staatkundig Gereformeerde Partij (SGP)":       "SGP",
    "JA21":                                         "JA21",
    "PVV (Partij voor de Vrijheid)":                "PVV",
    "Forum voor Democratie":                        "FvD",
}

PARTY_COLORS = {
    "D66":     "#4eac50",  # bright green
    "PVV":     "#152e62",  # dark navy
    "VVD":     "#142bc2",  # royal blue
    "GL-PvdA": "#ca3631",  # red (PvdA red)
    "CDA":     "#47915e",  # medium green
    "JA21":    "#29356f",  # dark indigo
    "FvD":     "#79231e",  # dark red/maroon
    "BBB":     "#9dc043",  # lime green
    "DENK":    "#52b4b1",  # teal
    "SGP":     "#d9652c",  # orange
    "PvdD":    "#2d6a34",  # forest green
    "CU":      "#4aa5e5",  # sky blue
    "SP":      "#e23122",  # bright red
    "50+":     "#862079",  # purple
    "Volt":    "#4a2675",  # dark purple
    "BIJ1":    "#D40075",  # pink
    "NSC":     "#1E3A5F",  # dark blue-grey
}


def load_gemeente_data(csv_path: str) -> dict[str, dict[str, int]]:
    """Load vote counts per party per gemeente from the CSV."""
    results = defaultdict(dict)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            regio, code = row[0], row[1]
            if not code.startswith("G") and not code.startswith("O"):
                continue
            veldtype = row[14]
            if veldtype != "LijstAantalStemmen":
                continue
            party = row[6]
            votes = int(row[15])
            if party in PARTIES:
                results[regio][party] = votes
    return dict(results)


def fptp(district_data: dict) -> dict[str, str]:
    """First past the post: winner of each district. Returns {district: winner}."""
    winners = {}
    for district, votes in district_data.items():
        winners[district] = max(votes, key=votes.get)
    return winners


def generate_ballots(vote_shares: dict[str, float], n_voters: int = 1000,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate ranked ballots where:
    - First choice matches actual vote shares exactly
    - Remaining preferences are ordered by proximity on the L-R spectrum
      (with some noise so similarly-positioned parties aren't deterministic)

    Returns rankings array (n_voters, n_parties) of party indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    party_names = list(vote_shares.keys())
    n_parties = len(party_names)
    shares = np.array([vote_shares[p] for p in party_names])
    positions = np.array([PARTIES[p] for p in party_names])

    # Assign each voter a first-choice party matching real vote shares
    first_choice = rng.choice(n_parties, size=n_voters, p=shares)

    # Place each voter near their chosen party on the spectrum
    sigma = 0.5
    voter_positions = positions[first_choice] + rng.normal(0, sigma, n_voters)

    # Build rankings: for each voter, rank parties by distance from voter position
    distances = np.abs(voter_positions[:, None] - positions[None, :])
    rankings = np.argsort(distances, axis=1)

    # Override: ensure actual first choice is always rank 0
    # Move the assigned first-choice party to the front of each ballot
    for i in range(n_voters):
        fc = first_choice[i]
        rank_of_fc = np.where(rankings[i] == fc)[0][0]
        if rank_of_fc != 0:
            # Shift everything before it right by one, put fc at front
            rankings[i, 1:rank_of_fc+1] = rankings[i, 0:rank_of_fc]
            rankings[i, 0] = fc

    return rankings, party_names


def instant_runoff_np(rankings: np.ndarray, party_names: list[str],
                      party_positions: np.ndarray) -> str:
    """
    Numpy-accelerated instant runoff voting.
    rankings: (n_voters, n_parties) array of party indices, ranked by preference
    party_names: list of party name strings
    party_positions: (n_parties,) array of positions (for tie-breaking only)
    """
    n_voters, n_parties = rankings.shape
    eliminated = np.zeros(n_parties, dtype=bool)

    for _ in range(n_parties - 1):
        # For each voter, find their top non-eliminated choice
        valid = ~eliminated[rankings]  # (n_voters, n_parties) bool
        first_choice_rank = np.argmax(valid, axis=1)  # (n_voters,)
        top_choice = rankings[np.arange(n_voters), first_choice_rank]

        # Count votes per party
        counts = np.bincount(top_choice, minlength=n_parties)
        counts[eliminated] = 0

        # Check majority
        active_total = counts.sum()
        if active_total == 0:
            break
        max_idx = np.argmax(counts)
        if counts[max_idx] > active_total / 2:
            return party_names[max_idx]

        # Eliminate party with fewest votes (among active)
        active_counts = counts.copy()
        active_counts[eliminated] = active_total + 1  # exclude eliminated
        min_votes = active_counts.min()
        # Tie-break: eliminate furthest from center
        tied = (active_counts == min_votes) & ~eliminated
        tied_indices = np.where(tied)[0]
        elim_idx = tied_indices[np.argmax(np.abs(party_positions[tied_indices] - 5.0))]
        eliminated[elim_idx] = True

        if eliminated.sum() == n_parties - 1:
            return party_names[np.where(~eliminated)[0][0]]

    return party_names[np.argmax(~eliminated)]


def runoff(vote_shares: dict[str, float]) -> str:
    """
    Two-round runoff: top two parties advance, then all other voters pick
    whichever of the two is closer on the L-R spectrum.
    """
    # Find top two by vote share
    sorted_parties = sorted(vote_shares, key=vote_shares.get, reverse=True)
    p1, p2 = sorted_parties[0], sorted_parties[1]

    # Midpoint between the two finalists on the spectrum
    mid = (PARTIES[p1] + PARTIES[p2]) / 2

    # Each remaining party's voters go to whichever finalist is closer
    votes1 = vote_shares[p1]
    votes2 = vote_shares[p2]
    for p in sorted_parties[2:]:
        if PARTIES[p] <= mid:
            # Closer to the more left-wing finalist
            if PARTIES[p1] < PARTIES[p2]:
                votes1 += vote_shares[p]
            else:
                votes2 += vote_shares[p]
        else:
            if PARTIES[p1] > PARTIES[p2]:
                votes1 += vote_shares[p]
            else:
                votes2 += vote_shares[p]

    return p1 if votes1 >= votes2 else p2


def run_runoff(district_data: dict) -> dict[str, str]:
    """Two-round runoff for each district."""
    winners = {}
    for district, dvotes in district_data.items():
        total = sum(dvotes.values())
        shares = {p: v / total for p, v in dvotes.items()}
        winners[district] = runoff(shares)
    return winners


def median_voter(vote_shares: dict[str, float]) -> str:
    """Find the party of the median voter on the L-R spectrum."""
    parties = sorted(vote_shares.keys(), key=lambda p: PARTIES[p])
    cumulative = 0.0
    for p in parties:
        cumulative += vote_shares[p]
        if cumulative >= 0.5:
            return p
    return parties[-1]


def run_median_voter(district_data: dict) -> dict[str, str]:
    """Median voter winner for each district."""
    winners = {}
    for district, dvotes in district_data.items():
        total = sum(dvotes.values())
        shares = {p: v / total for p, v in dvotes.items()}
        winners[district] = median_voter(shares)
    return winners


def run_ranked_choice(district_data: dict, n_voters: int = 1000,
                      n_simulations: int = 50) -> dict[str, str]:
    """
    Run ranked choice simulation across all districts.
    Returns {district: winner}.
    """
    rng = np.random.default_rng(42)
    winners = {}
    n_districts = len(district_data)

    for i, (district, votes) in enumerate(district_data.items()):
        total = sum(votes.values())
        shares = {p: v / total for p, v in votes.items()}

        party_names = list(votes.keys())
        party_pos = np.array([PARTIES[p] for p in party_names])

        wins = Counter()
        for _ in range(n_simulations):
            rankings, pnames = generate_ballots(shares, n_voters, rng)
            winner = instant_runoff_np(rankings, pnames, party_pos)
            wins[winner] += 1

        seat_winner = wins.most_common(1)[0][0]
        winners[district] = seat_winner
        if (i + 1) % 50 == 0 or i == n_districts - 1:
            print(f"  Processed {i+1}/{n_districts} districts...", flush=True)

    return winners


def instant_runoff_range(vote_shares: dict[str, float],
                         loyalty: float = 0.8) -> str:
    """
    Deterministic range-based instant runoff.

    Lay out parties left-to-right on a number line, each occupying a segment
    proportional to their vote share. Eliminate the smallest party each round.

    When a party is eliminated:
    - `loyalty` fraction (default 80%) of its votes split between the two
      neighboring surviving parties (based on original range proximity)
    - The remaining (1 - loyalty) fraction is distributed proportionally
      among ALL remaining parties, representing voters whose preference
      was idiosyncratic rather than spectrum-based.
    """
    # Sort parties by L-R position
    parties = sorted(vote_shares.keys(), key=lambda p: PARTIES[p])

    # Build initial ranges (fixed, never modified)
    original_ranges = {}
    cursor = 0.0
    for p in parties:
        width = vote_shares[p]
        original_ranges[p] = (cursor, cursor + width)
        cursor += width

    # Track current vote counts and the span of positions each party "owns"
    votes = {p: vote_shares[p] for p in parties}
    spans = {p: list(original_ranges[p]) for p in parties}

    active = list(parties)  # in L-R order

    while len(active) > 1:
        # Check for majority
        total = sum(votes[p] for p in active)
        for p in active:
            if votes[p] > total / 2:
                return p

        # Find party with fewest votes
        min_votes = min(votes[p] for p in active)
        # Tie-break: eliminate furthest from center
        candidates = [p for p in active if votes[p] == min_votes]
        eliminated = max(candidates, key=lambda p: abs(PARTIES[p] - 5.0))

        idx = active.index(eliminated)
        elim_span_left, elim_span_right = spans[eliminated]
        elim_total = votes[eliminated]

        if elim_total > 0:
            # Proportional portion: distribute to all remaining parties
            prop_amount = elim_total * (1 - loyalty)
            neighbor_amount = elim_total * loyalty

            remaining = [p for p in active if p != eliminated]
            remaining_total = sum(votes[p] for p in remaining)
            if remaining_total > 0:
                for p in remaining:
                    votes[p] += prop_amount * (votes[p] / remaining_total)

            # Neighbor portion: split between L-R neighbors
            left_neighbor = active[idx - 1] if idx > 0 else None
            right_neighbor = active[idx + 1] if idx < len(active) - 1 else None

            if left_neighbor and right_neighbor:
                ln_orig_right = original_ranges[left_neighbor][1]
                rn_orig_left = original_ranges[right_neighbor][0]
                split = (ln_orig_right + rn_orig_left) / 2
                split = max(elim_span_left, min(split, elim_span_right))

                span_width = elim_span_right - elim_span_left
                if span_width > 0:
                    left_frac = (split - elim_span_left) / span_width
                    votes[left_neighbor] += neighbor_amount * left_frac
                    votes[right_neighbor] += neighbor_amount * (1 - left_frac)
                    spans[left_neighbor][1] = split
                    spans[right_neighbor][0] = split
                else:
                    votes[right_neighbor] += neighbor_amount
            elif left_neighbor:
                votes[left_neighbor] += neighbor_amount
                spans[left_neighbor][1] = elim_span_right
            elif right_neighbor:
                votes[right_neighbor] += neighbor_amount
                spans[right_neighbor][0] = elim_span_left

        active.remove(eliminated)
        del votes[eliminated]
        del spans[eliminated]

    return active[0]


def run_ranked_choice_range(district_data: dict) -> dict[str, str]:
    """
    Deterministic range-based ranked choice across all districts.
    Returns {district: winner}.
    """
    winners = {}
    for district, dvotes in district_data.items():
        total = sum(dvotes.values())
        shares = {p: v / total for p, v in dvotes.items()}
        winners[district] = instant_runoff_range(shares)
    return winners


def plot_results(fptp_winners: dict, rcv_winners: dict, runoff_winners: dict,
                 median_winners: dict, district_data: dict):
    """Create comparison visualizations."""
    n_seats = len(district_data)

    fptp_seats = Counter(fptp_winners.values())
    rcv_seats = Counter(rcv_winners.values())
    runoff_seats = Counter(runoff_winners.values())
    med_seats = Counter(median_winners.values())

    national_votes = Counter()
    for votes in district_data.values():
        for party, count in votes.items():
            national_votes[party] += count
    total_national = sum(national_votes.values())

    all_parties_in_play = (set(fptp_seats.keys()) | set(rcv_seats.keys()) |
                          set(runoff_seats.keys()) | set(med_seats.keys()))
    for p, v in national_votes.items():
        if v / total_national > 0.02:
            all_parties_in_play.add(p)
    parties_sorted = sorted(all_parties_in_play, key=lambda p: PARTIES[p])
    short_sorted = [SHORT_NAMES[p] for p in parties_sorted]
    colors = [PARTY_COLORS[SHORT_NAMES[p]] for p in parties_sorted]

    fptp_vals = [fptp_seats.get(p, 0) for p in parties_sorted]
    rcv_vals = [rcv_seats.get(p, 0) for p in parties_sorted]
    runoff_vals = [runoff_seats.get(p, 0) for p in parties_sorted]
    med_vals = [med_seats.get(p, 0) for p in parties_sorted]
    prop_vals = [national_votes[p] / total_national * n_seats for p in parties_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"2025 Dutch Elections: 5 voting systems ({n_seats} gemeenten)",
                 fontsize=16, fontweight="bold")

    # ── Bar chart comparison ──
    ax = axes[0, 0]
    x = np.arange(len(short_sorted))
    w = 0.16
    ax.bar(x - 2*w, fptp_vals, w, label="FPTP", color=colors,
           edgecolor="black", linewidth=0.5)
    ax.bar(x - w, runoff_vals, w, label="Runoff", color=colors,
           edgecolor="black", linewidth=0.5, hatch="\\\\")
    ax.bar(x, rcv_vals, w, label="RCV (80/20)", color=colors,
           edgecolor="black", linewidth=0.5, hatch="//")
    ax.bar(x + w, med_vals, w, label="Median Voter", color=colors,
           edgecolor="black", linewidth=0.5, hatch="xx")
    ax.bar(x + 2*w, prop_vals, w, label="Proportional", color=colors,
           edgecolor="black", linewidth=0.5, hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(short_sorted, rotation=45, ha="right")
    ax.set_ylabel(f"Seats (out of {n_seats})")
    ax.set_title("Seat allocation comparison")
    ax.legend(fontsize=7)

    # ── Top 30: FPTP vs Runoff vs RCV vs Median ──
    ax = axes[0, 1]
    district_sizes = {d: sum(v.values()) for d, v in district_data.items()}
    top30 = sorted(district_sizes, key=district_sizes.get, reverse=True)[:30]
    y_pos = np.arange(len(top30))
    bar_h = 0.2
    systems = [
        ("FPTP", fptp_winners),
        ("Runoff", runoff_winners),
        ("RCV", rcv_winners),
        ("Median", median_winners),
    ]
    offsets = [1.5*bar_h, 0.5*bar_h, -0.5*bar_h, -1.5*bar_h]
    x_positions = [0.125, 0.375, 0.625, 0.875]
    for i, d in enumerate(top30):
        for (label, winners), offset, xp in zip(systems, offsets, x_positions):
            color = PARTY_COLORS[SHORT_NAMES[winners[d]]]
            ax.barh(i + offset, 1, bar_h, color=color, edgecolor="white", linewidth=0.3)
            ax.text(xp, i + offset, SHORT_NAMES[winners[d]], ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30, fontsize=6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([s[0] for s in systems], fontsize=7)
    ax.set_title("Top 30 gemeenten by size")
    ax.invert_yaxis()

    # ── Political spectrum ──
    ax = axes[1, 0]
    for party in parties_sorted:
        short = SHORT_NAMES[party]
        pos = PARTIES[party]
        vote_share = national_votes[party] / total_national * 100
        ax.scatter(pos, 0, s=vote_share * 80, color=PARTY_COLORS[short],
                   edgecolors="black", zorder=5)
        ax.annotate(short, (pos, 0), textcoords="offset points",
                    xytext=(0, 15 if parties_sorted.index(party) % 2 == 0 else -20),
                    ha="center", fontsize=8, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("Left \u2190 \u2192 Right")
    ax.set_title("Party positions on L-R spectrum (size = vote share)")

    # ── Seat summary table ──
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    headers = ["Party", "FPTP", "Runoff", "RCV", "Median", "Prop."]
    for i, p in enumerate(parties_sorted):
        table_data.append([
            short_sorted[i],
            str(fptp_vals[i]) if fptp_vals[i] > 0 else "-",
            str(runoff_vals[i]) if runoff_vals[i] > 0 else "-",
            str(rcv_vals[i]) if rcv_vals[i] > 0 else "-",
            str(med_vals[i]) if med_vals[i] > 0 else "-",
            f"{prop_vals[i]:.0f}" if prop_vals[i] >= 0.5 else "-",
        ])
    table = ax.table(cellText=table_data, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for i, p in enumerate(parties_sorted):
        table[i+1, 0].set_facecolor(colors[i])
        table[i+1, 0].set_text_props(color="white", fontweight="bold")
    ax.set_title("Seat allocation summary", pad=20)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("\nSaved results.png")
    plt.close()


def main():
    csv_path = Path(__file__).parent / "data2025" / "TK2025_uitslag.csv"
    print("Loading election data...")
    district_data = load_gemeente_data(csv_path)
    print(f"Loaded {len(district_data)} gemeenten\n")

    n_seats = len(district_data)

    # ── FPTP ──
    print("=" * 60)
    print("FIRST PAST THE POST")
    print("=" * 60)
    fptp_winners = fptp(district_data)
    fptp_seats = Counter(fptp_winners.values())
    for party, seats in fptp_seats.most_common():
        print(f"  {SHORT_NAMES[party]:8s}: {seats:3d} seats ({seats/n_seats*100:.1f}%)")

    # ── Runoff ──
    print()
    print("=" * 60)
    print("TWO-ROUND RUNOFF")
    print("=" * 60)
    runoff_winners = run_runoff(district_data)
    runoff_seats = Counter(runoff_winners.values())
    for party, seats in runoff_seats.most_common():
        print(f"  {SHORT_NAMES[party]:8s}: {seats:3d} seats ({seats/n_seats*100:.1f}%)")

    # ── Ranked Choice (range-based, 80/20) ──
    print()
    print("=" * 60)
    print("RANKED CHOICE — Range-based (80% neighbors, 20% proportional)")
    print("=" * 60)
    rcv_range_winners = run_ranked_choice_range(district_data)
    rcv_range_seats = Counter(rcv_range_winners.values())
    for party, seats in rcv_range_seats.most_common():
        print(f"  {SHORT_NAMES[party]:8s}: {seats:3d} seats ({seats/n_seats*100:.1f}%)")

    # ── Median Voter ──
    print()
    print("=" * 60)
    print("MEDIAN VOTER")
    print("=" * 60)
    median_winners = run_median_voter(district_data)
    median_seats = Counter(median_winners.values())
    for party, seats in median_seats.most_common():
        print(f"  {SHORT_NAMES[party]:8s}: {seats:3d} seats ({seats/n_seats*100:.1f}%)")

    # ── Proportional reference ──
    print()
    print("=" * 60)
    print("ACTUAL PROPORTIONAL RESULT (reference)")
    print("=" * 60)
    national_votes = Counter()
    for votes in district_data.values():
        for party, count in votes.items():
            national_votes[party] += count
    total = sum(national_votes.values())
    for party, votes in national_votes.most_common():
        prop_seats = votes / total * n_seats
        print(f"  {SHORT_NAMES[party]:8s}: {prop_seats:5.1f} proportional seats "
              f"({votes/total*100:.1f}%)")

    # ── Visualization ──
    plot_results(fptp_winners, rcv_range_winners, runoff_winners, median_winners,
                 district_data)


if __name__ == "__main__":
    main()
