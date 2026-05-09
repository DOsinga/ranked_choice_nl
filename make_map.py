#!/usr/bin/env python3
"""Generate 5 separate map PNGs, each with parliament half-circle."""

import sys
import csv
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.affinity import translate
import numpy as np
from collections import Counter

sys.path.insert(0, ".")
from ranked_choice import (
    PARTIES, SHORT_NAMES, PARTY_COLORS,
    load_gemeente_data, fptp, run_runoff, run_ranked_choice_range,
    run_median_voter,
)


def draw_parliament(ax, seats_counter: Counter, title: str):
    """
    Draw a half-circle parliament diagram with percentages.
    All parties labeled — small ones get leader lines.
    """
    parties_sorted = sorted(seats_counter.keys(), key=lambda p: PARTIES[p])
    short_parties = [SHORT_NAMES[p] for p in parties_sorted]
    counts = [seats_counter[p] for p in parties_sorted]
    colors = [PARTY_COLORS[SHORT_NAMES[p]] for p in parties_sorted]
    total_seats = sum(counts)

    angles = [c / total_seats * 180 for c in counts]

    # Start from the left (180°) and go right (toward 0°)
    # so left-wing parties sit on the left side of the semicircle
    start_angle = 180

    for i, (angle, color, party) in enumerate(zip(angles, colors, short_parties)):
        wedge = mpatches.Wedge(
            center=(0.5, 0.0), r=0.45,
            theta1=start_angle - angle, theta2=start_angle,
            facecolor=color, edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(wedge)

        mid_angle_deg = start_angle - angle / 2
        mid_angle = np.radians(mid_angle_deg)
        pct = counts[i] / total_seats * 100
        label = f"{party} {pct:.0f}%"

        if angle > 15:
            lx = 0.5 + 0.28 * np.cos(mid_angle)
            ly = 0.28 * np.sin(mid_angle)
            ax.text(lx, ly, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        elif angle > 3:
            inner_r, outer_r = 0.46, 0.58
            ix = 0.5 + inner_r * np.cos(mid_angle)
            iy = inner_r * np.sin(mid_angle)
            ox = 0.5 + outer_r * np.cos(mid_angle)
            oy = outer_r * np.sin(mid_angle)
            ax.plot([ix, ox], [iy, oy], color=color, linewidth=1.2)
            ha = "left" if mid_angle_deg > 90 else "right"
            ax.text(ox, oy, label, ha=ha, va="center",
                    fontsize=7, fontweight="bold", color=color)

        start_angle -= angle

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.08, 0.70)
    ax.set_aspect("equal")
    ax.set_axis_off()


def make_single_map(gdf, color_col, parliament_seats, title, filename,
                    striped_data=None, district_data=None, gemeente_codes=None):
    """
    Generate one map PNG with parliament half-circle.
    If striped_data is provided, draw striped municipalities for proportional.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.98)

    # Map on left
    ax_map = fig.add_axes([0.0, 0.02, 0.58, 0.90])

    if striped_data is not None:
        # Proportional: draw striped gemeenten
        draw_striped_map(ax_map, gdf, striped_data, district_data, gemeente_codes)
    else:
        gdf["_color"] = gdf[color_col].map(PARTY_COLORS)
        gdf.plot(ax=ax_map, color=gdf["_color"], edgecolor="white", linewidth=0.2)

    ax_map.set_axis_off()
    ax_map.set_xlim(10000, 280000)
    ax_map.set_ylim(300000, 620000)

    # Parliament on right
    ax_parl = fig.add_axes([0.54, 0.15, 0.44, 0.55])
    draw_parliament(ax_parl, parliament_seats, title)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


def draw_striped_map(ax, gdf, striped_data, district_data, gemeente_codes):
    """
    Draw municipalities with diagonal stripes in 1-3 colors.
    Uses repeating diagonal bands clipped to each municipality shape.
    """
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

    # Stripe width in map coordinates (EPSG:28992 = meters)
    STRIPE_WIDTH = 1500

    def draw_poly(ax, coords, color, edgecolor="none", lw=0):
        ax.add_patch(plt.Polygon(np.array(coords), facecolor=color,
                                 edgecolor=edgecolor, linewidth=lw))

    def draw_geom(ax, geom, color, edgecolor="none", lw=0):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            draw_poly(ax, geom.exterior.coords, color, edgecolor, lw)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                draw_poly(ax, poly.exterior.coords, color, edgecolor, lw)
        elif geom.geom_type == "GeometryCollection":
            for g in geom.geoms:
                if hasattr(g, "exterior"):
                    draw_poly(ax, g.exterior.coords, color, edgecolor, lw)

    for _, row in gdf.iterrows():
        gcode = row["gcode"]
        geom = row["geometry"]

        if gcode not in striped_data:
            draw_geom(ax, geom, "#cccccc", "white", 0.2)
            continue

        stripe_colors = striped_data[gcode]
        n_colors = len(stripe_colors)

        if n_colors == 1:
            draw_geom(ax, geom, stripe_colors[0], "white", 0.2)
            continue

        # Create diagonal stripe bands at 45 degrees
        minx, miny, maxx, maxy = geom.bounds
        # Diagonal span: project onto the 45-degree axis
        diag_min = minx + miny
        diag_max = maxx + maxy
        margin = STRIPE_WIDTH * 2
        diag_start = diag_min - margin
        diag_end = diag_max + margin

        # Generate diagonal bands and clip to the gemeente
        d = diag_start
        while d < diag_end:
            color_idx = int((d - diag_start) / STRIPE_WIDTH) % n_colors
            color = stripe_colors[color_idx]

            # A diagonal band is a parallelogram: x+y between d and d+STRIPE_WIDTH
            # Build a large parallelogram that covers the bbox
            band_lo = d
            band_hi = d + STRIPE_WIDTH
            band_poly = ShapelyPolygon([
                (minx - margin, band_lo - (minx - margin)),
                (maxx + margin, band_lo - (maxx + margin)),
                (maxx + margin, band_hi - (maxx + margin)),
                (minx - margin, band_hi - (minx - margin)),
            ])

            clipped = geom.intersection(band_poly)
            if not clipped.is_empty:
                draw_geom(ax, clipped, color)

            d += STRIPE_WIDTH

        # Draw outline on top
        draw_geom(ax, geom, "none", "white", 0.3)


def compute_stripes(district_data, gemeente_codes):
    """
    For each gemeente, determine 1-3 stripe colors:
    - Top party > 45%: 1 stripe (solid)
    - Top 2 > 45%: 2 stripes
    - Otherwise: 3 stripes with top 3 parties
    """
    result = {}
    for name, votes in district_data.items():
        if name not in gemeente_codes:
            continue
        code = gemeente_codes[name]
        total = sum(votes.values())
        sorted_parties = sorted(votes.keys(), key=lambda p: votes[p], reverse=True)

        top1_pct = votes[sorted_parties[0]] / total
        if top1_pct > 0.40:
            result[code] = [PARTY_COLORS[SHORT_NAMES[sorted_parties[0]]]]
        else:
            top2_pct = top1_pct + votes[sorted_parties[1]] / total
            if top2_pct > 0.40:
                result[code] = [
                    PARTY_COLORS[SHORT_NAMES[sorted_parties[0]]],
                    PARTY_COLORS[SHORT_NAMES[sorted_parties[1]]],
                ]
            else:
                result[code] = [
                    PARTY_COLORS[SHORT_NAMES[sorted_parties[0]]],
                    PARTY_COLORS[SHORT_NAMES[sorted_parties[1]]],
                    PARTY_COLORS[SHORT_NAMES[sorted_parties[2]]],
                ]
    return result


def main():
    # Load geo data
    gdf = gpd.read_file("gemeenten.gpkg")
    gdf = gdf[gdf["gemeentecode"] != "GM0998"]
    gdf["gcode"] = gdf["gemeentecode"].str.replace("GM", "G")

    # Load election data
    data = load_gemeente_data("data2025/TK2025_uitslag.csv")

    # Get gemeente codes for joining
    gemeente_codes = {}
    with open("data2025/TK2025_uitslag.csv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            code = row[1]
            if code.startswith("G"):
                gemeente_codes[row[0]] = code

    # Run all systems
    systems = {
        "First Past the Post": fptp(data),
        "Two-Round Runoff": run_runoff(data),
        "Ranked Choice Voting": run_ranked_choice_range(data),
        "Median Voter": run_median_voter(data),
    }

    # Join results to geodataframe
    col_names = {}
    for sys_name, winners in systems.items():
        col = sys_name.replace(" ", "_")
        col_names[sys_name] = col
        code_to_winner = {}
        for name, winner in winners.items():
            if name in gemeente_codes:
                code_to_winner[gemeente_codes[name]] = SHORT_NAMES[winner]
        gdf[col] = gdf["gcode"].map(code_to_winner)

    # Filter to mainland with data
    gdf_main = gdf[gdf[col_names["First Past the Post"]].notna()].copy()

    # Generate individual maps
    for sys_name, winners in systems.items():
        col = col_names[sys_name]
        seats = Counter(winners.values())
        filename = f"map_{col.lower()}.png"
        make_single_map(gdf_main, col, seats, sys_name, filename)

    # Proportional map with stripes
    print("Generating proportional map with stripes...")
    striped = compute_stripes(data, gemeente_codes)

    # Proportional parliament: national vote shares -> seats out of 346
    national_votes = Counter()
    for votes in data.values():
        for party, count in votes.items():
            national_votes[party] += count
    total_national = sum(national_votes.values())
    prop_seats = Counter()
    for party, v in national_votes.items():
        seats_f = v / total_national * len(data)
        if seats_f >= 0.5:
            prop_seats[party] = round(seats_f)

    make_single_map(gdf_main, None, prop_seats, "Proportional Representation",
                    "map_proportional.png",
                    striped_data=striped, district_data=data,
                    gemeente_codes=gemeente_codes)

    print("Done!")


if __name__ == "__main__":
    main()
