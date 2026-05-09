#!/usr/bin/env python3
"""Export gemeente shapes and election results as GeoJSON + JSON."""

import sys
import csv
import json
import geopandas as gpd
from collections import Counter

sys.path.insert(0, ".")
from ranked_choice import (
    PARTIES, SHORT_NAMES, PARTY_COLORS,
    load_gemeente_data, fptp, run_runoff, run_ranked_choice_range,
    run_median_voter,
)


def main():
    # Load and prep geo data
    gdf = gpd.read_file("gemeenten.gpkg")
    gdf = gdf[gdf["gemeentecode"] != "GM0998"]
    gdf["gcode"] = gdf["gemeentecode"].str.replace("GM", "G")
    gdf_wgs = gdf.to_crs(epsg=4326)

    # Load election data
    data = load_gemeente_data("data2025/TK2025_uitslag.csv")

    gemeente_codes = {}
    with open("data2025/TK2025_uitslag.csv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if row[1].startswith("G"):
                gemeente_codes[row[0]] = row[1]

    # Run all systems
    systems = {
        "fptp": fptp(data),
        "runoff": run_runoff(data),
        "rcv": run_ranked_choice_range(data),
        "median": run_median_voter(data),
    }

    # ── Export GeoJSON with gemeente code as feature id ──
    # Keep only gemeenten that have election data
    codes_with_data = set(gemeente_codes.values())
    gdf_export = gdf_wgs[gdf_wgs["gcode"].isin(codes_with_data)].copy()
    gdf_export = gdf_export[["gcode", "gemeentenaam", "geometry"]]
    # Dissolve duplicate features (multi-polygon parts) by gemeente code
    gdf_export = gdf_export.dissolve(by="gcode").reset_index()
    gdf_export = gdf_export.rename(columns={"gcode": "id", "gemeentenaam": "name"})
    gdf_export = gdf_export.set_index("id")
    gdf_export.to_file("static/gemeenten.geojson", driver="GeoJSON")
    print(f"Saved static/gemeenten.geojson ({len(gdf_export)} gemeenten)")

    # ── Export results.json ──
    results = {
        "party_colors": PARTY_COLORS,
        "party_positions": {SHORT_NAMES[p]: pos for p, pos in PARTIES.items()},
        "systems": {
            "fptp": {"label": "First Past the Post"},
            "runoff": {"label": "Two-Round Runoff"},
            "rcv": {"label": "Ranked Choice Voting"},
            "median": {"label": "Median Voter"},
            "proportional": {"label": "Proportional Representation"},
        },
        "gemeenten": {},
        "parliament": {},
    }

    # Per-gemeente data
    for name, votes in data.items():
        if name not in gemeente_codes:
            continue
        code = gemeente_codes[name]
        total = sum(votes.values())
        shares = {SHORT_NAMES[p]: round(v / total * 100, 1)
                  for p, v in sorted(votes.items(), key=lambda x: -x[1])}

        winners = {}
        for sys_name, sys_winners in systems.items():
            winners[sys_name] = SHORT_NAMES[sys_winners[name]]

        results["gemeenten"][code] = {
            "name": name,
            "shares": shares,
            "winners": winners,
        }

    # Parliament seat counts per system
    for sys_name, sys_winners in systems.items():
        seats = Counter(SHORT_NAMES[w] for w in sys_winners.values())
        results["parliament"][sys_name] = dict(seats.most_common())

    # Proportional parliament
    national_votes = Counter()
    for votes in data.values():
        for party, count in votes.items():
            national_votes[party] += count
    total_national = sum(national_votes.values())
    n_seats = len(data)
    prop_seats = {}
    for party, v in national_votes.items():
        s = round(v / total_national * n_seats)
        if s > 0:
            prop_seats[SHORT_NAMES[party]] = s
    results["parliament"]["proportional"] = dict(
        sorted(prop_seats.items(), key=lambda x: -x[1]))

    with open("static/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved static/results.json")

    # ── Render proportional background ──
    print("Rendering proportional background...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_map import compute_stripes, draw_striped_map
    from pyproj import Transformer

    # Render in EPSG:28992 (where stripe width in meters works)
    gdf_rd = gdf[gdf["gcode"].isin(codes_with_data)].copy()  # already in 28992
    striped = compute_stripes(data, gemeente_codes)

    fig, ax = plt.subplots(1, 1, figsize=(12, 14), dpi=200)
    draw_striped_map(ax, gdf_rd, striped, data, gemeente_codes)
    ax.set_axis_off()

    bounds_rd = gdf_rd.total_bounds  # minx, miny, maxx, maxy in RD
    margin_rd = 2000  # meters
    ax.set_xlim(bounds_rd[0] - margin_rd, bounds_rd[2] + margin_rd)
    ax.set_ylim(bounds_rd[1] - margin_rd, bounds_rd[3] + margin_rd)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig("static/proportional_bg.png", bbox_inches="tight", pad_inches=0,
                transparent=True, dpi=200)
    plt.close()

    # Convert the RD bounds to WGS84 for the mapbox image overlay
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds_rd[0] - margin_rd,
                                              bounds_rd[1] - margin_rd)
    max_lon, max_lat = transformer.transform(bounds_rd[2] + margin_rd,
                                              bounds_rd[3] + margin_rd)

    results["proportional_bounds"] = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    with open("static/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved static/proportional_bg.png")
    print("Done!")


if __name__ == "__main__":
    main()
