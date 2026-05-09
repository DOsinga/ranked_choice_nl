# Ranked Choice NL

What if the Netherlands used a different voting system? This project simulates
the 2025 Tweede Kamer election under five different systems, treating each of
the 343 gemeenten (Dutch municipalities) as a single-member district.

## [Live Demo](https://douwe.com/projects/ranked_choice_nl)

## Voting systems compared

- **First Past the Post** — whichever party has the most votes wins the gemeente
- **Two-Round Runoff** — top two parties advance, others' votes split based on left-right proximity
- **Ranked Choice (Instant Runoff)** — repeatedly eliminate the smallest party, redistributing voters to neighbors on the political spectrum (with a 20% proportional spillover)
- **Median Voter** — whoever the 50th-percentile voter on the spectrum supports
- **Proportional Representation** — what the Netherlands actually uses, applied per gemeente as a striped color overlay

Each party is placed on a 0-10 left-right scale using positions from the
[Chapel Hill Expert Survey 2024](https://www.chesdata.eu/2024-chapel-hill-expert-survey-ches).
The 1-D spectrum is the only assumption beyond the actual votes.

## How to reproduce

```bash
# 1. Get the raw election data from the Kiesraad
./download_data.sh

# 2. Run the simulations and produce the static artifacts the page consumes
python export_data.py
```

This writes:
- `static/gemeenten.geojson` — gemeente boundaries (WGS84)
- `static/results.json` — vote shares + per-system winners + party metadata
- `static/proportional_bg.png` — diagonally striped background image for the proportional view

The page itself (`ranked_choice_nl.html`) is pure JS — it loads those three
files and renders everything with Plotly. Nothing server-side at runtime.

## Files

- `ranked_choice_nl.html` — the page rendered on douwe.com
- `ranked_choice.py` — voting system implementations (FPTP, RCV, runoff, median)
- `export_data.py` — pipeline: parse CSV → simulate → write JSON/GeoJSON/PNG
- `make_map.py` — alternative matplotlib output (companion to the interactive page)
- `download_data.sh` — fetch the raw election results

## Data sources

- 2023 election: [data.overheid.nl/dataset/verkiezingsuitslag-tweede-kamer-2023](https://data.overheid.nl/dataset/verkiezingsuitslag-tweede-kamer-2023)
- 2025 election: [data.overheid.nl/dataset/verkiezingsuitslag-tweede-kamer-2025](https://data.overheid.nl/dataset/verkiezingsuitslag-tweede-kamer-2025)
- Gemeente boundaries: [PDOK CBS Wijk- en Buurtkaart](https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0)
- Party left-right positions: Chapel Hill Expert Survey 2024
- All under CC-0 / open licenses.

## Notes on the model

The 1-D left-right spectrum is a heavy simplification. Dutch politics has
religious, urban-rural, and EU-policy dimensions that don't reduce to a single
axis. CDA and D66 sit close together on the CHES scale but vote very
differently in practice. Treat the simulation as a thought experiment, not a
prediction.

There's a known dynamic where centrist parties always end up winning under
ranked choice in this model: any wing party's voters eventually flow toward
the center as smaller centrist parties get eliminated. The 20% proportional
spillover in our RCV implementation softens this but doesn't eliminate it —
that's a property of any 1-D Condorcet system, not a bug.
