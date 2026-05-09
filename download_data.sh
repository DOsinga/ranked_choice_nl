#!/usr/bin/env bash
# Fetch the raw election results and gemeente boundaries.
# Idempotent — safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

# 2023 Tweede Kamer election results
if [ ! -d data ]; then
    echo "Downloading 2023 election CSV..."
    curl -L -o tk2023_csv.zip \
        "https://data.overheid.nl/sites/default/files/dataset/e3fe6e42-06ab-4559-a466-a32b04247f68/resources/Verkiezingsuitslag%20Tweede%20Kamer%202023%20(CSV%20formaat).zip"
    unzip -o tk2023_csv.zip -d data
fi

# 2025 Tweede Kamer election results
if [ ! -d data2025 ]; then
    echo "Downloading 2025 election CSV..."
    curl -L -o tk2025_csv.zip \
        "https://data.overheid.nl/sites/default/files/dataset/a16f3352-c9ce-4831-a314-f989d442a258/resources/Verkiezingsuitslag%20Tweede%20Kamer%202025%20%28CSV%20Formaat%29.zip"
    unzip -o tk2025_csv.zip -d data2025
fi

# Gemeente boundaries from PDOK
if [ ! -f gemeenten.gpkg ]; then
    echo "Downloading gemeente boundaries..."
    python3 - <<'PY'
import requests, geopandas as gpd
url = ("https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0"
       "?service=WFS&version=2.0.0&request=GetFeature"
       "&typeName=wijkenbuurten:gemeenten&outputFormat=json"
       "&count=500&propertyName=gemeentecode,gemeentenaam,geom")
r = requests.get(url, timeout=180)
data = r.json()
gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:28992")
gdf.to_file("gemeenten.gpkg", driver="GPKG")
print(f"Saved gemeenten.gpkg ({len(gdf)} features)")
PY
fi

echo "Done."
