"""The Dutch districts (wijken) of 2021, with income, for notebooks/data.

The district map is CBS's Wijk- en Buurtkaart 2021 versie 3 (a GeoPackage from PDOK). Its income
columns were frozen before CBS published the income figures (mean income per inhabitant is filled
for 1,751 of 3,248 land districts), so every attribute is taken from the final StatLine table
85039NED, "Kerncijfers wijken en buurten 2021" (last corrected 2024-12-20), where it is filled for
3,105. The map gives only the geometry and the land/water flag. The geometry code is that of
build_netherlands.py.

Keeps the land districts (water = 'NEE') with every variable below published, builds queen and
rook contiguity, keeps the largest connected component, and writes

    netherlands_districts_2021.csv, netherlands_districts_2021_centroids.csv,
    netherlands_districts_2021_queen.gal, netherlands_districts_2021_rook.gal

Sources (CC BY 4.0):
  CBS, Wijk- en Buurtkaart 2021 versie 3,
    https://service.pdok.nl/cbs/wijken-en-buurten-2021/atom/downloads/wijkenbuurten_2021_v3.gpkg
  CBS StatLine 85039NED, https://opendata.cbs.nl/ODataApi/odata/85039NED/TypedDataSet with
    $filter=startswith(WijkenEnBuurten,'WK') and the $select of STATLINE below, saved as JSON

Run:  python -I build_netherlands_2021.py <wijkenbuurten_2021_v3.gpkg> <statline.json> <out dir> [--check]
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_netherlands import area_centroid, contiguity, csvq, largest_component, rings_of  # noqa: E402

STATLINE = {  # StatLine key: output name (counts become % of inhabitants below)
    "GemiddeldInkomenPerInwoner_72": "income",
    "GemiddeldeWOZWaardeVanWoningen_35": "house_value",
    "Bevolkingsdichtheid_33": "density",
    "Koopwoningen_40": "owner_occupied",
    "NietWestersTotaal_18": "non_western",
    "k_65JaarOfOuder_12": "aged_65_plus",
    "k_0Tot15Jaar_8": "under_15",
    "AantalInwoners_5": "population",
    "OppervlakteLand_112": "land_ha",
}
PERCENT_OF_POPULATION = ("non_western", "aged_65_plus", "under_15")
OUT = ["income", "house_value", "density", "owner_occupied", "non_western", "aged_65_plus",
       "under_15", "population", "land_ha"]


def main(gpkg, statline, out, check):
    stat = {r["WijkenEnBuurten"].strip(): r for r in json.load(open(statline, encoding="utf-8"))["value"]}
    db = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    rows = db.execute("SELECT wijkcode, wijknaam, gemeentecode, gemeentenaam, water, geom "
                      "FROM wijken ORDER BY wijkcode").fetchall()
    land = [r for r in rows if r[4] == "NEE"]
    print(f"districts {len(rows)}, land {len(land)}, in StatLine {sum(r[0] in stat for r in land)}")
    full, vals = [], []
    for r in land:
        s = stat.get(r[0])
        if s is None or any(s[k] is None for k in STATLINE):
            continue
        v = {STATLINE[k]: s[k] for k in STATLINE}
        if v["population"] <= 0 or v["income"] <= 0 or v["house_value"] <= 0 or v["density"] <= 0:
            continue
        for k in PERCENT_OF_POPULATION:
            v[k] = round(100 * v[k] / v["population"], 2)
        full.append(r); vals.append(v)
    print(f"with every variable published {len(full)}")
    geoms = [rings_of(r[-1]) for r in full]
    queen, rook = contiguity(geoms)
    keep, rest = largest_component(queen)
    print(f"queen components: {1 + len(rest)}; kept the largest, {len(keep)} districts; "
          f"dropped {sum(map(len, rest))} in {len(rest)} others")
    for comp in rest[:12]:
        print("   dropped:", ", ".join(f"{full[i][1]} ({full[i][3]})" for i in comp[:4]),
              "..." if len(comp) > 4 else "")
    idx = {old: new for new, old in enumerate(keep)}
    q = [sorted(idx[j] for j in queen[i] if j in idx) for i in keep]
    rk = [sorted(idx[j] for j in rook[i] if j in idx) for i in keep]
    print(f"links: queen {sum(map(len, q)) // 2}, rook {sum(map(len, rk)) // 2}; "
          f"queen neighbours per district {min(map(len, q))}-{max(map(len, q))}; "
          f"no rook neighbour {sum(1 for s in rk if not s)}")

    codes = [full[i][0] for i in keep]
    stem = f"{out}/netherlands_districts_2021"
    with open(f"{stem}.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("wijkcode,name,gemeentecode,municipality," + ",".join(OUT) + "\n")
        for i in keep:
            r = full[i]
            f.write(",".join([r[0], csvq(r[1]), r[2], csvq(r[3])] + [repr(vals[i][k]) if isinstance(vals[i][k], float)
                                                                        else str(vals[i][k]) for k in OUT]) + "\n")
    with open(f"{stem}_centroids.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("wijkcode,x_m,y_m\n")
        for i in keep:
            _, cx, cy = area_centroid(geoms[i])
            f.write(f"{full[i][0]},{cx:.1f},{cy:.1f}\n")
    for name, nb in (("queen", q), ("rook", rk)):
        with open(f"{stem}_{name}.gal", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"0 {len(keep)} wijken wijkcode\n")
            for c, s in zip(codes, nb):
                f.write(f"{c} {len(s)}\n{' '.join(codes[j] for j in s)}\n")
    print("wrote", len(keep), "districts to", out)

    if check:
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.strtree import STRtree
        shp = [MultiPolygon([Polygon(rs[0], rs[1:]) for rs in geoms[i]]) for i in keep]
        tree = STRtree(shp)
        sq = sr = dq = dr = 0
        for a, g in enumerate(shp):
            for b in tree.query(g, predicate="intersects"):
                b = int(b)
                if b <= a:
                    continue
                inter = g.intersection(shp[b])
                sq += 1
                sr += inter.length > 0
                dq += b not in q[a]
                dr += (inter.length > 0) != (b in rk[a])
        print(f"shapely: queen {sq} links ({dq} not found by vertex hashing), "
              f"rook {sr} ({dr} disagreements)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--check" in sys.argv)
