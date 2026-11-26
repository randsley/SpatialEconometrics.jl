"""The Dutch districts (wijken) of CBS's Wijk- en Buurtkaart 2023, for notebooks/data.

Reads the GeoPackage that PDOK serves (EPSG:28992, metres), with the Python standard library
only: a GeoPackage is SQLite, and each geometry is a GP header followed by WKB. Keeps the land
districts (water = 'NEE') with every variable below published (net labour participation is
left out: it is missing for 268 land districts, against 127 for the house value), builds queen and rook contiguity from
shared vertices and shared edges, keeps the largest connected component (the islands and any
district cut off by the dropped ones go), and writes

    netherlands_districts.csv, netherlands_districts_centroids.csv,
    netherlands_districts_queen.gal, netherlands_districts_rook.gal

`--check` compares the contiguity with shapely's (touching polygons, a shared edge of positive
length), if shapely is installed.

Source: Centraal Bureau voor de Statistiek (CBS), Wijk- en Buurtkaart 2023 versie 3, CC BY 4.0,
https://service.pdok.nl/cbs/wijken-en-buurten-2023/atom/downloads/wijkenbuurten_2023.gpkg

Run:  python -I build_netherlands.py <wijkenbuurten_2023.gpkg> <out dir> [--check]
"""
import math
import sqlite3
import struct
import sys
from collections import defaultdict

FIELDS = {  # output name: (CBS column, description)
    "house_value": ("gemiddelde_woningwaarde", "mean WOZ value of a dwelling, x 1,000 euro"),
    "density": ("bevolkingsdichtheid_inwoners_per_km2", "inhabitants per km2"),
    "owner_occupied": ("percentage_koopwoningen", "% of dwellings owner-occupied"),
    "non_european": ("percentage_met_herkomstland_buiten_europa", "% with a country of origin outside Europe"),
    "population": ("aantal_inwoners", "inhabitants"),
    "land_ha": ("oppervlakte_land_in_ha", "land area, ha"),
}


def rings_of(blob):
    """The polygons of a GeoPackage geometry, each a list of rings of (x, y)."""
    if blob[:2] != b"GP":
        raise ValueError("not a GeoPackage geometry")
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    wkb = memoryview(blob)[8 + env:]
    out, _ = _wkb(wkb, 0)
    return out


def _wkb(b, o):
    end = "<" if b[o] == 1 else ">"
    (t,) = struct.unpack_from(end + "I", b, o + 1)
    o += 5
    t2 = t % 1000
    dim = 3 if (t // 1000 in (1, 2)) else 4 if t // 1000 == 3 else 2
    if t2 == 3:
        (nr,) = struct.unpack_from(end + "I", b, o); o += 4
        rings = []
        for _ in range(nr):
            (npt,) = struct.unpack_from(end + "I", b, o); o += 4
            pts = struct.unpack_from(end + "d" * (dim * npt), b, o); o += 8 * dim * npt
            rings.append([(pts[dim * i], pts[dim * i + 1]) for i in range(npt)])
        return [rings], o
    if t2 == 6:
        (np_,) = struct.unpack_from(end + "I", b, o); o += 4
        polys = []
        for _ in range(np_):
            p, o = _wkb(b, o)
            polys.extend(p)
        return polys, o
    raise ValueError(f"WKB type {t} is not a polygon or multipolygon")


def area_centroid(polys):
    """Area and area-weighted centroid over all parts (holes subtract)."""
    A = cx = cy = 0.0
    for rings in polys:
        for r in rings:
            for (x0, y0), (x1, y1) in zip(r, r[1:]):
                c = x0 * y1 - x1 * y0
                A += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    A /= 2
    return abs(A), cx / (6 * A), cy / (6 * A)


def contiguity(geoms):
    """Queen (a shared vertex) and rook (a shared edge) neighbour sets."""
    vert = defaultdict(set)
    edge = defaultdict(set)
    for i, polys in enumerate(geoms):
        for rings in polys:
            for r in rings:
                for a, b in zip(r, r[1:]):
                    vert[a].add(i)
                    edge[(a, b) if a < b else (b, a)].add(i)
    queen = [set() for _ in geoms]
    rook = [set() for _ in geoms]
    for s in vert.values():
        for i in s:
            queen[i] |= s
    for s in edge.values():
        for i in s:
            rook[i] |= s
    for i in range(len(geoms)):
        queen[i].discard(i); rook[i].discard(i)
    return queen, rook


def largest_component(nb):
    seen = [-1] * len(nb)
    comps = []
    for s in range(len(nb)):
        if seen[s] >= 0:
            continue
        stack = [s]; seen[s] = len(comps); comp = [s]
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if seen[j] < 0:
                    seen[j] = len(comps); comp.append(j); stack.append(j)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return sorted(comps[0]), comps[1:]


def csvq(s):
    s = str(s)
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


def main(gpkg, out, check):
    db = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    cols = ["wijkcode", "wijknaam", "gemeentecode", "gemeentenaam", "water"] + [c for c, _ in FIELDS.values()]
    rows = db.execute(f"SELECT {', '.join(cols)}, geom FROM wijken ORDER BY wijkcode").fetchall()
    land = [r for r in rows if r[4] == "NEE"]
    full = [r for r in land if all(isinstance(v, (int, float)) and v >= 0 for v in r[5:-1])
            and r[5] > 0 and r[6] > 0]
    print(f"districts {len(rows)}, land {len(land)}, with every variable published {len(full)}")
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
    nq = sum(map(len, q)) // 2; nr = sum(map(len, rk)) // 2
    print(f"links: queen {nq}, rook {nr}; queen neighbours per district {min(map(len, q))}-{max(map(len, q))}")
    isolated_rook = sum(1 for s in rk if not s)
    print(f"districts with no rook neighbour: {isolated_rook}")

    codes = [full[i][0] for i in keep]
    with open(f"{out}/netherlands_districts.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("wijkcode,name,gemeentecode,municipality," + ",".join(FIELDS) + "\n")
        for i in keep:
            r = full[i]
            f.write(",".join([r[0], csvq(r[1]), r[2], csvq(r[3])] + [repr(v) if isinstance(v, float) else str(v)
                                                                        for v in r[5:-1]]) + "\n")
    with open(f"{out}/netherlands_districts_centroids.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("wijkcode,x_m,y_m\n")
        for i in keep:
            _, cx, cy = area_centroid(geoms[i])
            f.write(f"{full[i][0]},{cx:.1f},{cy:.1f}\n")
    for name, nb in (("queen", q), ("rook", rk)):
        with open(f"{out}/netherlands_districts_{name}.gal", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"0 {len(keep)} wijken wijkcode\n")
            for c, s in zip(codes, nb):
                f.write(f"{c} {len(s)}\n{' '.join(codes[j] for j in s)}\n")
    print("wrote", len(keep), "districts to", out)

    if check:
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.strtree import STRtree
        shp = [MultiPolygon([Polygon(rs[0], rs[1:]) for rs in geoms[i]]) for i in keep]
        tree = STRtree(shp)
        sq = sr = 0; dq = dr = 0
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
    main(sys.argv[1], sys.argv[2], "--check" in sys.argv)
