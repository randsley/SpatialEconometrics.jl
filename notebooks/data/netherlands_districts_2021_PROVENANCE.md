# Dutch districts (wijken) of 2021, with income: provenance and attribution

Two sources, both from CBS and both with a credit that is a condition of
their licence. Keep it with any published result.

> Bron: Centraal Bureau voor de Statistiek (CBS), Kerncijfers wijken en buurten 2021 (StatLine 85039NED) and Wijk- en Buurtkaart 2021 versie 3, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

Both are licensed **CC BY 4.0**: reuse is free, commercial or not, with
credit to the source and a note of any changes. The changes made here are
listed below. The package's MIT licence covers its code, not these files.

## Why a second Dutch file

`netherlands_districts.csv` (the 2023 map) has no usable income figures.
CBS adds income to the key figures about two years after the reference
year, after the map file is frozen. In the 2021 map, mean income per
inhabitant is filled for 1,751 of 3,248 land districts. In the final
StatLine table for 2021 it is filled for 3,105. So the attributes come
from StatLine and only the geometry from the map.

## What was downloaded

- `wijkenbuurten_2021_v3.gpkg` from
  <https://service.pdok.nl/cbs/wijken-en-buurten-2021/atom/downloads/wijkenbuurten_2021_v3.gpkg>
  (198,672,384 bytes, SHA-256 `04b27965703e8c83…`), downloaded 2026-11-26.
  Layer `wijken`: 3,331 districts in EPSG:28992 (Amersfoort / RD New,
  metres). Used for the geometry and the `water` flag only. The geometry is
  not redistributed here.
- StatLine table 85039NED, "Kerncijfers wijken en buurten 2021", last
  modified 2024-12-20 (reason: correction), from the CBS OData API:
  `https://opendata.cbs.nl/ODataApi/odata/85039NED/TypedDataSet` with
  `$filter=startswith(WijkenEnBuurten,'WK')`, `$format=json` and a
  `$select` of the keys below, downloaded 2026-11-26 (3,248 districts,
  SHA-256 `a683fb38a1e4ed5a…`).

## What was built, and how

`build/build_netherlands_2021.py` (Python standard library, with the
geometry code of `build/build_netherlands.py`; `--check` uses shapely).
Run as
`python -I build/build_netherlands_2021.py wijkenbuurten_2021_v3.gpkg statline_85039NED_wijken.json . --check`.

1. **Land districts.** `water = 'NEE'` in the map: 3,248 of 3,331, all
   of them in StatLine.
2. **Complete cases.** A district is kept only when StatLine publishes
   every variable below, and the population, income, house value and
   density are positive: 3,086.
3. **Contiguity.** Queen and rook from exact shared vertices and edges,
   as for the 2023 file. `--check` agrees with shapely: 8,359 queen and
   8,066 rook links, no disagreement.
4. **One connected region.** The largest queen component is kept: 3,027
   districts. The 59 dropped are in 10 components: the Wadden islands
   (Texel, Vlieland, Terschelling, Ameland); Hulst (Zeeuws-Vlaanderen),
   which touches the rest of the country only through districts with no
   published value; Lelystad, Wageningen, Zandvoort and Bebouwde kom
   Doorwerth (Renkum), cut off by their own rural or industrial districts
   for the same reason; Pernis (Rotterdam) and Kern Overslag (Terneuzen).
5. **Centroids.** Area-weighted over all parts of each district, in
   EPSG:28992 metres.

## Files

| File | Contents |
|---|---|
| `netherlands_districts_2021.csv` | `wijkcode`, `name`, `gemeentecode`, `municipality` (from the map), and the variables below, in `wijkcode` order |
| `netherlands_districts_2021_centroids.csv` | `wijkcode`, `x_m`, `y_m` (EPSG:28992) |
| `netherlands_districts_2021_queen.gal`, `..._rook.gal` | contiguity, same order; header `0 3027 wijken wijkcode` |

| Column | StatLine key | Meaning |
|---|---|---|
| `income` | `GemiddeldInkomenPerInwoner_72` | mean personal income per inhabitant of private households, × 1,000 euro, 2021 |
| `house_value` | `GemiddeldeWOZWaardeVanWoningen_35` | mean WOZ (tax) value of a dwelling, × 1,000 euro (value date 1 January 2020) |
| `density` | `Bevolkingsdichtheid_33` | inhabitants per km² |
| `owner_occupied` | `Koopwoningen_40` | % of dwellings owner-occupied |
| `non_western` | `NietWestersTotaal_18` | % of inhabitants with a non-Western migration background (count ÷ `population` × 100) |
| `aged_65_plus` | `k_65JaarOfOuder_12` | % of inhabitants aged 65 or over (count ÷ `population` × 100) |
| `under_15` | `k_0Tot15Jaar_8` | % of inhabitants under 15 (count ÷ `population` × 100) |
| `population` | `AantalInwoners_5` | inhabitants |
| `land_ha` | `OppervlakteLand_112` | land area, ha |

The three percentages of inhabitants are computed here from CBS's counts,
rounded to two decimals. CBS rounds the counts to multiples of 5 (all of
them in this table, the population included), and the other columns are
copied as published. The 143 land districts with no published income are
small: at most 1,510 inhabitants, median 45.
