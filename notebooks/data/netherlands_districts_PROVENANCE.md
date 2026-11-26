# Dutch districts (wijken): provenance and attribution

One source, with a credit that is a condition of its licence. Keep it with
any published result.

> Bron: Centraal Bureau voor de Statistiek (CBS), Wijk- en Buurtkaart 2023 versie 3, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

CBS publishes the Wijk- en Buurtkaart (the district and neighbourhood map,
with its key figures attached) through PDOK, licensed **CC BY 4.0**: reuse
is free, commercial or not, with credit to the source and a note of any
changes. The changes made here are listed below. The package's MIT
licence covers its code, not these files.

## What was downloaded

- `wijkenbuurten_2023.gpkg` from
  <https://service.pdok.nl/cbs/wijken-en-buurten-2023/atom/downloads/wijkenbuurten_2023.gpkg>
  (200,855,552 bytes), downloaded 2026-11-26. Found through
  data.europa.eu (dataset `83042-wijk-en-buurtkaart-2023-versie-3`).
- Layer `wijken`: 3,434 districts in EPSG:28992 (Amersfoort / RD New,
  metres), with 237 columns. The geometry is not redistributed here.

## What was built, and how

`build/build_netherlands.py` (Python standard library; `--check` uses
shapely). Run as
`python -I build/build_netherlands.py wijkenbuurten_2023.gpkg . --check`.

1. **Land districts.** `water = 'NEE'`: 3,352 of 3,434. The 81 water
   districts and the one "Buitenland" record are dropped.
2. **Complete cases.** CBS codes a suppressed or unavailable figure as a
   negative number (−99997 and similar). A district is kept only when all
   of the variables below are published and the house value and density
   are positive: 3,219. Net labour participation is not used, because it
   is missing for 268 land districts, against 127 for the house value.
3. **Contiguity.** Queen (a shared vertex) and rook (a shared edge),
   from exact coordinates: neighbouring districts share their boundary
   vertices. `--check` confirms it against shapely (polygons that touch,
   and a shared boundary of positive length): 8,853 queen and 8,538 rook
   links, no disagreement.
4. **One connected region.** The largest queen component is kept: 3,182
   districts. The 37 dropped are in 7 components:
   - the Wadden islands (Texel, Vlieland, Terschelling, Ameland), which
     touch no other land district;
   - Almere, cut off because the districts on its edge (its business
     parks and rural fringe, "-kant" and "Overig ...", with 0 to 150
     inhabitants each) have no published house value;
   - Wageningen, Zandvoort and Bebouwde kom Doorwerth (Renkum), cut off
     by their own rural districts for the same reason;
   - Pernis (Rotterdam), whose only neighbours are two port districts
     (Waalhaven-Eemhaven, Vondelingenplaat) with no published house value.
5. **Centroids.** Area-weighted over all parts of each district, in
   EPSG:28992 metres.

## Files

| File | Contents |
|---|---|
| `netherlands_districts.csv` | `wijkcode`, `name`, `gemeentecode`, `municipality`, and the variables below, in `wijkcode` order |
| `netherlands_districts_centroids.csv` | `wijkcode`, `x_m`, `y_m` (EPSG:28992) |
| `netherlands_districts_queen.gal`, `..._rook.gal` | contiguity, same order; header `0 3182 wijken wijkcode` |

| Column | CBS column | Meaning |
|---|---|---|
| `house_value` | `gemiddelde_woningwaarde` | mean WOZ (tax) value of a dwelling, × 1,000 euro |
| `density` | `bevolkingsdichtheid_inwoners_per_km2` | inhabitants per km² |
| `owner_occupied` | `percentage_koopwoningen` | % of dwellings owner-occupied |
| `non_european` | `percentage_met_herkomstland_buiten_europa` | % of inhabitants with a country of origin outside Europe |
| `population` | `aantal_inwoners` | inhabitants |
| `land_ha` | `oppervlakte_land_in_ha` | land area, ha |

CBS rounds the percentages to whole numbers and the house values to
thousands of euro, and these are copied as published.

## Why 2023, and not the income figures

CBS adds income to the map about two years after the reference year. In
the 2023 map, mean income per inhabitant is published for 1,834 of the
3,352 land districts, and in the 2022 map for 1,804 of 3,340, so neither
supports an income model. The house value is published for 96% of land
districts in both.
