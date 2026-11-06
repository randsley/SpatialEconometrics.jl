# Portuguese municipal weights: provenance and attribution

> Informação geográfica cedida pela Direção-Geral do Território (CAOP, CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/)

Boundaries from **CAOP2025**, the Carta Administrativa Oficial de
Portugal, served by Direção-Geral do Território's OGC API at
`https://ogcapi.dgterritorio.gov.pt/collections/municipios/items?f=json&limit=400`,
retrieved 2026-11-06. CAOP is licensed
**CC-BY-4.0** with public access unrestricted; the credit above is a
condition of that licence, so keep it with any published result.

## Files

| file | contents |
|---|---|
| `portugal_municipalities_queen.gal` | contiguity by shared vertex, 739 links |
| `portugal_municipalities_rook.gal` | contiguity by shared edge, 737 links |
| `portugal_municipalities_centroids.csv` | area-weighted centroids, WGS84 |

Units are the **278 mainland municipalities**, ordered by DICO, which
is the same key the attribute CSV uses, so sorting lines the two up.
The 30 island municipalities are excluded deliberately: they share no
land border with anything, so including them would give 30 zero rows
and a matrix that cannot be row-standardised.

## Queen against rook

The two differ by exactly **2 pairs**, which meet at a single point rather than
along a border:

- Mêda and Sernancelhe
- Lisboa and Vila Franca de Xira

Mean degree is 5.317 under queen and 5.302 under rook. The choice
between them is usually presented as consequential; on these
boundaries it moves two links out of 739.

## How contiguity was computed

By hashing vertices and edges, not by intersecting polygons, because
CAOP is a clean planar partition: a shared border carries identical
vertices in both rings. That was measured, not assumed -- hashing
exact `Float64` pairs and hashing coordinates rounded to six decimals
give the same graph. So no geometry engine is involved and the result
is exact rather than tolerance-dependent.

Produced by `scripts/build_weights.jl` in PortugalINE.jl.
