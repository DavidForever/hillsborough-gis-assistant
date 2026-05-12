import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import re

# Load GeoJSON once at module level
_gdf = None


def get_geodataframe(geojson_path: str = "data/hillsborough_tracts.geojson") -> gpd.GeoDataFrame:
    global _gdf
    if _gdf is None:
        _gdf = gpd.read_file(geojson_path)
        # Ensure CRS is set for distance calculations
        if _gdf.crs is None:
            _gdf = _gdf.set_crs("EPSG:4326")
    return _gdf


def parse_query_intent(user_query: str) -> dict:
    """
    Simple keyword-based intent parser.
    Returns a dict of filters to apply to the GeoDataFrame.
    LangChain will handle the sophisticated version — this is a fallback.
    """
    query_lower = user_query.lower()
    filters = {}

    # Flood risk
    if "flood risk" in query_lower or "flood" in query_lower:
        if "high" in query_lower:
            filters["flood_risk_min"] = 0.6
        elif "low" in query_lower:
            filters["flood_risk_max"] = 0.3

    # Income
    if "low income" in query_lower or "low median income" in query_lower:
        filters["income_below_median"] = True
    if "high income" in query_lower:
        filters["income_above_median"] = True

    # Home values
    if "high home value" in query_lower or "above average home" in query_lower:
        filters["home_value_above_avg"] = True

    # Poverty
    if "poverty" in query_lower:
        filters["high_poverty"] = True

    # Hospital proximity
    if "hospital" in query_lower:
        if "1 mile" in query_lower or "within 1" in query_lower:
            filters["hospital_distance_max_miles"] = 1.0
        elif "2 mile" in query_lower:
            filters["hospital_distance_max_miles"] = 2.0

    # Downtown Tampa
    if "downtown" in query_lower or "downtown tampa" in query_lower:
        filters["near_downtown_miles"] = 3.0

    return filters


def apply_spatial_filters(filters: dict, gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """Apply parsed filters to the GeoDataFrame and return matching tracts."""
    if gdf is None:
        gdf = get_geodataframe()

    result = gdf.copy()
    cols = result.columns.tolist()

    # Helper: find column by partial name match
    def find_col(keywords):
        for kw in keywords:
            matches = [c for c in cols if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    flood_col = find_col(["flood_risk", "flood"])
    income_col = find_col(["median_income", "income", "med_income"])
    poverty_col = find_col(["poverty", "poverty_rate", "pov"])
    home_val_col = find_col(["home_value", "median_value", "home_val"])
    hosp_col = find_col(["hospital_dist", "dist_hospital", "hosp"])

    if "flood_risk_min" in filters and flood_col:
        result = result[result[flood_col] >= filters["flood_risk_min"]]

    if "flood_risk_max" in filters and flood_col:
        result = result[result[flood_col] <= filters["flood_risk_max"]]

    if "income_below_median" in filters and income_col:
        median_val = gdf[income_col].median()
        result = result[result[income_col] < median_val]

    if "income_above_median" in filters and income_col:
        median_val = gdf[income_col].median()
        result = result[result[income_col] > median_val]

    if "high_poverty" in filters and poverty_col:
        p75 = gdf[poverty_col].quantile(0.75)
        result = result[result[poverty_col] >= p75]

    if "home_value_above_avg" in filters and home_val_col:
        avg_val = gdf[home_val_col].mean()
        result = result[result[home_val_col] > avg_val]

    if "hospital_distance_max_miles" in filters and hosp_col:
        max_dist = filters["hospital_distance_max_miles"]
        result = result[result[hosp_col] <= max_dist]

    if "near_downtown_miles" in filters:
        # Downtown Tampa approx coordinates
        downtown = Point(-82.4572, 27.9506)
        gdf_projected = result.to_crs("EPSG:3857")
        downtown_proj = gpd.GeoSeries([downtown], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        centroids = gdf_projected.geometry.centroid
        distances_miles = centroids.distance(downtown_proj) / 1609.34
        result = result[distances_miles <= filters["near_downtown_miles"]]

    return result


def format_spatial_results(gdf: gpd.GeoDataFrame, max_tracts: int = 20) -> str:
    """Format spatial results as a readable string for the Claude prompt."""
    if len(gdf) == 0:
        return "No census tracts matched the spatial filters."

    cols_to_show = [c for c in gdf.columns if c != "geometry"]
    summary_df = gdf[cols_to_show].head(max_tracts)

    lines = [f"Found {len(gdf)} matching census tracts. Showing up to {max_tracts}:\n"]

    for _, row in summary_df.iterrows():
        tract_info = " | ".join([f"{col}: {row[col]}" for col in cols_to_show[:8]])
        lines.append(f"• {tract_info}")

    if len(gdf) > max_tracts:
        lines.append(f"\n... and {len(gdf) - max_tracts} more tracts.")

    return "\n".join(lines)