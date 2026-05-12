import folium
import geopandas as gpd
import json
from typing import Optional

HILLSBOROUGH_CENTER = [27.9506, -82.4572]


def render_map(
        gdf: gpd.GeoDataFrame,
        highlight_tracts: list,
        map_title: str = "Hillsborough County",
        color_field: Optional[str] = None
) -> folium.Map:
    """
    Render a Folium choropleth map with highlighted census tracts.
    """
    m = folium.Map(
        location=HILLSBOROUGH_CENTER,
        zoom_start=10,
        tiles="CartoDB positron"
    )

    # Base layer: all tracts (light gray)
    folium.GeoJson(
        json.loads(gdf.to_json()),
        style_function=lambda feature: {
            "fillColor": "#e0e0e0",
            "color": "#999999",
            "weight": 0.5,
            "fillOpacity": 0.3
        },
        name="All tracts"
    ).add_to(m)

    # Highlighted tracts layer
    if highlight_tracts:
        # Try to find GEOID column
        geoid_col = None
        for col in ["GEOID", "geoid", "GEO_ID", "tract_id"]:
            if col in gdf.columns:
                geoid_col = col
                break

        if geoid_col:
            highlight_gdf = gdf[gdf[geoid_col].astype(str).isin(
                [str(t) for t in highlight_tracts]
            )]
        else:
            # If no GEOID col, highlight all returned tracts
            highlight_gdf = gdf.head(len(highlight_tracts))

        if len(highlight_gdf) > 0:
            def make_style(feature):
                return {
                    "fillColor": "#e74c3c",
                    "color": "#c0392b",
                    "weight": 1.5,
                    "fillOpacity": 0.6
                }

            def make_popup(feature):
                props = feature["properties"]
                rows = "".join(
                    f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                    for k, v in list(props.items())[:10]
                    if k not in ["geometry"] and v is not None
                )
                return folium.Popup(f"<table>{rows}</table>", max_width=300)

            folium.GeoJson(
                json.loads(highlight_gdf.to_json()),
                style_function=make_style,
                popup=folium.GeoJsonPopup(
                    fields=[c for c in highlight_gdf.columns if c != "geometry"][:6],
                    aliases=[c for c in highlight_gdf.columns if c != "geometry"][:6],
                ),
                name=map_title
            ).add_to(m)

    folium.LayerControl().add_to(m)

    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 8px 16px;
                border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                font-family: sans-serif; font-size: 14px; font-weight: 600;">
        {map_title}
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    return m