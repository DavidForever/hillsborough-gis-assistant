import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import retrieve_context
from spatial_engine import (
    get_geodataframe, parse_query_intent,
    apply_spatial_filters, format_spatial_results
)
from claude_client import query_claude
from response_parser import parse_claude_response
from map_renderer import render_map

st.set_page_config(
    page_title="Hillsborough GIS Assistant",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Hillsborough County GIS Assistant")
st.caption("Ask questions about census tracts, flood risk, income, and amenities in plain English.")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
if "query" not in st.session_state:
    st.session_state.query = ""

# Sidebar
with st.sidebar:
    st.header("Example queries")
    examples = [
        "Which census tracts have high flood risk and low median income?",
        "Show me tracts within 1 mile of a hospital that have above average home values",
        "What areas have the highest poverty rate near downtown Tampa?",
        "Where are the lowest income tracts with low flood risk?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.query = ex
            st.session_state.run_query = True
            st.rerun()

    st.divider()
    st.caption("Data: ACS 2022, FEMA, OSM")

# Query input
query = st.text_input(
    "Ask a geospatial question:",
    value=st.session_state.query,
    placeholder="e.g. Which tracts have high flood risk and low income?",
)

# Run analysis
run = st.session_state.pop("run_query", False)
if (st.button("🔍 Analyze", type="primary") or run) and query:

    with st.spinner("Running GIS analysis..."):
        try:
            gdf = get_geodataframe()
        except FileNotFoundError:
            st.error("GeoJSON file not found at data/hillsborough_tracts.geojson.")
            st.stop()

        retrieved_context = retrieve_context(query)
        filters = parse_query_intent(query)
        matched_gdf = apply_spatial_filters(filters, gdf)
        spatial_results = format_spatial_results(matched_gdf)
        raw_response = query_claude(query, retrieved_context, spatial_results)
        parsed = parse_claude_response(raw_response)

        # Store everything in session state
        st.session_state.results = {
            "parsed": parsed,
            "matched_gdf": matched_gdf,
            "gdf": gdf,
            "query": query
        }

# Display results from session state
if st.session_state.results:
    r = st.session_state.results
    parsed = r["parsed"]
    matched_gdf = r["matched_gdf"]
    gdf = r["gdf"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Analysis")
        st.write(parsed["text_answer"])

        if matched_gdf is not None and len(matched_gdf) > 0:
            st.metric("Matching tracts", len(matched_gdf))

            with st.expander("View matched tract data"):
                display_cols = [c for c in matched_gdf.columns if c != "geometry"][:10]
                st.dataframe(matched_gdf[display_cols].head(20))

    with col2:
        st.subheader("Map")
        folium_map = render_map(
            gdf=matched_gdf if len(matched_gdf) > 0 else gdf,
            highlight_tracts=parsed["highlight_tracts"],
            map_title=parsed["map_title"],
            color_field=parsed["color_field"]
        )
        st_folium(folium_map, width=700, height=500)