"""
Run this ONCE to embed your GeoJSON data into ChromaDB.
Usage: python scripts/ingest_data.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from chromadb.utils import embedding_functions
import geopandas as gpd


def create_tract_document(feature: dict, properties: dict) -> tuple[str, dict]:
    """Convert a GeoJSON feature into a text document + metadata for ChromaDB."""

    # Build a rich text description of the tract for semantic search
    parts = []

    if "GEOID" in properties or "geoid" in properties:
        geoid = properties.get("GEOID") or properties.get("geoid", "unknown")
        parts.append(f"Census tract {geoid} in Hillsborough County, Florida.")

    # Income
    income_keys = ["median_income", "med_income", "B19013_001E"]
    for k in income_keys:
        if k in properties and properties[k]:
            parts.append(f"Median household income: ${properties[k]:,.0f}.")
            break

    # Poverty
    poverty_keys = ["poverty_rate", "poverty_pct", "pov_rate"]
    for k in poverty_keys:
        if k in properties and properties[k]:
            parts.append(f"Poverty rate: {properties[k]:.1f}%.")
            break

    # Flood risk
    flood_keys = ["flood_risk", "flood_risk_score", "flood"]
    for k in flood_keys:
        if k in properties and properties[k]:
            score = properties[k]
            level = "high" if score > 0.6 else "moderate" if score > 0.3 else "low"
            parts.append(f"Flood risk: {level} ({score:.2f}).")
            break

    # Home values
    home_keys = ["median_home_value", "home_value", "med_home_val"]
    for k in home_keys:
        if k in properties and properties[k]:
            parts.append(f"Median home value: ${properties[k]:,.0f}.")
            break

    # Hospital distance
    hosp_keys = ["hospital_dist", "dist_to_hospital", "hosp_dist_miles"]
    for k in hosp_keys:
        if k in properties and properties[k]:
            parts.append(f"Distance to nearest hospital: {properties[k]:.1f} miles.")
            break

    doc_text = " ".join(parts) if parts else str(properties)

    # Metadata (for filtering in ChromaDB)
    metadata = {k: str(v) for k, v in properties.items() if k != "geometry" and v is not None}

    return doc_text, metadata


def ingest_geojson(geojson_path: str, collection_name: str = "hillsborough_tracts"):
    print(f"Loading {geojson_path}...")

    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"Found {len(features)} features.")

    # Set up ChromaDB with default embedding function (all-MiniLM-L6-v2)
    client = chromadb.PersistentClient(path="./vectorstore")

    ef = embedding_functions.DefaultEmbeddingFunction()

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    documents, metadatas, ids = [], [], []

    for i, feature in enumerate(features):
        props = feature.get("properties", {})
        doc_text, metadata = create_tract_document(feature, props)

        doc_id = props.get("GEOID") or props.get("geoid") or f"tract_{i}"

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(str(doc_id))

    # Batch upsert
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        print(f"  Embedded {min(i + batch_size, len(documents))}/{len(documents)} tracts...")

    print(f"\nDone! {len(documents)} tracts embedded into ChromaDB collection '{collection_name}'.")
    print("Vectorstore saved to ./vectorstore/")


if __name__ == "__main__":
    geojson_path = sys.argv[1] if len(sys.argv) > 1 else "data/hillsborough_tracts.geojson"
    ingest_geojson(geojson_path)