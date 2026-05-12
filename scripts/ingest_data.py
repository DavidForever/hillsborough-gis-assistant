import json
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
import faiss


def create_tract_document(properties: dict) -> str:
    parts = []

    geoid = properties.get("GEOID") or properties.get("geoid")
    if geoid:
        parts.append(f"Census tract {geoid} in Hillsborough County, Florida.")

    income_keys = ["median_income", "med_income", "B19013_001E"]
    for k in income_keys:
        if k in properties and properties[k]:
            parts.append(f"Median household income: ${float(properties[k]):,.0f}.")
            break

    poverty_keys = ["poverty_rate", "poverty_pct", "pov_rate"]
    for k in poverty_keys:
        if k in properties and properties[k]:
            parts.append(f"Poverty rate: {float(properties[k]):.1f}%.")
            break

    flood_keys = ["flood_risk", "flood_risk_score", "flood"]
    for k in flood_keys:
        if k in properties and properties[k]:
            score = float(properties[k])
            level = "high" if score > 0.6 else "moderate" if score > 0.3 else "low"
            parts.append(f"Flood risk: {level} ({score:.2f}).")
            break

    home_keys = ["median_home_value", "home_value", "med_home_val"]
    for k in home_keys:
        if k in properties and properties[k]:
            parts.append(f"Median home value: ${float(properties[k]):,.0f}.")
            break

    hosp_keys = ["hospital_dist", "dist_to_hospital", "hosp_dist_miles"]
    for k in hosp_keys:
        if k in properties and properties[k]:
            parts.append(f"Distance to nearest hospital: {float(properties[k]):.1f} miles.")
            break

    return " ".join(parts) if parts else str(properties)


def ingest_geojson(geojson_path: str):
    print(f"Loading {geojson_path}...")

    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"Found {len(features)} features.")

    documents = []
    for feature in features:
        props = feature.get("properties", {})
        doc = create_tract_document(props)
        documents.append(doc)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding documents...")
    embeddings = model.encode(documents, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    with open("vectorstore/faiss_index.pkl", "wb") as f:
        pickle.dump({"index": index, "documents": documents}, f)

    print(f"\nDone! {len(documents)} tracts embedded.")
    print("Saved to vectorstore/faiss_index.pkl")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/hillsborough_tracts.geojson"
    ingest_geojson(path)