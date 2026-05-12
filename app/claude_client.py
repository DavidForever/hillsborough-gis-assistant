import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are a GIS analyst assistant specializing in Hillsborough County, Florida.
You have access to census tract data including demographics, flood risk scores, home values, 
poverty rates, and distances to amenities like hospitals and schools.

When a user asks a geospatial question, you will receive:
1. RETRIEVED CONTEXT: Relevant census tract data chunks from the vector database
2. SPATIAL RESULTS: GeoJSON features that match the spatial/attribute filters

Your job is to:
1. Write a clear, informative answer in plain English (2-4 sentences)
2. Summarize key statistics from the data (e.g. "17 tracts match, median income $32k")
3. Return a JSON block with the tract GEOIDs to highlight on the map

Always end your response with a JSON block in this exact format:
```json
{
  "highlight_tracts": ["12057XXXXXX", "12057XXXXXX"],
  "map_title": "Short descriptive title for the map",
  "color_field": "flood_risk_score"
}
```

If no spatial results match, still return the JSON block with an empty array.
Be specific with numbers. Reference actual tract GEOIDs when relevant."""


def query_claude(user_question: str, retrieved_context: str, spatial_results: str) -> str:
    """Send the augmented prompt to Claude and return the response."""

    augmented_prompt = f"""USER QUESTION: {user_question}

RETRIEVED CONTEXT (from vector database):
{retrieved_context}

SPATIAL RESULTS (census tracts matching the query filters):
{spatial_results}

Please answer the question using the data above, then provide the JSON block."""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": augmented_prompt}]
    )

    return message.content[0].text