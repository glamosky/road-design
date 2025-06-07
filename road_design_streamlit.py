import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pydeck as pdk

st.set_page_config(page_title="City Road Network Explorer", layout="wide")

# ----------------------
# Load Preprocessed Data
# ----------------------
@st.cache_data
def load_data():
    # Replace with your actual paths or dynamic data sources
    gdf_baguio = gpd.read_file("C:/Users/franj/notebooks/road design/road-design/gpkg_exports/baguio_roads.gpkg")
    gdf_zurich = gpd.read_file("C:/Users/franj/notebooks/road design/road-design/gpkg_exports/zurich_roads.gpkg")
    return gdf_baguio, gdf_zurich

gdf_baguio, gdf_zurich = load_data()

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.title("Controls")
cities = {"Baguio": gdf_baguio, "Zurich": gdf_zurich}
city = st.sidebar.selectbox("Select City", list(cities.keys()))
metric = st.sidebar.selectbox("Metric", ["Road Hierarchy"])
road_type_filter = st.sidebar.multiselect("Road Types", options=["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street"], default=None)

# Filter Data
gdf = cities[city]

if road_type_filter:
    gdf = gdf[gdf['highway'].isin(road_type_filter)]
    if gdf.empty:
        st.warning("⚠️ No roads match the selected road types.")
        st.stop()

if "highway" not in gdf.columns:
    st.error("🚫 No 'highway' column found in GeoDataFrame. Map not shown.")
    st.stop()

# After filtering
if gdf.empty:
    st.warning("⚠️ No data remains after filtering. Try selecting more road types.")
    st.stop()

# ----------------------
# Main Layout
# ----------------------
st.title("🚦 City Road Network Explorer")
st.markdown(f"### Analyzing **{metric}** in **{city}**")

# ----------------------
# Visualizations
# ----------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Road Network Map")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if "highway" in gdf.columns:
        gdf.plot(ax=ax, column='highway', legend=True, linewidth=1)
        ax.set_title(f"{city} - Road Types", fontsize=14)
    else:
        st.warning("No 'highway' column found in GeoDataFrame. Map not shown.")
        ax.set_title("Road data unavailable.")
    
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.subheader("Breakdown by Road Type")
    road_type_counts = gdf['highway'].value_counts().sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    road_type_counts.plot(kind='barh', ax=ax2, color='skyblue')
    ax2.set_xlabel("Number of Edges")
    ax2.set_ylabel("Road Type")
    st.pyplot(fig2)

# ----------------------
# Extra Metrics (Optional Curvature)
# ----------------------


# ----------------------
# Color by Road Type or Curvature
# ----------------------
# Load GeoJSON data
def load_geojson(path):
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)  # Ensure it's in WGS84
    gdf = gdf.dropna(subset=["curvature"])
    return gdf

# Sidebar options
city = st.sidebar.radio("Choose a city:", ("Baguio", "Zurich"))
color_by = st.sidebar.radio("Color roads by:", ("highway", "curvature"))

# Assign current data
gdf = gdf_baguio if city == "Baguio" else gdf_zurich

# Assign color value
if color_by == "highway":
    gdf["color"] = gdf["highway"].astype("category").cat.codes * 20  # simplistic encoding
else:
    gdf["color"] = gdf["curvature"]

# Prepare pydeck Layer
layer = pdk.Layer(
    "PathLayer",
    data=gdf,
    get_path="geometry.coordinates",
    get_width=2,
    get_color="[color, 100, 150]",
    pickable=True,
    auto_highlight=True,
)

# Get center
center = gdf.geometry.unary_union.centroid

view_state = pdk.ViewState(
    longitude=center.x,
    latitude=center.y,
    zoom=12,
    pitch=0
)

# Display map
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": f"{color_by}: {{{color_by}}}"}
))

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Made with 💚 using OpenStreetMap + Streamlit + GeoPandas")

