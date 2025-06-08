import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import osmnx as ox
import pydeck as pdk
import numpy as np
from shapely.geometry import Point
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="City Road Network Explorer", layout="wide")

# ----------------------
# Load Preprocessed Data
# ----------------------
@st.cache_data
def load_data():
    # Load road data
    gdf_baguio = gpd.read_file("gpkg_exports/baguio_roads.gpkg")
    gdf_zurich = gpd.read_file("gpkg_exports/zurich_roads.gpkg")
    
    # Ensure curvature column exists
    for gdf in [gdf_baguio, gdf_zurich]:
        if 'curvature' not in gdf.columns:
            # Calculate curvature as length / straight-line distance
            gdf['curvature'] = gdf.apply(
                lambda row: calculate_curvature(row.geometry), 
                axis=1
            )
    
    return gdf_baguio, gdf_zurich

def calculate_curvature(line_geom):
    """Calculate road curvature as ratio of actual length to straight line distance"""
    if line_geom is None:
        return 0
        
    # Get length of the line
    length = line_geom.length
    
    # Get straight line distance (start to end points)
    start_point = Point(line_geom.coords[0])
    end_point = Point(line_geom.coords[-1])
    straight_dist = start_point.distance(end_point)
    
    # Avoid division by zero
    if straight_dist == 0:
        return 1.0
        
    # Curvature ratio (higher = more curved)
    return length / straight_dist

@st.cache_data
def calculate_intersection_density(gdf, city):
    """Calculate intersection density by buffering around roads"""
    # Convert to projected CRS for accurate distance measurements
    if city == "Baguio":
        # Philippines UTM Zone 51N
        gdf_proj = gdf.to_crs(epsg=32651)
    else:
        # Switzerland LV95
        gdf_proj = gdf.to_crs(epsg=2056)
    
    # Extract nodes from linestrings
    nodes = []
    for idx, row in gdf_proj.iterrows():
        if row.geometry is not None:
            for coord in row.geometry.coords:
                nodes.append(Point(coord))
    
    # Create GeoDataFrame of nodes
    nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=gdf_proj.crs)
    
    # Count duplicated points (intersections)
    # Group by geometry and count
    intersection_points = nodes_gdf.dissolve(by=nodes_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))
    intersection_points['count'] = intersection_points.index.value_counts().values
    
    # Filter for actual intersections (points where multiple roads meet)
    intersections = intersection_points[intersection_points['count'] > 1].copy()
    
    # Create 500m grid cells
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    cell_size = 500  # meters
    
    # Create grid of points
    x_points = np.arange(xmin, xmax + cell_size, cell_size)
    y_points = np.arange(ymin, ymax + cell_size, cell_size)
    
    # Create empty density grid
    grid_cells = []
    
    # For each cell in the grid
    for x in x_points[:-1]:
        for y in y_points[:-1]:
            # Create a cell polygon
            cell = gpd.GeoDataFrame(
                {'geometry': [gpd.GeoSeries(Point(x+cell_size/2, y+cell_size/2)).buffer(cell_size/2, cap_style=3)]},
                crs=gdf_proj.crs
            )
            
            # Spatial join to count intersections in this cell
            joined = gpd.sjoin(intersections, cell, predicate='within')
            intersection_count = len(joined)
            
            # Add count to cell attributes
            cell['intersection_count'] = intersection_count
            cell['intersection_density'] = intersection_count / (cell_size**2 / 1000000)  # per sq km
            
            grid_cells.append(cell)
    
    if grid_cells:
        # Combine all cells into one GeoDataFrame
        density_grid = pd.concat(grid_cells)
        # Convert back to WGS84 for mapping
        density_grid = density_grid.to_crs(epsg=4326)
        return density_grid
    else:
        # Return empty GeoDataFrame with required columns
        return gpd.GeoDataFrame(
            {'geometry': [], 'intersection_count': [], 'intersection_density': []},
            crs="EPSG:4326"
        )

# Load data
gdf_baguio, gdf_zurich = load_data()

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.title("Controls")
cities = {"Baguio": gdf_baguio, "Zurich": gdf_zurich}
city = st.sidebar.selectbox("Select City", list(cities.keys()))

analysis_type = st.sidebar.selectbox(
    "Analysis Type", 
    ["Road Hierarchy", "Road Curvature", "Intersection Density"]
)

road_type_filter = st.sidebar.multiselect(
    "Road Types", 
    options=["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street"], 
    default=["primary", "secondary", "tertiary"]
)

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
st.markdown(f"### Analyzing **{analysis_type}** in **{city}**")

# ----------------------
# Visualizations
# ----------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Road Network Map")
    
    if analysis_type == "Road Hierarchy":
        # Use matplotlib for road hierarchy
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a categorical colormap
        highway_types = gdf['highway'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(highway_types)))
        cmap = mcolors.ListedColormap(colors)
        
        # Create a dictionary mapping highway types to color indices
        highway_to_color = {highway: i for i, highway in enumerate(highway_types)}
        gdf['color_idx'] = gdf['highway'].map(highway_to_color)
        
        gdf.plot(ax=ax, column='color_idx', cmap=cmap, linewidth=1.5, legend=False)
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors[i], lw=2, label=highway)
            for i, highway in enumerate(highway_types)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f"{city} - Road Types", fontsize=14)
        ax.axis('off')
        st.pyplot(fig)
        
    elif analysis_type == "Road Curvature":
        # Make sure we have curvature data
        if 'curvature' not in gdf.columns:
            gdf['curvature'] = gdf.apply(lambda row: calculate_curvature(row.geometry), axis=1)
        
        # Create folium map for curvature
        m = folium.Map(
            location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
            zoom_start=13,
            tiles="CartoDB positron"
        )
        
        # Create a colormap for curvature
        min_curvature = max(1.0, gdf['curvature'].min())
        max_curvature = min(2.0, gdf['curvature'].max())
        
        # Define a function to determine line color based on curvature
        def get_color(curvature):
            norm_curvature = (curvature - min_curvature) / (max_curvature - min_curvature)
            # Color gradient from green (straight) to red (curved)
            return f'#{int(255 * norm_curvature):02x}{int(255 * (1-norm_curvature)):02x}00'
        
        # Add lines to map
        for idx, row in gdf.iterrows():
            if row.geometry is not None:
                curvature = row['curvature']
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, curvature=curvature: {
                        'color': get_color(curvature),
                        'weight': 3,
                        'opacity': 0.7
                    },
                    tooltip=f"Highway: {row['highway']}, Curvature: {curvature:.2f}"
                ).add_to(m)
        
        # Add legend
        colormap = folium.LinearColormap(
            ['green', 'yellow', 'red'],
            vmin=min_curvature,
            vmax=max_curvature,
            caption='Road Curvature (ratio of length to straight-line distance)'
        )
        colormap.add_to(m)
        
        # Display map
        folium_static(m)
        
    elif analysis_type == "Intersection Density":
        # Calculate intersection density
        with st.spinner("Calculating intersection density..."):
            density_grid = calculate_intersection_density(gdf, city)
        
        if not density_grid.empty:
            # Create folium map for intersection density
            m = folium.Map(
                location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
                zoom_start=13,
                tiles="CartoDB positron"
            )
            
            # Add density grid as choropleth
            folium.Choropleth(
                geo_data=density_grid.__geo_interface__,
                data=density_grid['intersection_density'],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Intersection Density (per sq km)'
            ).add_to(m)
            
            # Add road network on top
            folium.GeoJson(
                gdf.__geo_interface__,
                style_function=lambda x: {
                    'color': 'blue',
                    'weight': 2,
                    'opacity': 0.5
                }
            ).add_to(m)
            
            # Display map
            folium_static(m)
        else:
            st.warning("Could not calculate intersection density. Check if the data has valid geometries.")

with col2:
    if analysis_type == "Road Hierarchy":
        st.subheader("Breakdown by Road Type")
        road_type_counts = gdf['highway'].value_counts().sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        road_type_counts.plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_xlabel("Number of Edges")
        ax2.set_ylabel("Road Type")
        st.pyplot(fig2)
    
    elif analysis_type == "Road Curvature":
        st.subheader("Curvature Distribution")
        
        # Calculate curvature stats
        mean_curvature = gdf['curvature'].mean()
        median_curvature = gdf['curvature'].median()
        max_curvature = gdf['curvature'].max()
        
        # Display metrics
        st.metric("Average Curvature", f"{mean_curvature:.2f}")
        st.metric("Median Curvature", f"{median_curvature:.2f}")
        st.metric("Max Curvature", f"{max_curvature:.2f}")
        
        # Create histogram
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        gdf['curvature'].hist(ax=ax3, bins=15)
        ax3.set_xlabel("Curvature Ratio")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Road Curvature")
        st.pyplot(fig3)
        
        # Curvature by road type
        st.subheader("Curvature by Road Type")
        curvature_by_type = gdf.groupby('highway')['curvature'].mean().sort_values(ascending=False)
        
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        curvature_by_type.plot(kind='barh', ax=ax4, color='orange')
        ax4.set_xlabel("Average Curvature")
        ax4.set_ylabel("Road Type")
        st.pyplot(fig4)
    
    elif analysis_type == "Intersection Density":
        if 'density_grid' in locals() and not density_grid.empty:
            st.subheader("Intersection Density Statistics")
            
            # Calculate density stats
            mean_density = density_grid['intersection_density'].mean()
            max_density = density_grid['intersection_density'].max()
            total_intersections = density_grid['intersection_count'].sum()
            
            # Display metrics
            st.metric("Total Intersections", f"{total_intersections}")
            st.metric("Avg Density (per sq km)", f"{mean_density:.2f}")
            st.metric("Max Density (per sq km)", f"{max_density:.2f}")
            
            # Create histogram
            fig5, ax5 = plt.subplots(figsize=(5, 4))
            density_grid['intersection_density'].hist(ax=ax5, bins=10)
            ax5.set_xlabel("Intersection Density (per sq km)")
            ax5.set_ylabel("Frequency")
            ax5.set_title("Distribution of Intersection Density")
            st.pyplot(fig5)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Made with 💚 using OpenStreetMap + Streamlit + GeoPandas")

