import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import osmnx as ox
import pydeck as pdk
import numpy as np
from shapely.geometry import Point, LineString
import folium
from streamlit_folium import folium_static
import networkx as nx
import io
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import base64
import seaborn as sns

st.set_page_config(page_title="City Road Network Explorer", layout="wide",  page_icon="🛣️",  faviconlayout="wide",)

st.image("baguio and zurich collage.png")
# ----------------------
# Load Preprocessed Data
# ----------------------
@st.cache_data
def load_data():
    # Load road data
    # uncomment if deploying to streamlit cloud
    gdf_baguio = gpd.read_file("gpkg_exports/baguio_roads.gpkg")
    gdf_zurich = gpd.read_file("gpkg_exports/zurich_roads.gpkg")
    
    # uncomment if running locally
    # gdf_baguio = gpd.read_file("C:/Users/franj/notebooks/road design/road-design/gpkg_exports/baguio_roads.gpkg")
    # gdf_zurich = gpd.read_file("C:/Users/franj/notebooks/road design/road-design/gpkg_exports/zurich_roads.gpkg")
    
    # Calculate additional metrics for each road
    for gdf in [gdf_baguio, gdf_zurich]:
        # Calculate road length in meters
        if gdf.crs is not None:
            # Calculate length in meters (ensure it's in projected CRS for accuracy)
            if gdf.crs.is_geographic:
                # Convert to appropriate UTM zone if in geographic coordinates
                gdf['length_m'] = gdf.to_crs('+proj=utm +zone=51 +datum=WGS84').length
            else:
                gdf['length_m'] = gdf.length
        else:
            # Fallback if CRS is unknown
            gdf['length_m'] = gdf.geometry.length
        
        # Ensure curvature column exists
        if 'curvature' not in gdf.columns:
            # Calculate curvature as length / straight-line distance
            gdf['curvature'] = gdf.apply(
                lambda row: calculate_curvature(row.geometry), 
                axis=1
            )
        
        # Extract lanes if available
        if 'lanes' in gdf.columns:
            # Convert to numeric, handle errors
            gdf['lanes'] = pd.to_numeric(gdf['lanes'], errors='coerce')
        else:
            gdf['lanes'] = np.nan
        
        # Extract max speed if available
        if 'maxspeed' in gdf.columns:
            # Extract numeric values from maxspeed
            gdf['maxspeed'] = gdf['maxspeed'].str.extract('(\d+)').astype(float)
        else:
            gdf['maxspeed'] = np.nan
    
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
def calculate_intersection_density(_gdf, city):
    """Calculate intersection density by buffering around roads"""
    # Convert to projected CRS for accurate distance measurements
    if city == "Baguio":
        # Philippines UTM Zone 51N
        gdf_proj = _gdf.to_crs(epsg=32651)
    else:
        # Switzerland LV95
        gdf_proj = _gdf.to_crs(epsg=2056)
    
    # Extract nodes from linestrings
    nodes = []
    for idx, row in gdf_proj.iterrows():
        if row.geometry is not None and not row.geometry.is_empty:
            try:
                for coord in row.geometry.coords:
                    nodes.append(Point(coord))
            except (AttributeError, TypeError):
                # Skip geometries that don't have coords attribute
                continue
    
    # Create GeoDataFrame of nodes
    nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=gdf_proj.crs)
    
    # Handle case with no nodes
    if len(nodes_gdf) == 0:
        return gpd.GeoDataFrame(
            {'geometry': [], 'intersection_count': [], 'intersection_density': []},
            crs="EPSG:4326"
        )
    
    # Count duplicated points (intersections)
    # Group by geometry and count
    intersection_points = nodes_gdf.dissolve(by=nodes_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))
    intersection_points['count'] = intersection_points.index.value_counts().values
    
    # Filter for actual intersections (points where multiple roads meet)
    intersections = intersection_points[intersection_points['count'] > 1].copy()
    
    # Handle case with no intersections
    if len(intersections) == 0:
        return gpd.GeoDataFrame(
            {'geometry': [], 'intersection_count': [], 'intersection_density': []},
            crs="EPSG:4326"
        )
    
    # Create 500m grid cells
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    
    # Ensure grid area is reasonable (not too small, not too large)
    area_width = xmax - xmin
    area_height = ymax - ymin
    
    # If area is too small, use a smaller cell size
    if area_width < 1000 or area_height < 1000:
        cell_size = min(area_width, area_height) / 5  # Divide area into at least 5 cells
        cell_size = max(cell_size, 50)  # But no smaller than 50m
    else:
        cell_size = 500  # Default 500m grid cells
    
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

@st.cache_data
def create_road_network(_gdf):
    """Create a NetworkX graph from road GeoDataFrame for routing"""
    # Create an empty graph
    G = nx.Graph()
    
    # Process each road segment
    for idx, row in _gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
            
        try:
            # Get coordinates from the geometry
            coords = list(row.geometry.coords)
            
            # Skip if not enough coordinates
            if len(coords) < 2:
                continue
                
            # Add nodes and edges to the graph
            for i in range(len(coords) - 1):
                # Use coordinates as node identifiers
                u = f"{coords[i][0]:.6f},{coords[i][1]:.6f}"
                v = f"{coords[i+1][0]:.6f},{coords[i+1][1]:.6f}"
                
                # Add nodes with their coordinates
                G.add_node(u, x=coords[i][0], y=coords[i][1])
                G.add_node(v, x=coords[i+1][0], y=coords[i+1][1])
                
                # Calculate segment length
                segment_length = Point(coords[i]).distance(Point(coords[i+1]))
                
                # Add edge with attributes
                G.add_edge(u, v, 
                           length=segment_length,
                           highway=row.get('highway', 'unknown'),
                           maxspeed=row.get('maxspeed', 50),  # Default 50 km/h if unknown
                           id=idx)
                           
        except (AttributeError, IndexError, ValueError) as e:
            # Skip problematic geometries
            continue
            
    return G

@st.cache_data
def find_shortest_path(_G, start_point, end_point):
    """Find the shortest path between two points in the road network"""
    if _G is None or len(_G.nodes) == 0:
        return None, None, None
        
    # Find the nearest nodes to the start and end points
    start_node = find_nearest_node(_G, start_point)
    end_node = find_nearest_node(_G, end_point)
    
    if start_node is None or end_node is None:
        return None, None, None
        
    try:
        # Find the shortest path
        path = nx.shortest_path(_G, start_node, end_node, weight='length')
        
        # Extract coordinates for the path
        path_coords = [(float(_G.nodes[node]['x']), float(_G.nodes[node]['y'])) for node in path]
        
        # Create a LineString geometry from the path
        path_geom = LineString(path_coords)
        
        # Calculate path length and travel time
        path_length = sum(float(_G[path[i]][path[i+1]]['length']) for i in range(len(path)-1))
        
        # Extract path edges for highlighting
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        return path_geom, path_length, path_edges
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None, None

def find_nearest_node(_G, point):
    """Find the nearest node to a given point"""
    if len(_G.nodes) == 0:
        return None
        
    # Calculate distances to all nodes
    distances = []
    for node in _G.nodes:
        x = float(_G.nodes[node]['x'])
        y = float(_G.nodes[node]['y'])
        distance = ((x - point[0])**2 + (y - point[1])**2)**0.5
        distances.append((node, distance))
    
    # Find the nearest node
    nearest_node = min(distances, key=lambda x: x[1])[0]
    return nearest_node

def generate_pdf_report(city, analysis_type, charts, maps):
    """Generate a PDF report with charts and maps"""
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.5, f"{city} - {analysis_type} Analysis", 
                 ha='center', va='center', fontsize=24)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add each chart to the PDF
        for chart in charts:
            pdf.savefig(chart)
            
        # Add maps (convert to matplotlib figures)
        for map_img in maps:
            fig = plt.figure(figsize=(8.5, 11))
            plt.imshow(map_img)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
    
    buffer.seek(0)
    return buffer

# Load data
gdf_baguio, gdf_zurich = load_data()

# ----------------------
# Sidebar Filters
# ----------------------
st.sidebar.title("Controls")
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Single City Analysis", "City Comparison", "Route Finder"]
)

if app_mode == "Single City Analysis":
    cities = {"Baguio": gdf_baguio, "Zurich": gdf_zurich}
    city = st.sidebar.selectbox("Select City", list(cities.keys()))
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type", 
        ["Road Hierarchy", "Road Curvature", "Intersection Density", "Road Statistics"]
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
    # Main Layout for Single City
    # ----------------------
    st.title("🚦 City Road Network Explorer")
    st.markdown(f"### Analyzing **{analysis_type}** in **{city}**")
    
    # Download options
    download_format = st.sidebar.selectbox(
        "Download Report Format",
        ["None", "PDF", "PNG"]
    )
    
    if download_format != "None":
        generate_report = st.sidebar.button("Generate Report")
    
    # Visualizations based on analysis type
    if analysis_type in ["Road Hierarchy", "Road Curvature", "Intersection Density"]:
        # Use existing visualization code
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
                    st.warning("Could not calculate intersection density. This could be due to:")
                    st.markdown("""
                    - Not enough road segments in the selected area
                    - No intersections found in the selected road types
                    - Try selecting more road types or switching to a different area
                    """)

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

    elif analysis_type == "Road Statistics":
        # Create detailed statistics view
        st.subheader(f"Road Statistics for {city}")
        
        # Overall stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_length = gdf['length_m'].sum()
            st.metric("Total Road Length", f"{total_length/1000:.2f} km")
            
        with col2:
            avg_length = gdf['length_m'].mean()
            st.metric("Avg. Road Segment Length", f"{avg_length:.2f} m")
            
        with col3:
            total_segments = len(gdf)
            st.metric("Total Road Segments", f"{total_segments}")
        
        # Detailed stats by road type
        st.subheader("Detailed Statistics by Road Type")
        
        # Group by road type
        road_stats = gdf.groupby('highway').agg({
            'length_m': ['count', 'sum', 'mean', 'std'],
            'curvature': ['mean', 'median', 'max']
        }).reset_index()
        
        # Rename columns for clarity
        road_stats.columns = ['Road Type', 'Segments', 'Total Length (m)', 'Avg Length (m)', 
                             'Std Dev Length (m)', 'Avg Curvature', 'Median Curvature', 'Max Curvature']
        
        # Convert total length to km
        road_stats['Total Length (km)'] = road_stats['Total Length (m)'] / 1000
        road_stats = road_stats.drop(columns=['Total Length (m)'])
        
        # Round values for display
        for col in road_stats.columns:
            if col != 'Road Type':
                road_stats[col] = road_stats[col].round(2)
                
        # Display as table
        st.dataframe(road_stats, use_container_width=True)
        
        # Visualizations for road stats
        col1, col2 = st.columns(2)
        
        with col1:
            # Total length by road type
            st.subheader("Total Length by Road Type")
            fig, ax = plt.subplots(figsize=(8, 5))
            road_type_lengths = gdf.groupby('highway')['length_m'].sum() / 1000  # km
            road_type_lengths.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_xlabel("Total Length (km)")
            ax.set_ylabel("Road Type")
            st.pyplot(fig)
            
        with col2:
            # Average length by road type
            st.subheader("Average Segment Length by Road Type")
            fig, ax = plt.subplots(figsize=(8, 5))
            road_type_avg_length = gdf.groupby('highway')['length_m'].mean()
            road_type_avg_length.sort_values(ascending=True).plot(kind='barh', ax=ax, color='lightgreen')
            ax.set_xlabel("Average Length (m)")
            ax.set_ylabel("Road Type")
            st.pyplot(fig)
            
        # Additional stats if available
        if 'lanes' in gdf.columns and not gdf['lanes'].isna().all():
            st.subheader("Lane Distribution")
            
            # Count lanes
            lane_counts = gdf['lanes'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            lane_counts.plot(kind='bar', ax=ax, color='orange')
            ax.set_xlabel("Number of Lanes")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
        if 'maxspeed' in gdf.columns and not gdf['maxspeed'].isna().all():
            st.subheader("Speed Limit Distribution")
            
            # Group speed limits into bins
            speed_bins = pd.cut(gdf['maxspeed'], bins=[0, 30, 50, 70, 90, 130], 
                               labels=['≤30', '31-50', '51-70', '71-90', '>90'])
            speed_counts = speed_bins.value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            speed_counts.plot(kind='bar', ax=ax, color='tomato')
            ax.set_xlabel("Speed Limit (km/h)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
    
    # Report generation
    if download_format != "None" and 'generate_report' in locals() and generate_report:
        # Capture all figures for report
        charts = []
        map_images = []
        
        # For PNG, use streamlit download button
        if download_format == "PNG":
            # Save the main figure
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            btn = st.download_button(
                label="Download Map as PNG",
                data=buffer,
                file_name=f"{city}_{analysis_type.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
            
        # For PDF, generate a full report
        elif download_format == "PDF":
            with st.spinner("Generating PDF report..."):
                # Get all figures from matplotlib
                figs = [plt.figure(n) for n in plt.get_fignums()]
                
                # Generate PDF
                pdf_buffer = generate_pdf_report(city, analysis_type, figs, map_images)
                
                # Create download button
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"{city}_{analysis_type.lower().replace(' ', '_')}_report.pdf",
                    mime="application/pdf"
                )

elif app_mode == "City Comparison":
    st.title("🏙️ City Comparison Dashboard")
    st.markdown("Compare road networks between cities to identify differences and similarities.")
    
    # Select comparison metric
    comparison_metric = st.selectbox(
        "Select Comparison Metric",
        ["Road Type Distribution", "Curvature Comparison", "Road Length Comparison", "Intersection Density"]
    )
    
    # Filter options
    road_type_filter = st.multiselect(
        "Road Types to Include", 
        options=["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street"], 
        default=["primary", "secondary", "tertiary"]
    )
    
    # Apply filters to both cities
    filtered_baguio = gdf_baguio[gdf_baguio['highway'].isin(road_type_filter)] if road_type_filter else gdf_baguio
    filtered_zurich = gdf_zurich[gdf_zurich['highway'].isin(road_type_filter)] if road_type_filter else gdf_zurich
    
    # Check if data is available
    if filtered_baguio.empty or filtered_zurich.empty:
        st.warning("⚠️ No roads match the selected filters for one or both cities.")
        st.stop()
    
    # Display comparison based on selected metric
    if comparison_metric == "Road Type Distribution":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baguio Road Types")
            baguio_road_types = filtered_baguio['highway'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            baguio_road_types.plot.pie(ax=ax1, autopct='%1.1f%%', startangle=90)
            ax1.set_ylabel('')
            st.pyplot(fig1)
            
            # Show stats
            total_roads_baguio = len(filtered_baguio)
            st.metric("Total Road Segments", f"{total_roads_baguio}")
            
        with col2:
            st.subheader("Zurich Road Types")
            zurich_road_types = filtered_zurich['highway'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            zurich_road_types.plot.pie(ax=ax2, autopct='%1.1f%%', startangle=90)
            ax2.set_ylabel('')
            st.pyplot(fig2)
            
            # Show stats
            total_roads_zurich = len(filtered_zurich)
            st.metric("Total Road Segments", f"{total_roads_zurich}")
            
        # Normalized comparison bar chart
        st.subheader("Normalized Road Type Distribution Comparison")
        
        # Combine data and normalize
        baguio_pct = baguio_road_types / baguio_road_types.sum() * 100
        zurich_pct = zurich_road_types / zurich_road_types.sum() * 100
        
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame({
            'Baguio': baguio_pct,
            'Zurich': zurich_pct
        }).fillna(0)
        
        # Plot comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax3)
        ax3.set_xlabel('Road Type')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Road Type Distribution Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
    elif comparison_metric == "Curvature Comparison":
        # Calculate curvature stats for both cities
        baguio_curvature = filtered_baguio['curvature'].mean()
        zurich_curvature = filtered_zurich['curvature'].mean()
        
        # Display curvature comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baguio Curvature")
            st.metric("Average Curvature", f"{baguio_curvature:.3f}")
            
            # Histogram
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            filtered_baguio['curvature'].hist(ax=ax1, bins=20, alpha=0.7)
            ax1.set_xlabel('Curvature')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Baguio Road Curvature Distribution')
            st.pyplot(fig1)
            
            # Curvature by road type
            baguio_curve_by_type = filtered_baguio.groupby('highway')['curvature'].mean().sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            baguio_curve_by_type.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_xlabel('Road Type')
            ax3.set_ylabel('Average Curvature')
            ax3.set_title('Baguio Curvature by Road Type')
            plt.xticks(rotation=45)
            st.pyplot(fig3)
            
        with col2:
            st.subheader("Zurich Curvature")
            st.metric("Average Curvature", f"{zurich_curvature:.3f}")
            
            # Histogram
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            filtered_zurich['curvature'].hist(ax=ax2, bins=20, alpha=0.7)
            ax2.set_xlabel('Curvature')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Zurich Road Curvature Distribution')
            st.pyplot(fig2)
            
            # Curvature by road type
            zurich_curve_by_type = filtered_zurich.groupby('highway')['curvature'].mean().sort_values(ascending=False)
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            zurich_curve_by_type.plot(kind='bar', ax=ax4, color='lightgreen')
            ax4.set_xlabel('Road Type')
            ax4.set_ylabel('Average Curvature')
            ax4.set_title('Zurich Curvature by Road Type')
            plt.xticks(rotation=45)
            st.pyplot(fig4)
            
        # Overall comparison bar chart
        st.subheader("City Curvature Comparison")
        
        # Create comparison dataframe
        city_comparison = pd.DataFrame({
            'City': ['Baguio', 'Zurich'],
            'Average Curvature': [baguio_curvature, zurich_curvature]
        })
        
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        sns_colors = ['#1f77b4', '#2ca02c']  # Blue and green
        ax5.bar(city_comparison['City'], city_comparison['Average Curvature'], color=sns_colors)
        ax5.set_xlabel('City')
        ax5.set_ylabel('Average Curvature')
        ax5.set_title('Average Road Curvature by City')
        st.pyplot(fig5)
        
    elif comparison_metric == "Road Length Comparison":
        # Calculate length statistics
        baguio_total_length = filtered_baguio['length_m'].sum() / 1000  # km
        zurich_total_length = filtered_zurich['length_m'].sum() / 1000  # km
        
        baguio_avg_length = filtered_baguio['length_m'].mean()
        zurich_avg_length = filtered_zurich['length_m'].mean()
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baguio Road Length")
            st.metric("Total Road Length", f"{baguio_total_length:.2f} km")
            st.metric("Avg. Segment Length", f"{baguio_avg_length:.2f} m")
            
            # Length distribution
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            filtered_baguio['length_m'].hist(ax=ax1, bins=20, alpha=0.7)
            ax1.set_xlabel('Length (m)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Baguio Road Length Distribution')
            st.pyplot(fig1)
            
            # Length by road type
            baguio_length_by_type = filtered_baguio.groupby('highway')['length_m'].sum() / 1000  # km
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            baguio_length_by_type.sort_values(ascending=False).plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_xlabel('Road Type')
            ax3.set_ylabel('Total Length (km)')
            ax3.set_title('Baguio Total Length by Road Type')
            plt.xticks(rotation=45)
            st.pyplot(fig3)
            
        with col2:
            st.subheader("Zurich Road Length")
            st.metric("Total Road Length", f"{zurich_total_length:.2f} km")
            st.metric("Avg. Segment Length", f"{zurich_avg_length:.2f} m")
            
            # Length distribution
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            filtered_zurich['length_m'].hist(ax=ax2, bins=20, alpha=0.7)
            ax2.set_xlabel('Length (m)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Zurich Road Length Distribution')
            st.pyplot(fig2)
            
            # Length by road type
            zurich_length_by_type = filtered_zurich.groupby('highway')['length_m'].sum() / 1000  # km
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            zurich_length_by_type.sort_values(ascending=False).plot(kind='bar', ax=ax4, color='lightgreen')
            ax4.set_xlabel('Road Type')
            ax4.set_ylabel('Total Length (km)')
            ax4.set_title('Zurich Total Length by Road Type')
            plt.xticks(rotation=45)
            st.pyplot(fig4)
            
        # Overall comparison
        st.subheader("City Road Length Comparison")
        
        # Create comparison dataframe
        length_comparison = pd.DataFrame({
            'City': ['Baguio', 'Zurich'],
            'Total Length (km)': [baguio_total_length, zurich_total_length],
            'Avg. Segment Length (m)': [baguio_avg_length, zurich_avg_length]
        })
        
        # Plot total length comparison
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        ax5.bar(length_comparison['City'], length_comparison['Total Length (km)'])
        ax5.set_xlabel('City')
        ax5.set_ylabel('Total Road Length (km)')
        ax5.set_title('Total Road Network Length by City')
        st.pyplot(fig5)
        
        # Plot avg length comparison
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        ax6.bar(length_comparison['City'], length_comparison['Avg. Segment Length (m)'])
        ax6.set_xlabel('City')
        ax6.set_ylabel('Average Segment Length (m)')
        ax6.set_title('Average Road Segment Length by City')
        st.pyplot(fig6)
        
    elif comparison_metric == "Intersection Density":
        st.info("Calculating intersection density for both cities. This may take a moment...")
        
        # Calculate intersection density for both cities
        baguio_density = calculate_intersection_density(filtered_baguio, "Baguio")
        zurich_density = calculate_intersection_density(filtered_zurich, "Zurich")
        
        # Check if calculations were successful
        if baguio_density.empty or zurich_density.empty:
            st.warning("Could not calculate intersection density for one or both cities.")
            st.stop()
            
        # Calculate density statistics
        baguio_mean_density = baguio_density['intersection_density'].mean()
        zurich_mean_density = zurich_density['intersection_density'].mean()
        
        baguio_total_intersections = baguio_density['intersection_count'].sum()
        zurich_total_intersections = zurich_density['intersection_count'].sum()
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baguio Intersection Density")
            st.metric("Average Density (per sq km)", f"{baguio_mean_density:.2f}")
            st.metric("Total Intersections", f"{baguio_total_intersections}")
            
            # Create map
            m1 = folium.Map(
                location=[filtered_baguio.geometry.centroid.y.mean(), filtered_baguio.geometry.centroid.x.mean()],
                zoom_start=13,
                tiles="CartoDB positron"
            )
            
            # Add choropleth if we have data
            if not baguio_density.empty:
                folium.Choropleth(
                    geo_data=baguio_density.__geo_interface__,
                    data=baguio_density['intersection_density'],
                    key_on='feature.id',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Intersection Density (per sq km)'
                ).add_to(m1)
                
                # Add road network
                folium.GeoJson(
                    filtered_baguio.__geo_interface__,
                    style_function=lambda x: {
                        'color': 'blue',
                        'weight': 2,
                        'opacity': 0.5
                    }
                ).add_to(m1)
                
            folium_static(m1)
            
        with col2:
            st.subheader("Zurich Intersection Density")
            st.metric("Average Density (per sq km)", f"{zurich_mean_density:.2f}")
            st.metric("Total Intersections", f"{zurich_total_intersections}")
            
            # Create map
            m2 = folium.Map(
                location=[filtered_zurich.geometry.centroid.y.mean(), filtered_zurich.geometry.centroid.x.mean()],
                zoom_start=13,
                tiles="CartoDB positron"
            )
            
            # Add choropleth if we have data
            if not zurich_density.empty:
                folium.Choropleth(
                    geo_data=zurich_density.__geo_interface__,
                    data=zurich_density['intersection_density'],
                    key_on='feature.id',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Intersection Density (per sq km)'
                ).add_to(m2)
                
                # Add road network
                folium.GeoJson(
                    filtered_zurich.__geo_interface__,
                    style_function=lambda x: {
                        'color': 'blue',
                        'weight': 2,
                        'opacity': 0.5
                    }
                ).add_to(m2)
                
            folium_static(m2)
            
        # Overall comparison
        st.subheader("Intersection Density Comparison")
        
        # Create comparison dataframe
        density_comparison = pd.DataFrame({
            'City': ['Baguio', 'Zurich'],
            'Average Density (per sq km)': [baguio_mean_density, zurich_mean_density],
            'Total Intersections': [baguio_total_intersections, zurich_total_intersections]
        })
        
        # Plot density comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(density_comparison['City'], density_comparison['Average Density (per sq km)'])
        ax.set_xlabel('City')
        ax.set_ylabel('Average Intersection Density (per sq km)')
        ax.set_title('Average Intersection Density by City')
        st.pyplot(fig)

elif app_mode == "Route Finder":
    st.title("🔍 Road Network Route Finder")
    st.markdown("Find the shortest path between two points in the road network.")
    
    # City selection
    city = st.selectbox("Select City", ["Baguio", "Zurich"])
    
    # Get data for selected city
    gdf = gdf_baguio if city == "Baguio" else gdf_zurich
    
    # Filter road types
    road_type_filter = st.multiselect(
        "Road Types to Include", 
        options=["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street"], 
        default=["primary", "secondary", "tertiary", "residential"]
    )
    
    # Apply filters
    if road_type_filter:
        gdf = gdf[gdf['highway'].isin(road_type_filter)]
        if gdf.empty:
            st.warning("⚠️ No roads match the selected filters.")
            st.stop()
    
    # Create network graph
    with st.spinner("Building road network graph..."):
        G = create_road_network(gdf)
        
    # Map for selecting points
    st.subheader("Select Start and End Points")
    st.markdown("Click on the map to select the start and end points for routing.")
    
    # Create base map
    m = folium.Map(
        location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
        zoom_start=13,
        tiles="CartoDB positron"
    )
    
    # Add road network
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda x: {
            'color': 'blue',
            'weight': 2,
            'opacity': 0.5
        }
    ).add_to(m)
    
    # Create a map with points selection capability
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use Streamlit's implementation to render the map
        map_data = folium_static(m)
        
    with col2:
        st.write("Instructions:")
        st.markdown("""
        1. Click on the map to select your starting point
        2. Click again to select your destination
        3. Press 'Find Route' to calculate the shortest path
        """)
        
        # Since we can't directly capture clicks with Streamlit, provide input boxes
        st.subheader("Start Point")
        start_lat = st.number_input("Start Latitude", 
                                    value=gdf.geometry.centroid.y.mean(),
                                    format="%.6f")
        start_lon = st.number_input("Start Longitude", 
                                    value=gdf.geometry.centroid.x.mean(),
                                    format="%.6f")
        
        st.subheader("End Point")
        end_lat = st.number_input("End Latitude", 
                                  value=gdf.geometry.centroid.y.mean() + 0.01,
                                  format="%.6f")
        end_lon = st.number_input("End Longitude", 
                                  value=gdf.geometry.centroid.x.mean() + 0.01,
                                  format="%.6f")
        
        # Button to find route
        find_route = st.button("Find Route")
    
    # Find and display route
    if find_route:
        with st.spinner("Finding shortest path..."):
            # Get start and end points
            start_point = (start_lon, start_lat)
            end_point = (end_lon, end_lat)
            
            # Find shortest path
            path_geom, path_length, path_edges = find_shortest_path(G, start_point, end_point)
            
            if path_geom is not None:
                # Display results
                st.subheader("Route Results")
                
                # Display route length
                st.metric("Route Length", f"{path_length/1000:.2f} km")
                
                # Estimate travel time (assuming average speed of 30 km/h for urban areas)
                travel_time_min = (path_length / 1000) / 30 * 60
                st.metric("Estimated Travel Time", f"{travel_time_min:.1f} minutes")
                
                # Create new map with route
                route_map = folium.Map(
                    location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
                    zoom_start=13,
                    tiles="CartoDB positron"
                )
                
                # Add road network (background)
                folium.GeoJson(
                    gdf.__geo_interface__,
                    style_function=lambda x: {
                        'color': 'gray',
                        'weight': 1,
                        'opacity': 0.5
                    }
                ).add_to(route_map)
                
                # Add route
                folium.GeoJson(
                    {"type": "Feature", "geometry": path_geom.__geo_interface__},
                    style_function=lambda x: {
                        'color': 'red',
                        'weight': 4,
                        'opacity': 0.8
                    }
                ).add_to(route_map)
                
                # Add markers for start and end
                folium.Marker(
                    location=[start_lat, start_lon],
                    popup="Start",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(route_map)
                
                folium.Marker(
                    location=[end_lat, end_lon],
                    popup="End",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(route_map)
                
                # Display map
                folium_static(route_map)
                
                # Add download option
                if st.button("Export Route"):
                    # Create GeoDataFrame from path
                    route_gdf = gpd.GeoDataFrame({'geometry': [path_geom]}, crs="EPSG:4326")
                    
                    # Save to GeoJSON
                    route_geojson = route_gdf.to_json()
                    
                    # Create download button
                    st.download_button(
                        label="Download Route as GeoJSON",
                        data=route_geojson,
                        file_name=f"{city}_route.geojson",
                        mime="application/json"
                    )
            else:
                st.error("Could not find a route between the selected points. Try points closer to the road network.")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Made with 💚 using OpenStreetMap + Streamlit + GeoPandas")

