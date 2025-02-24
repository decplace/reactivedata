import dash
import os
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import geopandas as gpd
import pandas as pd
import json
import re
import math
import numpy as np

# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "regionalreactive.csv")

map_path = os.path.join(BASE_DIR, "ons_regions/RGN_DEC_24_EN_BFE.shp")

# --- Helper function: map postcode to region ---
def map_postcode_to_region(postcode):
    postcode = postcode.strip().upper()
    match = re.match(r'([A-Z]+)', postcode)
    if not match:
        return None
    area = match.group(1)
    mapping = {
        # London
        'EC': 'London', 'WC': 'London', 'N': 'London', 'NW': 'London', 
        'E': 'London', 'SE': 'London', 'SW': 'London', 'W': 'London',
        # South East
        'BR': 'South East', 'CR': 'South East', 'CT': 'South East',
        'GU': 'South East', 'RH': 'South East', 'SL': 'South East',
        'TN': 'South East', 'HP': 'South East', 'KT': 'South East',
        'BN': 'South East',
        # South West
        'BA': 'South West', 'BH': 'South West', 'BS': 'South West',
        'DT': 'South West', 'EX': 'South West', 'PL': 'South West',
        'TQ': 'South West', 'TR': 'South West',
        # East of England
        'CB': 'East of England', 'CM': 'East of England', 'CO': 'East of England',
        'IP': 'East of England', 'NR': 'East of England', 'SG': 'East of England',
        'SS': 'East of England',
        # West Midlands
        'B': 'West Midlands', 'CV': 'West Midlands', 'DY': 'West Midlands',
        'HR': 'West Midlands', 'WS': 'West Midlands', 'WR': 'West Midlands',
        # East Midlands
        'DE': 'East Midlands', 'LE': 'East Midlands', 'LN': 'East Midlands',
        'NG': 'East Midlands', 'NN': 'East Midlands',
        # North West
        'BL': 'North West', 'CA': 'North West', 'CH': 'North West',
        'FY': 'North West', 'L': 'North West', 'LA': 'North West',
        'M': 'North West', 'OL': 'North West', 'PR': 'North West',
        'WA': 'North West', 'WN': 'North West',
        # North East
        'NE': 'North East', 'SR': 'North East', 'TS': 'North East',
        # Yorkshire and The Humber
        'BD': 'Yorkshire and The Humber', 'DN': 'Yorkshire and The Humber',
        'HD': 'Yorkshire and The Humber', 'HG': 'Yorkshire and The Humber',
        'LS': 'Yorkshire and The Humber', 'S': 'Yorkshire and The Humber',
        'WF': 'Yorkshire and The Humber', 'YO': 'Yorkshire and The Humber'
    }
    for key, region in mapping.items():
        if area.startswith(key):
            return region
    return None

# --- Load the ONS Regions shapefile and convert to GeoJSON ---
shapefile_path = map_path
regions_gdf = gpd.read_file(shapefile_path)
regions_gdf["region_name"] = regions_gdf["RGN24NM"]

# Reproject from EPSG:27700 to EPSG:4326
regions_gdf = regions_gdf.to_crs(epsg=4326)
geojson_data = json.loads(regions_gdf.to_json())

# --- Load reactive CSV data and aggregate ---
df = pd.read_csv(csv_path)
df["Region"] = df["Postcode"].apply(map_postcode_to_region)
df = df[df["Region"].notnull()]
df["TotalTasks"] = pd.to_numeric(df["TotalTasks"], errors="coerce").fillna(0)

# Aggregate by Region, Category, and Supplier
agg = df.groupby(["Region", "Category", "SupplierName"], as_index=False)["TotalTasks"].sum()
agg["TotalVolume"] = agg.groupby(["Region", "Category"])["TotalTasks"].transform("sum")
agg["Percentage"] = agg["TotalTasks"] / agg["TotalVolume"] * 100

# --- Build the Dash app layout ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Supplier Volume Dashboard"),
    html.Div([
        html.Label("Select Category:"),
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": cat, "value": cat} for cat in sorted(agg["Category"].unique())],
            value=sorted(agg["Category"].unique())[0],
            clearable=False,
            style={"width": "300px"}
        )
    ]),
    html.Div([
        dcc.Graph(id="map-graph", style={"width": "60%", "display": "inline-block"}),
        dcc.Graph(id="pie-chart", style={"width": "38%", "display": "inline-block", "vertical-align": "top"})
    ])
])

# --- Callback to update the map based on the selected category ---
@app.callback(
    Output("map-graph", "figure"),
    [Input("category-dropdown", "value")]
)
def update_map(selected_category):
    # Filter the aggregated data for the selected category
    df_cat = agg[agg["Category"] == selected_category]
    
    # Compute the dominant supplier (by Percentage) for each region (for marker coloring)
    region_info = df_cat.groupby("Region").apply(lambda x: x.sort_values("Percentage", ascending=False).iloc[0]).reset_index(drop=True)
    
    # Get centroids from the regions GeoDataFrame
    centroids = regions_gdf.copy()
    centroids["centroid_lon"] = centroids.geometry.centroid.x
    centroids["centroid_lat"] = centroids.geometry.centroid.y
    
    # Merge region_info with centroids
    merged = centroids.merge(region_info, left_on="region_name", right_on="Region", how="left")
    
    # Build the map figure without drawing the heavy boundaries.
    fig = go.Figure()

    # Add markers at the centroids (clickable) with region names
    fig.add_trace(go.Scattergeo(
        lon=merged["centroid_lon"],
        lat=merged["centroid_lat"],
        mode="markers+text",
        marker=dict(size=12, color="red"),
        text=merged["region_name"],
        hoverinfo="text",
        name="Region"
    ))
    
    # Update geos to auto-fit the marker locations and provide a wider bounding box.
    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="lightgray",
        fitbounds="locations"
    )
    fig.update_layout(
        margin={"r":0, "t":40, "l":0, "b":0},
        clickmode="event+select",
        title=f"ONS Regions - {selected_category}",
        height=600
    )
    return fig

# --- Callback to update the pie chart when a region is clicked ---
@app.callback(
    Output("pie-chart", "figure"),
    [Input("map-graph", "clickData"),
     Input("category-dropdown", "value")]
)
def update_pie_chart(clickData, selected_category):
    if clickData is None:
        # If no region is clicked, show an empty figure with a message.
        return go.Figure(data=[go.Pie(labels=["No region selected"],
                                       values=[1],
                                       textinfo="none")],
                        layout=go.Layout(title="Click on a region in the map"))
    # Get the region name from the clicked marker (stored in the "text" field)
    region = clickData["points"][0].get("text", None)
    if region is None:
        return go.Figure()
    
    # Filter aggregated data for the selected region and category, and limit to top 6 suppliers.
    df_region = agg[(agg["Category"] == selected_category) & (agg["Region"] == region)]
    df_region = df_region.sort_values("Percentage", ascending=False).head(6)
    
    # Create a standard pie chart for the selected region.
    fig = go.Figure(data=[go.Pie(
        labels=df_region["SupplierName"],
        values=df_region["TotalTasks"],
        text=[f"{perc:.0f}%" for perc in df_region["Percentage"]],
        textinfo="label+percent",
        hole=0.3
    )])
    fig.update_layout(title=f"Supplier Split for {region} - {selected_category}")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)