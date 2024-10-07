import pandas as pd
import folium

# Load the CSV file
csv_file = 'predictions_rain_2025.csv'  # Path to your CSV file
data = pd.read_csv(csv_file)

# Create a Folium map centered around the middle of the data points
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
map = folium.Map(location=map_center, zoom_start=6)

# Function to determine color based on precipitation
def get_color(precipitation):
    if precipitation > 20:
        return 'blue'
    elif precipitation > 10:
        return 'green'
    else:
        return 'red'

# Add circle markers to the map
for index, row in data.iterrows():
    # Create a popup with Latitude, Longitude, and Precipitation
    popup_content = f"Latitude: {row['Latitude']} \n Longitude: {row['Longitude']} \n Precipitation: {row['Precipitation']} mm"
    
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2 + row['Precipitation'] * 0.2,  # Circle size proportional to precipitation
        color=get_color(row['Precipitation']),
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_content, parse_html=True)  # Use parse_html to allow HTML in popup
    ).add_to(map)

# Save the map to an HTML file
map.save('precipitation_map.html')

# If you are in a Jupyter Notebook, use this to display the map
# map
