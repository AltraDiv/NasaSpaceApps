import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from pyhdf.SD import SD, SDC
import netCDF4 as nc
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic  # Optional for precise distance filtering
import folium

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants for MODIS Sinusoidal projection
EARTH_RADIUS = 6371007.181  # Radius of Earth in meters
TILE_SIZE = 1200  # Number of pixels in each tile (1200 x 1200 for MOD11A1)
PIXEL_SIZE = 1000  # Size of each pixel in meters (1km resolution)

# Define the grid dimensions
h, v = 12, 4  # Tile h12v04
t_width = 1200  # Width of the tile in pixels

# Functions to calculate latitude and longitude from MODIS Sinusoidal coordinates
def modis_sinusoidal_to_lon(h_tile, i):
    return (h_tile - 18) * TILE_SIZE * PIXEL_SIZE / EARTH_RADIUS * (180 / np.pi) + (i - t_width // 2) * PIXEL_SIZE / EARTH_RADIUS * (180 / np.pi)

def modis_sinusoidal_to_lat(v_tile, j):
    return (v_tile - 9) * TILE_SIZE * PIXEL_SIZE / EARTH_RADIUS * (180 / np.pi) - (j - TILE_SIZE // 2) * PIXEL_SIZE / EARTH_RADIUS * (180 / np.pi)

# Custom Dataset class for DataLoader
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Define the Weather Predictor model
class WeatherPredictor(nn.Module):
    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    date_input = data.get('date')  # e.g., "2023091"
    data_type = data.get('data_type')  # e.g., "temp", "rain", "groundwater"
    epochs = data.get('epochs', 20)  # Default to 10 epochs if not provided
    batch_size = data.get('batch_size', 32)  # Default batch size
    learning_rate = data.get('learning_rate', 0.001)  # Default learning rate
    overwrite = data.get('overwrite', False)  # Whether to overwrite existing model

    # Format date_input
    date_input = f"2023{date_input[4:]}" 
    date_input = date_input.replace("-", "")  # Removes all dashes
    print(f"Received request to train {data_type} model on {date_input}.")

    # Define target variables and model paths
    data_info = {
        'temp': {
            'target': 'Temperature',
            'model_path': f'./models/weather_predictor_temp_model_{date_input}.pth',
            'dataset_name': 'LST_Day_1km'  # Replace with actual dataset name if different
        },
        'rain': {
            'target': 'Precipitation',
            'model_path': f'./models/weather_predictor_rain_model_{date_input}.pth',
            'dataset_name': 'precipitation'  # Replace with actual dataset name if different
        },
        'groundwater': {
            'target': 'Groundwater',
            'model_path': f'./models/weather_predictor_groundwater_model_{date_input}.pth',
            'dataset_name': 'Optical_Depth_047'  # Replace with actual dataset name if different
        }
    }

    if data_type not in data_info:
        return jsonify({"error": "Invalid data type. Choose 'temp', 'rain', or 'groundwater'."}), 400

    target_variable = data_info[data_type]['target']
    model_path = data_info[data_type]['model_path']
    dataset_name = data_info[data_type]['dataset_name']

    # Check if model already exists
    if os.path.exists(model_path) and not overwrite:
        return jsonify({"message": f"Model for {data_type} on {date_input} already exists. Set 'overwrite' to true to retrain."}), 200

    # File paths based on the input date
    file_paths = {
        'temp': f'temp/*.A{date_input}.*.hdf',
        'rain': f'rain/*.{date_input}*.nc4',
        'groundwater': f'groundwater/*.A{date_input}.*.hdf'
    }

    # Initialize DataFrame for model training
    data_list = []

    # ------------------- Data Loading -------------------
    if data_type == "temp":
        # Load temperature data
        if (file_paths['temp'][14] == '0'):
            print(file_paths['temp'])
            file_paths['temp'] = file_paths['temp'][0:14] + file_paths['temp'][15:]
        print(file_paths['temp'])
        hdf_lst_files = glob.glob(file_paths['temp'])
        if not hdf_lst_files:
            return jsonify({"error": "No land surface temperature files found for the specified date."}), 404
        else:
            hdf_lst = SD(hdf_lst_files[0], SDC.READ)
            lst_data = hdf_lst.select(dataset_name)
            lst_data_array = lst_data[:]

            nrows, ncols = lst_data_array.shape
            lats = np.zeros((nrows, ncols))
            lons = np.zeros((nrows, ncols))

            for i in range(nrows):
                for j in range(ncols):
                    lats[i, j] = -1 * modis_sinusoidal_to_lat(v, i)
                    lons[i, j] =  modis_sinusoidal_to_lon(h, j)
            print(lats[0,0])
            print(lons[0,0])
            print(lats[12,12])
            print(lons[12,12])
            lat_flat = lats.flatten()
            lon_flat = lons.flatten()
            temp_flat = lst_data_array.flatten()

            lst_df = pd.DataFrame({
                'Latitude': lat_flat,
                'Longitude': lon_flat,
                'Temperature': temp_flat
            })
            print(lats[0,0])
            print(lons[0,0])
            print(lats[12,12])
            print(lons[12,12])
            data_list.append(lst_df)

    elif data_type == "rain":
        print(file_paths['rain'])
        # Load precipitation data
        rain_files = glob.glob(file_paths['rain'])
        if not rain_files:
            return jsonify({"error": "No precipitation files found for the specified date."}), 404
        else:
            with nc.Dataset(rain_files[0], 'r') as ds:
                precipitation_data = ds.variables['precipitation'][:]
                lat = ds.variables['lat'][:]
                lon = ds.variables['lon'][:]
                print(lat)
                print(lon)
                precipitation_data = precipitation_data[0, :, :]

                precipitation_data = np.ma.masked_invalid(precipitation_data)
                precip_latitude = np.repeat(lat, len(lon)).reshape(precipitation_data.shape)
                precip_longitude = np.tile(lon, len(lat)).reshape(precipitation_data.shape)

                precip_df = pd.DataFrame({
                    'Latitude': precip_latitude.flatten(),
                    'Longitude': precip_longitude.flatten(),
                    'Precipitation': precipitation_data.flatten()
                })

                data_list.append(precip_df)

    elif data_type == "groundwater":
        # Load groundwater data
        hdf_groundwater_files = glob.glob(file_paths['groundwater'])
        if not hdf_groundwater_files:
            return jsonify({"error": "No groundwater files found for the specified date."}), 404
        else:
            hdf_groundwater = SD(hdf_groundwater_files[0], SDC.READ)
            groundwater_data = hdf_groundwater.select(dataset_name)
            groundwater_data_array = groundwater_data[:]

            if groundwater_data_array.ndim == 3:
                n_layers, nrows_gw, ncols_gw = groundwater_data_array.shape
                gw_lats = np.zeros((n_layers, nrows_gw, ncols_gw))
                gw_lons = np.zeros((n_layers, nrows_gw, ncols_gw))

                for layer in range(n_layers):
                    for i in range(nrows_gw):
                        for j in range(ncols_gw):
                            gw_lats[layer, i, j] = -1 * modis_sinusoidal_to_lat(v, i)
                            gw_lons[layer, i, j] = modis_sinusoidal_to_lon(h, j)

                gw_lat_flat = gw_lats.flatten()
                gw_lon_flat = gw_lons.flatten()
                gw_flat = groundwater_data_array.flatten()

                groundwater_df = pd.DataFrame({
                    'Latitude': gw_lat_flat,
                    'Longitude': gw_lon_flat,
                    'Groundwater': gw_flat
                })

                data_list.append(groundwater_df)
            else:
                return jsonify({"error": "Unexpected data dimensions for groundwater data."}), 500

    # Combine all data for model training
    combined_df = pd.concat(data_list, ignore_index=True)

    # ------------------- Filtering Data -------------------
    # Define center coordinates and range
    CENTER_LAT = 43.6532
    CENTER_LON = -79.3832
    RANGE_DEGREES = 10  # Adjust this value as needed (e.g., 0.1 for smaller area)

    # Apply filter to include only points within the specified range
    combined_df = combined_df[
        (combined_df['Latitude'] >= CENTER_LAT - RANGE_DEGREES) &
        (combined_df['Latitude'] <= CENTER_LAT + RANGE_DEGREES) &
        (combined_df['Longitude'] >= CENTER_LON - RANGE_DEGREES) &
        (combined_df['Longitude'] <= CENTER_LON + RANGE_DEGREES)
    ]

    print(f"Filtered data to include points within ±{RANGE_DEGREES} degrees of ({CENTER_LAT}, {CENTER_LON}).")
    print(f"Number of points after filtering: {len(combined_df)}")

    # Remove rows with NaN or 0 values in the target variable
    combined_df = combined_df[(combined_df[target_variable].notna()) & (combined_df[target_variable] != 0)]

    # Prepare data for the model
    X = combined_df[['Latitude', 'Longitude']].values
    y = combined_df[target_variable].values

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create Datasets and DataLoaders
    train_dataset = CustomDataset(X_train_scaled, y_train)
    val_dataset = CustomDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f} - Validation Loss: {val_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}.")

    # ------------------- Prediction for 2025 -------------------
    # Define the date_input for 2025
    date_input_2025 = f"2025{date_input[4:]}"  # Assuming similar format
    date_input_2025 = date_input_2025.replace("-", "")  # Removes all dashes

    print(f"Starting prediction for {data_type} on {date_input_2025}.")

    # Define file paths for 2025 data
    file_paths_2025 = {
        'temp': f'temp/*.A{date_input_2025}.*.hdf',
        'rain': f'rain/*.{date_input_2025}*.nc4',
        'groundwater': f'groundwater/*.A{date_input_2025}.*.hdf'
    }

    # Initialize DataFrame for prediction
    prediction_data_list = []

    # ------------------- Data Loading for 2025 -------------------
    if data_type == "temp":
        # Load temperature data for 2025
        hdf_lst_files_2025 = glob.glob(file_paths['temp'])
        if not hdf_lst_files_2025:
            return jsonify({"error": "No land surface temperature files found for the year 2025."}), 404
        else:
            hdf_lst_2025 = SD(hdf_lst_files_2025[0], SDC.READ)
            lst_data_2025 = hdf_lst_2025.select(dataset_name)
            lst_data_array_2025 = lst_data_2025[:]

            nrows_2025, ncols_2025 = lst_data_array_2025.shape
            lats_2025 = np.zeros((nrows_2025, ncols_2025))
            lons_2025 = np.zeros((nrows_2025, ncols_2025))

            for i in range(nrows_2025):
                for j in range(ncols_2025):
                    lats_2025[i, j] = -1 * modis_sinusoidal_to_lat(v, i)
                    lons_2025[i, j] = modis_sinusoidal_to_lon(h, j)
            print(lats_2025[0,0])
            print(lons_2025[0,0])
            print(lats_2025[12,12])
            print(lons_2025[12,12])
            lat_flat_2025 = lats_2025.flatten()
            lon_flat_2025 = lons_2025.flatten()
            temp_flat_2025 = lst_data_array_2025.flatten()

            lst_df_2025 = pd.DataFrame({
                'Latitude': lat_flat_2025,
                'Longitude': lon_flat_2025,
                'Temperature': temp_flat_2025  
            })

            prediction_data_list.append(lst_df_2025)

    elif data_type == "rain":
        # Load precipitation data for 2025
        rain_files_2025 = glob.glob(file_paths['rain'])
        if not rain_files_2025:
            return jsonify({"error": "No precipitation files found for the year 2025."}), 404
        else:
            with nc.Dataset(rain_files_2025[0], 'r') as ds:
                precipitation_data_2025 = ds.variables['precipitation'][:]
                lat_2025 = ds.variables['lat'][:]
                lon_2025 = ds.variables['lon'][:]
                print(lat_2025)
                print(lon_2025)
                precipitation_data_2025 = precipitation_data_2025[0, :, :]

                precipitation_data_2025 = np.ma.masked_invalid(precipitation_data_2025)
                precip_latitude_2025 = np.repeat(lat_2025, len(lon_2025)).reshape(precipitation_data_2025.shape)
                precip_longitude_2025 = np.tile(lon_2025, len(lat_2025)).reshape(precipitation_data_2025.shape)

                precip_df_2025 = pd.DataFrame({
                    'Latitude': precip_latitude_2025.flatten(),
                    'Longitude': precip_longitude_2025.flatten(),
                    'Precipitation': precipitation_data_2025.flatten()  # Not used for prediction
                })

                prediction_data_list.append(precip_df_2025)

    elif data_type == "groundwater":
        # Load groundwater data for 2025
        hdf_groundwater_files_2025 = glob.glob(file_paths['groundwater'])
        if not hdf_groundwater_files_2025:
            return jsonify({"error": "No groundwater files found for the year 2025."}), 404
        else:
            hdf_groundwater_2025 = SD(hdf_groundwater_files_2025[0], SDC.READ)
            groundwater_data_2025 = hdf_groundwater_2025.select(dataset_name)
            groundwater_data_array_2025 = groundwater_data_2025[:]

            if groundwater_data_array_2025.ndim == 3:
                n_layers_2025, nrows_gw_2025, ncols_gw_2025 = groundwater_data_array_2025.shape
                gw_lats_2025 = np.zeros((n_layers_2025, nrows_gw_2025, ncols_gw_2025))
                gw_lons_2025 = np.zeros((n_layers_2025, nrows_gw_2025, ncols_gw_2025))

                for layer in range(n_layers_2025):
                    for i in range(nrows_gw_2025):
                        for j in range(ncols_gw_2025):
                            gw_lats_2025[layer, i, j] = -1 * modis_sinusoidal_to_lat(v, i)
                            gw_lons_2025[layer, i, j] = modis_sinusoidal_to_lon(h, j)

                gw_lat_flat_2025 = gw_lats_2025.flatten()
                gw_lon_flat_2025 = gw_lons_2025.flatten()
                gw_flat_2025 = groundwater_data_array_2025.flatten()  # Not used for prediction

                groundwater_df_2025 = pd.DataFrame({
                    'Latitude': gw_lat_flat_2025,
                    'Longitude': gw_lon_flat_2025,
                    'Groundwater': gw_flat_2025  # Not used for prediction
                })

                prediction_data_list.append(groundwater_df_2025)
            else:
                return jsonify({"error": "Unexpected data dimensions for groundwater data."}), 500

    # Combine all prediction data for 2025
    combined_df_2025 = pd.concat(prediction_data_list, ignore_index=True)

    # ------------------- Filtering Data for 2025 -------------------
    # Apply the same geographical filter
    combined_df_2025 = combined_df_2025[
        (combined_df_2025['Latitude'] >= CENTER_LAT - RANGE_DEGREES) &
        (combined_df_2025['Latitude'] <= CENTER_LAT + RANGE_DEGREES) &
        (combined_df_2025['Longitude'] >= CENTER_LON - RANGE_DEGREES) &
        (combined_df_2025['Longitude'] <= CENTER_LON + RANGE_DEGREES)
    ]

    print(f"Filtered 2025 data to include points within ±{RANGE_DEGREES} degrees of ({CENTER_LAT}, {CENTER_LON}).")
    print(f"Number of points after filtering: {len(combined_df_2025)}")

    # Prepare data for prediction
    X_2025 = combined_df_2025[['Latitude', 'Longitude']].values

    # Scale features using the same scaler fitted during training
    X_2025_scaled = scaler.transform(X_2025)

    # Convert to tensor
    X_2025_tensor = torch.tensor(X_2025_scaled, dtype=torch.float32).to(device)

    # Load the trained model
    model.eval()
    with torch.no_grad():
        predictions_2025 = model(X_2025_tensor).cpu().numpy()

    # Create a DataFrame for predictions
    predictions_df_2025 = pd.DataFrame({
        'Latitude': combined_df_2025['Latitude'],
        'Longitude': combined_df_2025['Longitude'],
        target_variable: predictions_2025.flatten()
    })

    # Save predictions to CSV
    predictions_file_2025 = f'./predictions/predictions_{data_type}_{date_input_2025}.csv'
    os.makedirs(os.path.dirname(predictions_file_2025), exist_ok=True)
    predictions_df_2025.to_csv(predictions_file_2025, index=False)
    print(f"Saved 2025 predictions to {predictions_file_2025}.")

    return jsonify({"message": f"Model trained and saved to {model_path}. Predictions for 2025 saved to {predictions_file_2025}."}), 200


def generate_map(date, data_type):
    # Load the CSV file
    date = date.replace("-", "")  # Removes all dashes
    csv_file = f'./predictions/predictions_{data_type}_{date}.csv'  # Path to your CSV file
    print(csv_file)

    try:
        data = pd.read_csv(csv_file)
        print("File read successfully.")
    except FileNotFoundError:
        return jsonify(error='CSV file not found'), 404
    except Exception as e:
        return jsonify(error=f'Error reading CSV: {str(e)}'), 500

    print("Mean Latitude:", data['Latitude'].mean())

    try:
        map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
        map_instance = folium.Map(location=map_center, zoom_start=6)

        def get_color(value, data_type):
            if data_type == "rain":
                if value > 20:
                    return 'blue'
                elif value > 10:
                    return 'green'
                else:
                    return 'red'
            elif data_type == "temp":
                if value > 3200:
                    return 'red'
                elif value > 4000:
                    return 'orange'
                else:
                    return 'blue'
            return 'gray'  # Default color

        if data_type == "rain":
            for index, row in data.iterrows():
                popup_content = f"Latitude: {row['Latitude']}<br>Longitude: {row['Longitude']}<br>Precipitation: {row['Precipitation']} mm"
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5 + row['Precipitation'] * 0.2,
                    color=get_color(row['Precipitation'], "rain"),
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, parse_html=True)
                ).add_to(map_instance)
        elif data_type == "temp":
            for index, row in data.iterrows():
                popup_content = (f"Latitude: {row['Latitude']}<br>"
                                 f"Longitude: {row['Longitude']}<br>"
                                 f"Temperature: {row['Temperature']} °C<br>")
                standard_radius = 8  # Standard radius for all markers
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=standard_radius,
                    color=get_color(row['Temperature'], "temp"),
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, parse_html=True)
                ).add_to(map_instance)

        print("Finished adding markers to the map.")

        # Ensure the directory exists
        map_directory = './map'
        if not os.path.exists(map_directory):
            os.makedirs(map_directory)
            print("Created the map directory.")
        else:
            print("Map directory already exists.")

        try:
            # Save the map to an HTML file
            map_file_path = f'{map_directory}/{data_type}_map_{date}.html'  # Use date in filename
            map_instance.save(map_file_path)
            print("Map has been saved at:", map_file_path)
        except Exception as save_error:
            print(f"Error saving the map: {save_error}")
            return jsonify(error=f'Error saving map: {str(save_error)}'), 500

    except Exception as e:
        print(f"An error occurred while generating the map: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/get-map', methods=['POST'])
def get_map():

    # Serve the map file as a response
    data = request.json
    date = data.get('date')
    data_input = data.get('data_type')
    print("Received request to generate map with data:", data)  # Debugging line
    try:
        generate_map(date, data_input)
        print("Map generated successfully.")  # Debugging line
    except Exception as e:
        print(f"Error during map generation: {str(e)}")  # Debugging line
        return jsonify(error='Map generation failed'), 500

    map_file_path = f'./map/{data_input}_map_{date}.html'
    print(f"Checking for map file at: {map_file_path}")  # Debugging line
    if os.path.exists(map_file_path):
        print("Map file exists, returning URL.")  # Debugging line
        return jsonify(mapUrl=map_file_path)
    else:
        print("Map file not found.")  # Debugging line
        return jsonify(error='Map not found'), 404

@app.route('/map/<path:filename>')
def serve_map(filename):
    return send_from_directory('map', f'./map/{filename}')


if __name__ == '__main__':
    app.run(debug=True)
