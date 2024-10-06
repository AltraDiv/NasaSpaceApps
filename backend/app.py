import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta  # For handling future dates
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pyhdf.SD import SD, SDC  # For HDF file handling
import netCDF4 as nc  # For NetCDF file handling
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


# Constants for MODIS Sinusoidal projection
EARTH_RADIUS = 6371007.181  # radius of Earth in meters
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
    # Parse input JSON data
    data = request.json
    date_input = data.get('date')  # e.g., "2023091"
    data_type = data.get('data_type')  # e.g., "temp", "rain", "groundwater"
    date_input = f"2023{date_input[4:]}" 
    date_input = date_input.replace("-", "")  # Removes all dashes
    print(f"Received request for {data_type} data on {date_input}.")
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

    # File paths based on the input date
    file_paths = {
        'temp': f'temp/*.A{date_input}.*.hdf',
        'rain': f'rain/*.{date_input}*.nc4',
        'groundwater': f'groundwater/*.A{date_input}.*.hdf'
    }

    # Initialize DataFrame for future predictions
    data_list = []

    # ------------------- Data Loading -------------------
    if data_type == "temp":
        # Load temperature data
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
                    lats[i, j] = modis_sinusoidal_to_lat(v, i)
                    lons[i, j] = modis_sinusoidal_to_lon(h, j)

            lat_flat = lats.flatten()
            lon_flat = lons.flatten()
            temp_flat = lst_data_array.flatten()

            lst_df = pd.DataFrame({
                'Latitude': lat_flat,
                'Longitude': lon_flat,
                'Temperature': temp_flat
            })

            data_list.append(lst_df)

    elif data_type == "rain":
        # Load precipitation data
        rain_files = glob.glob(file_paths['rain'])
        if not rain_files:
            return jsonify({"error": "No precipitation files found for the specified date."}), 404
        else:
            with nc.Dataset(rain_files[0], 'r') as ds:
                precipitation_data = ds.variables['precipitation'][:]
                lat = ds.variables['lat'][:]
                lon = ds.variables['lon'][:]
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
                            gw_lats[layer, i, j] = modis_sinusoidal_to_lat(v, i)
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

    # Remove rows with NaN or 0 values in the target variable
    combined_df = combined_df[(combined_df[target_variable].notna()) & (combined_df[target_variable] != 0)]

    # Prepare data for the model
    X = combined_df[['Latitude', 'Longitude']].values
    y = combined_df[target_variable].values

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherPredictor().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        return jsonify({"error": f"No pre-trained model found for {data_type}."}), 404

    # Make predictions
    model.eval()
    with torch.no_grad():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        predictions = model(X_tensor).cpu().numpy()

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Latitude': combined_df['Latitude'],
        'Longitude': combined_df['Longitude'],
        target_variable: predictions.flatten()
    })

    # Save predictions to CSV
    date_input = f"2025{date_input[4:]}"
    predictions_file = f'./predictions/predictions_{data_type}_{date_input}.csv'
    predictions_df.to_csv(predictions_file, index=False)

    return send_file(predictions_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)