import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta  # For handling future dates
from pyhdf.SD import SD, SDC  # For HDF file handling
import netCDF4 as nc  # For NetCDF file handling
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

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

# User inputs for date and data type
date_input = input("Enter the date (e.g., 2023091): ")
data_type = input("Enter the data type (temp, rain, groundwater): ").strip().lower()

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
    print("Invalid data type entered. Please choose from 'temp', 'rain', or 'groundwater'.")
    exit()

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
    # Land Surface Temperature Data
    hdf_lst_files = glob.glob(file_paths['temp'])
    if not hdf_lst_files:
        print("No land surface temperature files found for the specified date.")
        exit()
    else:
        hdf_lst = SD(hdf_lst_files[0], SDC.READ)
        lst_data = hdf_lst.select(dataset_name)  # Use the actual dataset name for LST
        lst_data_array = lst_data[:]  # Extract the LST data

        # Get the shape of the LST data
        nrows, ncols = lst_data_array.shape

        # Initialize arrays for latitudes and longitudes
        lats = np.zeros((nrows, ncols))
        lons = np.zeros((nrows, ncols))

        # Compute latitudes and longitudes for each pixel
        for i in range(nrows):
            for j in range(ncols):
                lats[i, j] = modis_sinusoidal_to_lat(v, i)
                lons[i, j] = modis_sinusoidal_to_lon(h, j)

        # Flatten the arrays for latitude, longitude, and temperature
        lat_flat = lats.flatten()
        lon_flat = lons.flatten()
        temp_flat = lst_data_array.flatten()

        # Create a DataFrame for LST
        lst_df = pd.DataFrame({
            'Latitude': lat_flat,
            'Longitude': lon_flat,
            'Temperature': temp_flat
        })

        print("Land Surface Temperature Data:")
        print(lst_df.head())
        data_list.append(lst_df)

elif data_type == "rain":
    # Precipitation Data
    rain_files = glob.glob(file_paths['rain'])
    if not rain_files:
        print("No precipitation files found for the specified date.")
        exit()
    else:
        with nc.Dataset(rain_files[0], 'r') as ds:
            # Load the 'precipitation', 'lat', and 'lon' variables
            precipitation_data = ds.variables['precipitation'][:]
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]

            # Remove the first dimension (time dimension)
            precipitation_data = precipitation_data[0, :, :]

            # Mask invalid data (e.g., NaNs or extreme values)
            precipitation_data = np.ma.masked_invalid(precipitation_data)

            # Create a DataFrame for Precipitation
            precip_latitude = np.repeat(lat, len(lon)).reshape(precipitation_data.shape)
            precip_longitude = np.tile(lon, len(lat)).reshape(precipitation_data.shape)

            precip_df = pd.DataFrame({
                'Latitude': precip_latitude.flatten(),
                'Longitude': precip_longitude.flatten(),
                'Precipitation': precipitation_data.flatten()
            })

            print("Precipitation Data:")
            print(precip_df.head())
            data_list.append(precip_df)

elif data_type == "groundwater":
    # Groundwater Data
    hdf_groundwater_files = glob.glob(file_paths['groundwater'])
    if not hdf_groundwater_files:
        print("No groundwater files found for the specified date.")
        exit()
    else:
        hdf_groundwater = SD(hdf_groundwater_files[0], SDC.READ)
        groundwater_data = hdf_groundwater.select(dataset_name)  # Use the actual dataset name for groundwater
        groundwater_data_array = groundwater_data[:]  # Extract groundwater data

        # Check the shape of the groundwater data array
        print("Groundwater data shape:", groundwater_data_array.shape)

        # Handle the dimensions of groundwater_data_array
        if groundwater_data_array.ndim == 3:  # Shape is (layers, 1200, 1200)
            n_layers, nrows_gw, ncols_gw = groundwater_data_array.shape

            # Initialize arrays for groundwater latitudes and longitudes
            gw_lats = np.zeros((n_layers, nrows_gw, ncols_gw))
            gw_lons = np.zeros((n_layers, nrows_gw, ncols_gw))

            # Compute latitudes and longitudes for groundwater data
            for layer in range(n_layers):
                for i in range(nrows_gw):
                    for j in range(ncols_gw):
                        gw_lats[layer, i, j] = modis_sinusoidal_to_lat(v, i)
                        gw_lons[layer, i, j] = modis_sinusoidal_to_lon(h, j)

            # Flatten the arrays for latitude, longitude, and groundwater values
            gw_lat_flat = gw_lats.flatten()
            gw_lon_flat = gw_lons.flatten()
            gw_flat = groundwater_data_array.flatten()

            # Create a DataFrame for Groundwater
            groundwater_df = pd.DataFrame({
                'Latitude': gw_lat_flat,
                'Longitude': gw_lon_flat,
                'Groundwater': gw_flat
            })

            print("Groundwater Data:")
            print(groundwater_df.head())
            data_list.append(groundwater_df)
        else:
            print("Unexpected data dimensions for groundwater data.")
            exit()

# Combine all data for model training
combined_df = pd.concat(data_list, ignore_index=True)

# Remove rows with NaN or 0 values in the target variable
combined_df = combined_df[(combined_df[target_variable].notna()) & (combined_df[target_variable] != 0)]

# Prepare data for the model
X = combined_df[['Latitude', 'Longitude']].values
y = combined_df[target_variable].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Create DataLoader
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
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

# Instantiate the model, define loss and optimizer
model = WeatherPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Check if the model is already trained
if os.path.exists(model_path):
    print(f"Loading the pre-trained model for {data_type}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"No pre-trained model found for {data_type}. Training the model...")

    # Training loop with mixed precision
    num_epochs = 2  # Increase epochs for better training
    scaler_amp = GradScaler()  # Mixed precision scaler

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'.")

# Generate future predictions (after a year)
model.eval()
with torch.no_grad():
    # Assuming the same coordinates are used for prediction
    X_all = combined_df[['Latitude', 'Longitude']].values
    X_all_scaled = scaler.transform(X_all)
    X_tensor = torch.tensor(X_all_scaled, dtype=torch.float32).to(device)
    future_outputs = model(X_tensor).cpu().numpy()

# Generate future date
future_date = datetime.now() + timedelta(days=365)
future_date_str = future_date.strftime("%Y-%m-%d")

# Create a DataFrame for predictions
future_df = pd.DataFrame({
    'Latitude': combined_df['Latitude'],
    'Longitude': combined_df['Longitude'],
    f'Predicted {target_variable}': future_outputs.flatten(),
    'Date': [future_date_str] * len(future_outputs)
})

# Save the predictions to a CSV file
output_file = f'./predictions/predicted_{data_type}_{future_date.strftime("%Y%m%d")}.csv'
future_df.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'.")
