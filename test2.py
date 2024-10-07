import pandas as pd
from netCDF4 import Dataset

# Specify the path to your NetCDF4 file
netcdf_file_path = './backend/rain/3B-DAY.MS.MRG.3IMERG.20230401-S000000-E235959.V07B.nc4'  # Update this path
csv_file_path = 'output.csv'  # Name of the output CSV file

# Open the NetCDF file
with Dataset(netcdf_file_path, 'r') as nc:
    # Print the variable names to understand the structure
    print("Available variables:", nc.variables.keys())
    
    # Extract relevant data
    latitudes = nc.variables['lat'][:]
    longitudes = nc.variables['lon'][:]
    precipitation = nc.variables['precipitation'][:]
    
    # If precipitation has more than 2 dimensions (e.g., time), you may need to average or select a specific time slice
    # Here, I'll assume it's a 2D array for simplicity. If it's 3D, you'll need to adjust accordingly.
    if precipitation.ndim == 3:  # Time dimension exists
        precipitation = precipitation[0]  # Select the first time step, adjust if necessary

    # Flatten arrays if necessary and create a DataFrame
    df = pd.DataFrame({
        'Latitude': latitudes.flatten(),
        'Longitude': longitudes.flatten(),
        'Precipitation': precipitation.flatten()
    })

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    print(f'Successfully saved data to {csv_file_path}')
