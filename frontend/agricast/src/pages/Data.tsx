import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import axios from 'axios';
import * as Papa from 'papaparse'; // Library to parse CSV
import CSVMap from './components/csvplot';

const Data: React.FC = () => {
  const [coordinates, setCoordinates] = useState<{ lat: number; lng: number } | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date | null>(new Date());
  const [dataType, setDataType] = useState<string>('rain');
  const [showCoords, setShowCoords] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]); // State to hold parsed CSV data

  const MapClickHandler: React.FC = () => {
    useMapEvents({
      click: (event) => {
        const { lat, lng } = event.latlng;
        setCoordinates({ lat, lng });
        setShowCoords(true);
      },
    });
    return null;
  };

  const fetchPredictions = async () => {
    if (!coordinates || !selectedDate || !dataType) return;
    setLoading(true);
    setError(null);

    try {
      // Send request to backend to generate predictions and download CSV
      await axios.post('http://127.0.0.1:5000/predict', {
        date: selectedDate.toISOString().split('T')[0],
        data_type: dataType,
        coordinates: coordinates,
      });

      const trimmedDate = selectedDate.toISOString().split('T')[0].replace(/-/g, '');

      // Fetch the CSV file after the backend processing is done
      const response = await axios.get(`../../backend/predictions/predictions_${dataType}_${trimmedDate}.csv`, {
        responseType: 'blob', // Set response type to blob to handle the file
      });

      const reader = new FileReader();
      reader.onload = (e) => {
        const csvData = e.target?.result as string;

        // Parse the CSV data
        Papa.parse(csvData, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setPredictions(results.data); // Set parsed CSV data to state
          },
        });
      };
      reader.readAsText(response.data);

      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-green-200">
      <h1 className="text-4xl font-bold mb-4">Data Page</h1>

      <div className="relative w-full h-1/2 mb-4">
        <MapContainer center={[43.7, -79.42]} zoom={12} className="w-full h-full">
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          <MapClickHandler />
          {coordinates && (
            <Marker position={[coordinates.lat, coordinates.lng]} icon={L.divIcon({ className: 'custom-icon' })}>
              <Popup>
                <div>
                  <p>Latitude: {coordinates.lat}</p>
                  <p>Longitude: {coordinates.lng}</p>
                </div>
              </Popup>
            </Marker>
          )}
        </MapContainer>
      </div>

      {/* Data Type Selection */}
      <div className="mb-4">
        <label className="block mb-2 text-lg font-semibold">Select Data Type:</label>
        <select
          value={dataType}
          onChange={(e) => setDataType(e.target.value)}
          className="p-2 border border-gray-300 rounded"
        >
          <option value="rain">Precipitation</option>
          <option value="groundwater">Groundwater</option>
          <option value="temp">Temperature</option>
        </select>
      </div>

      {/* Date Picker */}
      <div className="mb-4">
        <label className="block mb-2 text-lg font-semibold">Select Date (Pick from 2025 April - November):</label>
        <DatePicker
          selected={selectedDate}
          onChange={(date) => setSelectedDate(date)}
          dateFormat="yyyy/MM/dd"
          minDate={new Date(2025, 3, 1)} // April 1, 2025
          maxDate={new Date(2025, 10, 31)} // November 30, 2025
          className="p-2 border border-gray-300 rounded"
        />
      </div>

      <p className="mb-4">Selected Date: {selectedDate?.toLocaleDateString()}</p>
      <p className="mb-4">Selected Data Type: {dataType}</p>

      {/* Button to fetch predictions */}
      <button
        onClick={fetchPredictions}
        className="px-4 py-2 bg-blue-500 text-white rounded"
      >
        Fetch Predictions
      </button>

      {/* Display loading state */}
      {loading && <p className="mt-4">Loading predictions...</p>}
      {error && <p className="mt-4 text-red-500">{error}</p>}

      {/* Conditional rendering of the textbox for coordinates */}
      {showCoords && coordinates && (
        <div className="mt-4">
          <label className="block mb-2 text-lg font-semibold">Coordinates:</label>
          <input
            type="text"
            value={`Latitude: ${coordinates.lat}`} 
            readOnly
            className="p-2 border border-gray-300 rounded"
          />
          <input
            type="text"
            value={`Longitude: ${coordinates.lng}`} 
            readOnly
            className="p-2 border border-gray-300 rounded"
          />
        </div>
      )}

      <CSVMap />
      
    </div>
  );
};

export default Data;
