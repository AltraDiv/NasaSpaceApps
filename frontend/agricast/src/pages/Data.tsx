import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

const Data: React.FC = () => {
  const [coordinates, setCoordinates] = useState<{ lat: number; lng: number } | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date | null>(new Date()); // State to hold the selected date
  const [dataType, setDataType] = useState<string>('precipitation'); // State to hold the selected data type
  const [showCoords, setShowCoords] = useState<boolean>(false); // State to control textbox visibility

  // Custom hook for handling map click events
  const MapClickHandler: React.FC = () => {
    useMapEvents({
      click: (event) => {
        const { lat, lng } = event.latlng;
        setCoordinates({ lat, lng });
        setShowCoords(true); // Show the textbox when coordinates are clicked
      },
    });
    return null;
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-green-200">
      <h1 className="text-4xl font-bold mb-4">Data Page</h1>

      <div className="relative w-full h-1/2 mb-4"> {/* Add mb-4 for margin */}
        <MapContainer 
          center={[43.7, -79.42]} 
          zoom={12} 
          className="w-full h-full"
        >
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
          <option value="precipitation">Precipitation</option>
          <option value="groundwater">Groundwater</option>
          <option value="temperature">Temperature</option>
          <option value="humidity">Humidity</option>
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

      {/* Conditional rendering of the textbox for coordinates */}
      {showCoords && coordinates && (
        <div className="mt-4">
          <label className="block mb-2 text-lg font-semibold">Coordinates:</label>
          <input
            type="text"
            value={`Latitude: ${coordinates.lat}`} // Correctly format to show both
            readOnly
            className="p-2 border border-gray-300 rounded"
          />
          <input
            type="text"
            value={`Longitude: ${coordinates.lng}`} // Correctly format to show both
            readOnly
            className="p-2 border border-gray-300 rounded"
          />
        </div>
      )}
    </div>
  );
};

export default Data;
