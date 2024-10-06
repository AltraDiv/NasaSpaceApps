import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';

const Data: React.FC = () => {
  const [coordinates, setCoordinates] = useState<{ lat: number; lng: number } | null>(null);
  const [currentDate] = useState<string>(new Date().toLocaleDateString());
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
      <p className="mb-4">Current Date: {currentDate}</p>
      <MapContainer 
        center={[43.7, -79.42]} 
        zoom={12} 
        className="w-1/2 h-1/2"
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
