import React, { useState } from 'react';
import Papa from 'papaparse';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Import marker icons using ES module syntax
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// Ensure the default marker icon is set up correctly
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const CSVMap = () => {
  const [data, setData] = useState<{ lat: number; lng: number; precipitation: number }[]>([]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) return;

    Papa.parse(file, {
      header: true, // Treat first row as headers
      skipEmptyLines: true,
      complete: (result) => {
        const parsedData = result.data.map((row: any) => ({
          lat: parseFloat(row.Latitude),
          lng: parseFloat(row.Longitude),
          precipitation: parseFloat(row.Precipitation),
        }));
        setData(parsedData);
      },
    });
  };

  return (
    <div>
      <input type="file" accept=".csv" onChange={handleFileUpload} />
      <MapContainer center={[-89.95, -165.75]} zoom={5} style={{ height: '600px', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {data.map((point, index) => (
          <Marker key={index} position={[point.lat, point.lng]}>
            <Popup>
              Precipitation: {point.precipitation}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default CSVMap;
