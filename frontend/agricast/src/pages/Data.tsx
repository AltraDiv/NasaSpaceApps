<<<<<<< HEAD
// src/Data.tsx
import React, { useState } from 'react';
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
=======
// src/App.tsx
import React, { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
>>>>>>> 39e2589ee0c4bb355f0c0fa2f432a2ab6ca33ad5

const Data: React.FC = () => {
  // State variables for form inputs
  const [date, setDate] = useState<string>("");
  const [dataType, setDataType] = useState<string>("temp");

  // State variables for additional options (optional)
  const [epochs, setEpochs] = useState<number>(20);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [overwrite, setOverwrite] = useState<boolean>(false);

  // State variables for handling loading and response
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [responseMessage, setResponseMessage] = useState<string>("");

  // Handler for form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Basic validation
    if (!date) {
      toast.error("Please select a date.");
      return;
    }

    if (!["temp", "rain", "groundwater"].includes(dataType)) {
      toast.error("Invalid data type selected.");
      return;
    }

    setIsLoading(true);
    setResponseMessage("");

    try {
      // Format date as per backend expectations
      // Assuming backend expects 'YYYYMMDD'
      const formattedDate = date.replace(/-/g, "");

      // Prepare the payload
      const payload = {
        date: formattedDate, // Formatted date
        data_type: dataType,
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        overwrite: overwrite,
      };

      // Make the POST request to the backend
      const response = await axios.post(
        "http://localhost:5000/predict",
        payload,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      // Handle success response
      if (response.status === 200) {
        setResponseMessage(response.data.message);
        toast.success(response.data.message);
        const map_resp = await axios.post('http://localhost:5000/get-map', {
          date: formattedDate,    
          data_type: dataType,
        }, {
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (map_resp.status === 200) {
          console.log(map_resp.data);
          const fileName = `${dataType}_map_${date}.html`; // Update based on your filename format
          window.open(`/map/${fileName}`, '_blank');

        }

      } else {
        toast.error("Unexpected response from the server.");
      }
    } catch (error: any) {
      // Handle error response
      if (error.response && error.response.data && error.response.data.error) {
        toast.error(`Error: ${error.response.data.error}`);
      } else if (
        error.response &&
        error.response.data &&
        error.response.data.message
      ) {
        toast.info(`Info: ${error.response.data.message}`);
      } else {
        toast.error("An error occurred while processing your request.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-green-100 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-8">
        <h1 className="text-2xl font-bold mb-6 text-center">
          Water and Weather Predictor
        </h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Date Input */}
          <div>
            <label
              htmlFor="date"
              className="block text-sm font-medium text-gray-700"
            >
              Date:
            </label>
            <input
              type="date"
              id="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Data Type Selection */}
          <div>
            <label
              htmlFor="dataType"
              className="block text-sm font-medium text-gray-700"
            >
              Data Type:
            </label>
            <select
              id="dataType"
              value={dataType}
              onChange={(e) => setDataType(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="temp">Temperature</option>
              <option value="rain">Precipitation</option>
              <option value="groundwater">Groundwater</option>
            </select>
          </div>

          {/* Additional Options (Optional) */}
          <fieldset className="border border-gray-300 rounded-md p-4">
            <legend className="text-sm font-medium text-gray-700">
              Training Parameters (Optional)
            </legend>

            <div className="mt-4 space-y-4">
              {/* Epochs */}
              <div>
                <label
                  htmlFor="epochs"
                  className="block text-sm font-medium text-gray-700"
                >
                  Epochs:
                </label>
                <input
                  type="number"
                  id="epochs"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  min={1}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Batch Size */}
              <div>
                <label
                  htmlFor="batchSize"
                  className="block text-sm font-medium text-gray-700"
                >
                  Batch Size:
                </label>
                <input
                  type="number"
                  id="batchSize"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  min={1}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Learning Rate */}
              <div>
                <label
                  htmlFor="learningRate"
                  className="block text-sm font-medium text-gray-700"
                >
                  Learning Rate:
                </label>
                <input
                  type="number"
                  id="learningRate"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  step="0.0001"
                  min={0}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Overwrite Checkbox */}
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="overwrite"
                  checked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label
                  htmlFor="overwrite"
                  className="ml-2 block text-sm text-gray-700"
                >
                  Overwrite Existing Model
                </label>
              </div>
            </div>
          </fieldset>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              isLoading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
          >
            {isLoading ? "Processing..." : "Predict"}
          </button>
        </form>

        {/* Response Message */}
        {responseMessage && (
          <div className="mt-6 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
            <p>{responseMessage}</p>
          </div>
        )}
      </div>

      {/* Toast Notifications */}
      <ToastContainer />
    </div>
  );
};

export default Data;
