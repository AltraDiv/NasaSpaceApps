import React from "react";

const Home: React.FC = () => {
  return (
    <div className="flex flex-col justify-start min-h-screen bg-blue-200 p-8 space-y-4">
      <h1 className="text-4xl font-bold">
        Challenge:{" "}
        <span className="text-3xl font-semibold">
          Leveraging Earth Observation Data for Informed Agricultural
          Decision-Making
        </span>
      </h1>

      <p className="text-lg">
        Farmers face a deluge of water-related challenges due to unpredictable
        weather, pests, and diseases. These factors can significantly impact
        crop health, farmers’ profits, and food security. Depending upon the
        geography, many farmers may face droughts or floods—sometimes both of
        these extreme events occur within the same season! Your challenge is to
        design a tool that empowers farmers to easily explore, analyze, and
        utilize NASA datasets to address these water-related concerns and
        improve their farming practices.
      </p>

      <h1 className="text-4xl font-bold">Agricast</h1>

      <p className="text-lg">
        With Agricast, local farmers in the GTA (Greater Toronto Area) can put
        in a future date and select temperature, precipitation, and groundwater.
        Based on NASA satellite data (IMERG, MODIS), Agricast will output a map
        of the GTA with predicted data for that future date, allowing farmers to
        prepare and protect their crops well in advance.
      </p>
    </div>
  );
};

export default Home;
