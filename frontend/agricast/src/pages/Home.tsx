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

      <h1 className="text-4xl font-bold">Introducing Agricast</h1>

      <p className="text-lg">
        Agricast utilizes NASA's IMERG and MODIS precipitation, groundwater, and
        temperature data to train multiple machine learning models that are able
        to provide a prediction of said data, up to 2 years in the future. With
        Agricast, local farmers in the GTA (Greater Toronto Area) can put in a
        future date and select their data of interest, and Agricast will output
        a map of the GTA with the predicted data for that date, along with
        suggestions on how farmers can prepare their crops.
      </p>

      <h2 className="text-3xl font-semibold text-center">
        Local Farmer Interview
      </h2>

      {/* YouTube Video Embed */}
      <div className="flex justify-center">
        <iframe
          width="560"
          height="315"
          src="https://www.youtube.com/embed/3eGCSQ4Qan0"
          title="YouTube video player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>
      </div>
    </div>
  );
};

export default Home;
