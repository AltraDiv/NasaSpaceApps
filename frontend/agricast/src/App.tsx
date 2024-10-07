import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/home.tsx";
import Data from "./pages/Data";

const App: React.FC = () => {
  return (
    <Router>
      <nav className="bg-gray-800 p-4">
        <ul className="flex space-x-4">
          <li>
            <Link className="text-white" to="/">
              Home
            </Link>
          </li>
          <li>
            <Link className="text-white" to="/data">
              Predict
            </Link>
          </li>
        </ul>
      </nav>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/data" element={<Data />} />
      </Routes>
    </Router>
  );
};

export default App;
