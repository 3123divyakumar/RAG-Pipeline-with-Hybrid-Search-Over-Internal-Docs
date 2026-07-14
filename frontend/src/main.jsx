// Entry point — mounts the App component into the #root div of index.html.
// StrictMode is a dev-only tool: it double-invokes renders to surface
// side-effect bugs early. It disappears in production builds.
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
