/**
 * React application entry point.
 * 
 * This file bootstraps the React app by rendering the root App component
 * into the DOM element with id="root" in public/index.html.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
