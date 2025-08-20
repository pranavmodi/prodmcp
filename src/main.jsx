import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
// Alternative: import { HashRouter } from 'react-router-dom' // Use this if you have routing issues
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      {/* Alternative: <HashRouter> for hash-based routing */}
      <App />
      {/* Alternative: </HashRouter> */}
    </BrowserRouter>
  </React.StrictMode>,
) 