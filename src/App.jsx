import React, { useState } from 'react';
import { Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Ask from './pages/Ask';
import Crawler from './components/Crawler';
import Navbar from './components/Navbar';

function App() {
  const [activeTab, setActiveTab] = useState('scrape');
  const navigate = useNavigate();
  const location = useLocation();

  // Update active tab based on current route
  React.useEffect(() => {
    console.log('Current pathname:', location.pathname);
    if (location.pathname === '/ask') {
      console.log('Setting active tab to ask');
      setActiveTab('ask');
    } else if (location.pathname === '/crawl') {
      console.log('Setting active tab to crawl');
      setActiveTab('crawl');
    } else {
      console.log('Setting active tab to scrape');
      setActiveTab('scrape');
    }
  }, [location.pathname]);

  const handleTabChange = (tab) => {
    console.log('Tab changed to:', tab);
    setActiveTab(tab);
    if (tab === 'scrape') {
      navigate('/');
    } else if (tab === 'crawl') {
      navigate('/crawl');
    } else if (tab === 'ask') {
      navigate('/ask');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      <div className="container mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-lg shadow-md p-1">
            <div className="flex space-x-1">
              <button
                onClick={() => handleTabChange('scrape')}
                className={`px-6 py-3 rounded-md font-medium transition-colors ${
                  activeTab === 'scrape'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                📄 Single Page Scraper
              </button>
              <button
                onClick={() => handleTabChange('crawl')}
                className={`px-6 py-3 rounded-md font-medium transition-colors ${
                  activeTab === 'crawl'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                🕷️ Website Crawler
              </button>
              <button
                onClick={() => handleTabChange('ask')}
                className={`px-6 py-3 rounded-md font-medium transition-colors ${
                  activeTab === 'ask'
                    ? 'bg-green-600 text-white shadow-sm'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                🤖 Ask Questions
              </button>
            </div>
          </div>
        </div>

        {/* Main Content Area with Routes */}
        <Routes>
          <Route path="/" element={
            <div>
              <div className="text-center mb-6">
                <h1 className="text-4xl font-bold text-gray-800 mb-4">
                  📄 Single Page Scraper
                </h1>
                <p className="text-xl text-gray-600">
                  Scrape individual web pages and save their content
                </p>
              </div>
              <Home />
            </div>
          } />
          
          <Route path="/crawl" element={
            <div>
              <div className="text-center mb-6">
                <h1 className="text-4xl font-bold text-gray-800 mb-4">
                  🕷️ Website Crawler
                </h1>
                <p className="text-xl text-gray-600">
                  Crawl entire websites to discover and scrape all pages automatically
                </p>
              </div>
              <Crawler />
            </div>
          } />
          
          <Route path="/ask" element={
            <div>
              <div className="text-center mb-6">
                <h1 className="text-4xl font-bold text-gray-800 mb-4">
                  🤖 Ask Questions About Scraped Content
                </h1>
                <p className="text-xl text-gray-600">
                  Get AI-powered answers about the content you've scraped
                </p>
              </div>
              <Ask />
            </div>
          } />
          
          {/* Test route to verify routing is working */}
          <Route path="/test" element={
            <div className="text-center p-8">
              <h1 className="text-2xl font-bold text-green-600 mb-4">✅ Routing Test Successful!</h1>
              <p className="text-gray-600">If you can see this, React Router is working properly.</p>
              <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">← Go back</Link>
            </div>
          } />
        </Routes>

        {/* Navigation Links - Only show when not on ask page */}
        {location.pathname !== '/ask' && (
          <div className="text-center mt-8">
            <div className="inline-flex space-x-4 bg-white rounded-lg shadow-md p-4">
              <Link
                to="/ask"
                onClick={() => console.log('Ask link clicked, navigating to /ask')}
                className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors"
              >
                🤖 Ask Questions About Scraped Content
              </Link>
              <Link
                to="/test"
                className="px-4 py-3 bg-yellow-500 text-white rounded-lg font-medium hover:bg-yellow-600 transition-colors text-sm"
              >
                🧪 Test Route
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 