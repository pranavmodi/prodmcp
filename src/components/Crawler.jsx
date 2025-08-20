import React, { useState } from 'react';
import axios from 'axios';

const Crawler = () => {
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState(50);
  const [delayRange, setDelayRange] = useState([3, 7]);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleCrawl = async (e) => {
    e.preventDefault();
    if (!url.trim()) {
      setError('Please enter a website URL');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/crawl', {
        url: url.trim(),
        max_pages: parseInt(maxPages),
        delay_range: delayRange
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to crawl website');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelayChange = (index, value) => {
    const newDelayRange = [...delayRange];
    newDelayRange[index] = parseInt(value);
    setDelayRange(newDelayRange);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          🕷️ Website Crawler
        </h2>
        
        <p className="text-gray-600 mb-6 text-center">
          Crawl entire websites to discover and scrape all pages automatically
        </p>

        <form onSubmit={handleCrawl} className="space-y-6">
          {/* URL Input */}
          <div>
            <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
              Website URL *
            </label>
            <input
              type="url"
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
            <p className="text-sm text-gray-500 mt-1">
              Enter the starting URL of the website you want to crawl
            </p>
          </div>

          {/* Max Pages */}
          <div>
            <label htmlFor="maxPages" className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Pages to Crawl
            </label>
            <input
              type="number"
              id="maxPages"
              value={maxPages}
              onChange={(e) => setMaxPages(e.target.value)}
              min="1"
              max="1000"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-sm text-gray-500 mt-1">
              Limit the total number of pages to crawl (1-1000)
            </p>
          </div>

          {/* Delay Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Delay Between Requests (seconds)
            </label>
            <div className="flex space-x-4">
              <div className="flex-1">
                <label htmlFor="minDelay" className="block text-xs text-gray-500 mb-1">
                  Minimum Delay
                </label>
                <input
                  type="number"
                  id="minDelay"
                  value={delayRange[0]}
                  onChange={(e) => handleDelayChange(0, e.target.value)}
                  min="1"
                  max="30"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div className="flex-1">
                <label htmlFor="maxDelay" className="block text-xs text-gray-500 mb-1">
                  Maximum Delay
                </label>
                <input
                  type="number"
                  id="maxDelay"
                  value={delayRange[1]}
                  onChange={(e) => handleDelayChange(1, e.target.value)}
                  min="1"
                  max="30"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            <p className="text-sm text-gray-500 mt-1">
              Random delay between requests to be respectful to servers
            </p>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition-colors ${
              isLoading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Crawling Website...
              </span>
            ) : (
              '🕷️ Crawl Entire Website'
            )}
          </button>
        </form>

        {/* Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3 flex-1">
                <h3 className="text-lg font-medium text-green-800">Website Crawling Completed!</h3>
                <div className="mt-2 text-sm text-green-700">
                  <p><strong>Status:</strong> {result.status}</p>
                  <p><strong>Message:</strong> {result.message}</p>
                  <p><strong>Scraped Files:</strong> {result.scraped_files?.length || 0}</p>
                  <p><strong>Failed URLs:</strong> {result.failed_urls?.length || 0}</p>
                  
                  {result.crawl_stats && (
                    <div className="mt-3 p-3 bg-white rounded border">
                      <h4 className="font-medium text-green-800 mb-2">Crawl Statistics:</h4>
                      <ul className="text-sm space-y-1">
                        <li>Total Accepted URLs: {result.crawl_stats.total_accepted}</li>
                        <li>Total Rejected URLs: {result.crawl_stats.total_rejected}</li>
                        <li>Total URLs: {result.crawl_stats.total_urls}</li>
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info Box */}
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">How Website Crawling Works</h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>Unlike single page scraping, website crawling:</p>
                <ul className="mt-1 list-disc list-inside space-y-1">
                  <li>Discovers ALL pages on the website automatically</li>
                  <li>Follows internal links to find new pages</li>
                  <li>Scrapes each discovered page</li>
                  <li>Organizes content by domain and page structure</li>
                  <li>Filters out irrelevant content (blogs, news, etc.)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Crawler; 