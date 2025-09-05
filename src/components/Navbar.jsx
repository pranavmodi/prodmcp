import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import api from '../services/api'
import { Globe, MessageCircle } from 'lucide-react'

const Navbar = () => {
  const location = useLocation()
  const [crawlStatus, setCrawlStatus] = React.useState({ status: 'idle', crawl: { percent: 0 }, scrape: { percent: 0 } })

  React.useEffect(() => {
    let timer = null
    const poll = async () => {
      try {
        const { data } = await api.get('/crawl/stats')
        const job = data?.crawl_stats || {}
        setCrawlStatus(job)
      } catch (e) {
        // ignore
      }
    }
    // start polling every 1s so progress is visible across tabs
    poll()
    timer = setInterval(poll, 1000)
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [])

  const isActive = (path) => {
    return location.pathname === path
  }

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <Globe className="h-8 w-8 text-primary-600" />
            <span className="text-xl font-bold text-gray-900">MCP Scraper</span>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center space-x-1">
            <Link
              to="/crawl"
              className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                isActive('/crawl') || isActive('/')
                  ? 'bg-primary-100 text-primary-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              <div className="flex items-center space-x-2">
                <Globe className="h-4 w-4" />
                <span>Crawl</span>
              </div>
            </Link>
            
            <Link
              to="/ask"
              className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                isActive('/ask')
                  ? 'bg-primary-100 text-primary-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              <div className="flex items-center space-x-2">
                <MessageCircle className="h-4 w-4" />
                <span>Ask</span>
              </div>
            </Link>

            {/* Global Crawl Indicator */}
            {crawlStatus?.status && crawlStatus.status !== 'idle' && (
              <div className="ml-3 px-3 py-1 text-xs rounded-full bg-gray-100 text-gray-700 border">
                <span className="font-medium">{crawlStatus.status === 'crawl' ? 'Crawling' : crawlStatus.status === 'scrape' ? 'Scraping' : crawlStatus.status}</span>
                {typeof crawlStatus?.crawl?.percent === 'number' && (
                  <span className="ml-2">{crawlStatus.crawl.percent}%</span>
                )}
                {typeof crawlStatus?.scrape?.percent === 'number' && crawlStatus.scrape.percent > 0 && (
                  <span className="ml-2">| {crawlStatus.scrape.percent}%</span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar 