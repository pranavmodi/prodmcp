import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Globe, MessageCircle } from 'lucide-react'

const Navbar = () => {
  const location = useLocation()

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
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar 