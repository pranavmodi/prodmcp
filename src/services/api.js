import axios from 'axios'

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_SERVER_URL || 'http://127.0.0.1:8000',
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.url}`)
    return config
  },
  (error) => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('Response error:', error)
    
    // Handle different types of errors
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response
      console.error(`Server error ${status}:`, data)
      
      // Create a more user-friendly error message
      if (data?.detail) {
        error.userMessage = data.detail
      } else if (data?.error) {
        error.userMessage = data.error
      } else {
        error.userMessage = `Server error: ${status}`
      }
    } else if (error.request) {
      // Request was made but no response received
      error.userMessage = 'No response from server. Please check if the server is running.'
    } else {
      // Something else happened
      error.userMessage = 'An unexpected error occurred.'
    }
    
    return Promise.reject(error)
  }
)

// API methods
export const apiService = {
  // Scrape a website
  async scrapeWebsite(url) {
    try {
      const response = await api.post('/scrape', { url })
      return response.data
    } catch (error) {
      throw error
    }
  },

  // Ask a question
  async askQuestion(question) {
    try {
      const response = await api.post('/ask', { question })
      const data = response.data
      if (data?.mcp_debug) {
        console.log('[MCP DEBUG] client', data.mcp_debug.client)
        console.log('[MCP DEBUG] server', data.mcp_debug.server)
        console.log('[MCP DEBUG] tool', data.mcp_debug.tool)
      }
      return data
    } catch (error) {
      throw error
    }
  },

  // Get server status
  async getStatus() {
    try {
      const response = await api.get('/status')
      return response.data
    } catch (error) {
      throw error
    }
  },

  // List scraped files
  async listFiles() {
    try {
      const response = await api.get('/files')
      return response.data
    } catch (error) {
      throw error
    }
  },

  // Upload a document (txt, docx, pdf)
  async uploadDocument(file) {
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      return response.data
    } catch (error) {
      throw error
    }
  },

  // Test server connection
  async testConnection() {
    try {
      const response = await api.get('/')
      return response.data
    } catch (error) {
      throw error
    }
  }
}

export default api 