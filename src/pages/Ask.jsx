import React, { useState, useEffect } from 'react'
import { MessageCircle, Send, Bot, User, AlertCircle, Info, ArrowLeft } from 'lucide-react'
import { apiService } from '../services/api'
import Loader from '../components/Loader'
import { Link } from 'react-router-dom'

const Ask = () => {
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversation, setConversation] = useState([])
  const [error, setError] = useState(null)
  const [availableFiles, setAvailableFiles] = useState(null)

  useEffect(() => {
    // Load available files on component mount
    loadAvailableFiles()
  }, [])

  const loadAvailableFiles = async () => {
    try {
      const files = await apiService.listFiles()
      setAvailableFiles(files)
    } catch (error) {
      console.error('Failed to load files:', error)
    }
  }

  

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!question.trim()) {
      setError('Please enter a question')
      return
    }

    const userQuestion = question.trim()
    setIsLoading(true)
    setError(null)
    setQuestion('')

    // Add user question to conversation
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: userQuestion,
      timestamp: new Date().toLocaleTimeString()
    }
    
    setConversation(prev => [...prev, userMessage])

    try {
      const response = await apiService.askQuestion(userQuestion)
      
      // Add AI response to conversation
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: response.answer,
        contextInfo: response.context_info,
        timestamp: new Date().toLocaleTimeString()
      }
      
      setConversation(prev => [...prev, aiMessage])
      
      // Refresh available files
      loadAvailableFiles()
      
    } catch (error) {
      // Add error message to conversation
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: error.userMessage || 'Failed to get answer',
        timestamp: new Date().toLocaleTimeString()
      }
      
      setConversation(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const clearConversation = () => {
    setConversation([])
    setError(null)
  }

  const getMessageIcon = (type) => {
    switch (type) {
      case 'user':
        return <User className="h-5 w-5 text-blue-600" />
      case 'ai':
        return <Bot className="h-5 w-5 text-green-600" />
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-600" />
      default:
        return <MessageCircle className="h-5 w-5 text-gray-600" />
    }
  }

  const getMessageStyle = (type) => {
    switch (type) {
      case 'user':
        return 'bg-blue-100 border-blue-200 text-blue-900'
      case 'ai':
        return 'bg-green-100 border-green-200 text-green-900'
      case 'error':
        return 'bg-red-100 border-red-200 text-red-900'
      default:
        return 'bg-gray-100 border-gray-200 text-gray-900'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="flex justify-center mb-4">
          <MessageCircle className="h-16 w-16 text-green-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Ask Questions
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Ask questions about the websites you've scraped. 
          The AI will analyze the content and provide answers.
        </p>
      </div>

      {/* Back Button */}
      <div className="flex justify-start">
        <Link
          to="/"
          className="inline-flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Scraper</span>
        </Link>
      </div>

      {/* Available Files Info */}
      {availableFiles && (
        <div className="card bg-blue-50 border-blue-200">
          <div className="flex items-start space-x-3">
            <Info className="h-5 w-5 text-blue-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-medium text-blue-800 mb-2">
                Available Content
              </h3>
              <p className="text-sm text-blue-700 mb-2">
                You have {availableFiles.total_files} scraped website{availableFiles.total_files !== 1 ? 's' : ''} available for analysis.
              </p>
              
              {availableFiles.files && availableFiles.files.length > 0 && (
                <div className="text-xs text-blue-600 space-y-1">
                  {availableFiles.files.slice(0, 3).map((file, index) => (
                    <p key={index}>• {file}</p>
                  ))}
                  {availableFiles.files.length > 3 && (
                    <p>... and {availableFiles.files.length - 3} more</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Question Form */}
      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
              Your Question
            </label>
            <div className="flex space-x-3">
              <input
                type="text"
                id="question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask anything about the scraped websites..."
                className="input-field flex-1"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !question.trim()}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {isLoading ? (
                  <Loader size="sm" text="" />
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    <span>Ask</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="card text-center">
          <Loader size="lg" text="Analyzing content and generating answer..." />
        </div>
      )}

      {/* Conversation */}
      {conversation.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Conversation</h3>
            <button
              onClick={clearConversation}
              className="text-sm text-gray-500 hover:text-gray-700 underline"
            >
              Clear
            </button>
          </div>
          
          <div className="space-y-4">
            {conversation.map((message) => (
              <div
                key={message.id}
                className={`p-4 rounded-lg border ${getMessageStyle(message.type)}`}
              >
                <div className="flex items-start space-x-3">
                  {getMessageIcon(message.type)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">
                        {message.type === 'user' ? 'You' : 
                         message.type === 'ai' ? 'AI Assistant' : 'Error'}
                      </span>
                      <span className="text-xs opacity-70">{message.timestamp}</span>
                    </div>
                    
                    <div className="text-sm leading-relaxed">
                      {message.content}
                    </div>
                    
                    {/* Context Info for AI responses */}
                    {message.type === 'ai' && message.contextInfo && (
                      <div className="mt-3 pt-3 border-t border-green-300">
                        <p className="text-xs text-green-700 font-medium mb-1">
                          Based on {message.contextInfo.total_files} file(s):
                        </p>
                        <div className="text-xs text-green-600 space-y-1">
                          {message.contextInfo.files?.slice(0, 3).map((file, index) => (
                            <p key={index}>• {file.filename}</p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="card border-red-200 bg-red-50">
          <div className="flex items-start space-x-3">
            <AlertCircle className="h-6 w-6 text-red-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-800 mb-2">
                Error
              </h3>
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="card bg-green-50 border-green-200">
        <h3 className="text-lg font-semibold text-green-800 mb-3">
          Tips for better questions
        </h3>
        <div className="space-y-2 text-sm text-green-700">
          <p>• Ask specific questions about content, features, or information</p>
          <p>• Use "What", "How", "Why", "When", "Where" questions</p>
          <p>• Ask about specific topics mentioned in the scraped websites</p>
          <p>• Request summaries or comparisons between different pages</p>
        </div>
      </div>
    </div>
  )
}

export default Ask 