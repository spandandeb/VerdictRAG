import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  Search, 
  FileText, 
  MessageCircle, 
  Book, 
  Scale, 
  AlertCircle,
  Download,
  ExternalLink,
  Loader2,
  CheckCircle,
  XCircle,
  Database,
  Zap,
  Activity
} from 'lucide-react';

const VerdictRAG = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [documentStats, setDocumentStats] = useState(null);
  const [chunks, setChunks] = useState(null);
  const fileInputRef = useRef(null);

  // Backend API base URL
  const API_BASE_URL = 'http://localhost:5000';

  // Check backend health
  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const health = await response.json();
      setBackendHealth(health);
      console.log('Backend Health:', health);
      return health;
    } catch (error) {
      console.error('Backend not available:', error);
      setError('Backend server is not running. Please start the Flask server on port 5000.');
      return null;
    }
  };

  // Upload and process document with RAG
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setUploadedFile(file);
    setError(null);
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      // Store document stats and case info
      setDocumentStats(result.document_stats);
      
      // Set basic analysis info
      setAnalysis({
        title: result.case_info?.title || 'Legal Document',
        caseInfo: result.case_info,
        processingStatus: result.message,
        ragStatus: result.rag_status,
        documentStats: result.document_stats
      });

      // Switch to analysis tab
      setActiveTab('analysis');
      
    } catch (error) {
      setError(`Upload failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Perform comprehensive analysis using RAG
  const handleComprehensiveAnalysis = async () => {
    if (!uploadedFile || !documentStats) {
      setError('Please upload a document first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      // Update analysis with comprehensive results
      setAnalysis(prev => ({
        ...prev,
        comprehensiveAnalysis: result.comprehensive_analysis,
        system: result.system,
        documentStats: result.document_stats
      }));

    } catch (error) {
      setError(`Analysis failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Get chunks information
  const getChunksInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chunks`);
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      setChunks(result);
    } catch (error) {
      console.error('Failed to get chunks info:', error);
    }
  };

  // Handle RAG query (chat with document)
  const handleChatSubmit = async () => {
    if (!chatInput.trim()) return;

    if (!documentStats) {
      setError('Please upload and process a document first to ask questions');
      return;
    }

    const userMessage = { type: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    const currentInput = chatInput;
    setChatInput('');

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentInput,
          k: 5 // Number of chunks to retrieve
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      const aiMessage = { 
        type: 'ai', 
        content: result.answer,
        confidence: result.confidence,
        metadata: result.rag_metadata
      };
      setChatMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      const errorMessage = { 
        type: 'ai', 
        content: `Sorry, I encountered an error: ${error.message}` 
      };
      setChatMessages(prev => [...prev, errorMessage]);
    }
  };

  // Component initialization
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const TabButton = ({ id, icon: Icon, label, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
        isActive 
          ? 'bg-blue-600 text-white shadow-lg' 
          : 'text-gray-600 hover:bg-gray-100'
      }`}
    >
      <Icon size={18} />
      {label}
    </button>
  );

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-blue-900 via-blue-800 to-blue-600">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm border-b border-white/20 w-full">
        <div className="px-6 py-8 w-full">
          <div className="flex items-center justify-between w-full max-w-none">
            <div className="flex items-center gap-3">
              <Scale className="text-white" size={32} />
              <div>
                <h1 className="text-2xl font-bold text-white">VerdictRAG</h1>
                <p className="text-blue-100 text-sm">AI-Powered Legal Document Analysis with RAG</p>
              </div>
            </div>
            <div className="text-white text-right">
              <div className="flex items-center gap-2 mb-1">
                <Activity size={16} />
                <p className="text-sm">
                  {backendHealth ? 'Connected' : 'Disconnected'}
                </p>
              </div>
              {backendHealth && (
                <p className="text-xs opacity-75">{backendHealth.version}</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="px-6 py-8 w-full">
        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center gap-2">
            <XCircle className="text-red-500" size={20} />
            <span className="text-red-700">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              ×
            </button>
          </div>
        )}

        {/* Backend Status */}
        {backendHealth && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="text-green-500" size={20} />
              <span className="text-green-700 font-medium">RAG System Online</span>
            </div>
            <div className="text-sm text-green-600 grid grid-cols-2 md:grid-cols-4 gap-2">
              <div>Embeddings: {backendHealth.models.sentence_model ? '✅' : '❌'}</div>
              <div>QA Model: {backendHealth.models.qa_model ? '✅' : '❌'}</div>
              <div>Vector Store: {backendHealth.models.vector_store ? '✅' : '❌'}</div>
              <div>Chunks: {backendHealth.document_status.chunks_loaded}</div>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-6 mb-8 shadow-xl w-full">
          <div className="flex flex-wrap gap-2 mb-6 w-full">
            <TabButton 
              id="upload" 
              icon={Upload} 
              label="Upload Document" 
              isActive={activeTab === 'upload'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="analysis" 
              icon={FileText} 
              label="RAG Analysis" 
              isActive={activeTab === 'analysis'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="chat" 
              icon={MessageCircle} 
              label="Query Document" 
              isActive={activeTab === 'chat'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="chunks" 
              icon={Database} 
              label="Document Chunks" 
              isActive={activeTab === 'chunks'} 
              onClick={setActiveTab} 
            />
          </div>

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Upload Legal Document</h2>
              
              {/* File Upload */}
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors w-full">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept=".pdf"
                  className="hidden"
                />
                <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-lg text-gray-600 mb-2">Upload PDF file (Max 10MB)</p>
                <p className="text-sm text-gray-500 mb-4">Supports legal documents, case laws, contracts</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center gap-2 mx-auto"
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin" size={16} />
                      Processing with RAG...
                    </>
                  ) : (
                    'Choose PDF File'
                  )}
                </button>
                {uploadedFile && (
                  <p className="mt-2 text-green-600 font-medium">{uploadedFile.name}</p>
                )}
              </div>

              {/* Document Processing Status */}
              {documentStats && (
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-blue-800 mb-3 flex items-center gap-2">
                    <Database size={20} />
                    Document Processed with RAG
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-blue-600 font-medium">Characters</p>
                      <p className="text-blue-800">{documentStats.total_characters?.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-blue-600 font-medium">Words</p>
                      <p className="text-blue-800">{documentStats.total_words?.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-blue-600 font-medium">Chunks</p>
                      <p className="text-blue-800">{documentStats.total_chunks}</p>
                    </div>
                    <div>
                      <p className="text-blue-600 font-medium">Embeddings</p>
                      <p className="text-blue-800">{documentStats.embedding_dimension}D</p>
                    </div>
                  </div>
                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={handleComprehensiveAnalysis}
                      disabled={loading}
                      className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                      <Zap size={16} />
                      Comprehensive Analysis
                    </button>
                    <button
                      onClick={getChunksInfo}
                      className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2"
                    >
                      <Database size={16} />
                      View Chunks
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">RAG-Powered Analysis</h2>
              
              {loading && (
                <div className="text-center py-12">
                  <Loader2 className="animate-spin mx-auto text-blue-600 mb-4" size={48} />
                  <p className="text-gray-600">Performing comprehensive RAG analysis...</p>
                  <p className="text-gray-500 text-sm mt-2">Retrieving relevant chunks and generating insights</p>
                </div>
              )}

              {analysis?.comprehensiveAnalysis && !loading && (
                <div className="space-y-6 w-full">
                  {/* System Info */}
                  <div className="bg-blue-50 border-l-4 border-blue-600 p-6 rounded-r-xl w-full">
                    <h3 className="text-xl font-bold text-blue-900">{analysis.title}</h3>
                    <p className="text-blue-700 text-sm mt-1">
                      Analyzed by {analysis.system} • {analysis.documentStats?.total_chunks} chunks processed
                    </p>
                  </div>

                  {/* Analysis Results */}
                  <div className="space-y-4">
                    {Object.entries(analysis.comprehensiveAnalysis).map(([key, value]) => (
                      <div key={key} className="bg-white border border-gray-200 rounded-xl p-6 w-full">
                        <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                          <FileText size={20} />
                          {value.question}
                        </h4>
                        <p className="text-gray-700 leading-relaxed mb-3">{value.answer}</p>
                        <div className="flex items-center justify-between text-sm text-gray-500">
                          <span>Confidence: {(value.confidence * 100).toFixed(1)}%</span>
                          <span>Sources: {value.sources?.length || 0} chunks</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!analysis && !loading && (
                <div className="text-center py-12 text-gray-500">
                  <FileText size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Upload a legal document to see RAG-powered analysis</p>
                  <p className="text-sm mt-2">Supports PDF files with semantic search and retrieval</p>
                </div>
              )}
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Query Document with RAG</h2>
              
              {/* Chat Messages */}
              <div className="bg-gray-50 rounded-xl p-6 h-96 overflow-y-auto w-full">
                {chatMessages.length === 0 ? (
                  <div className="text-center text-gray-500 mt-20">
                    <MessageCircle size={48} className="mx-auto mb-4 opacity-50" />
                    <p>Ask questions about your legal document</p>
                    <p className="text-sm">RAG will retrieve relevant sections and provide contextual answers</p>
                    {!documentStats && (
                      <p className="text-sm text-orange-600 mt-2">Upload and process a document first</p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {chatMessages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                            message.type === 'user'
                              ? 'bg-blue-600 text-white'
                              : 'bg-white border border-gray-200 text-gray-800'
                          }`}
                        >
                          <p>{message.content}</p>
                          {message.confidence && (
                            <p className="text-xs mt-1 opacity-75">
                              Confidence: {(message.confidence * 100).toFixed(1)}% | 
                              Sources: {message.metadata?.source_chunks_used || 0}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div className="flex gap-3 w-full">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit()}
                  placeholder="Ask about the document... (e.g., 'What is the main issue?', 'Who are the parties involved?')"
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-full"
                />
                <button
                  onClick={handleChatSubmit}
                  disabled={!chatInput.trim() || !documentStats}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                >
                  Query
                </button>
              </div>
            </div>
          )}

          {/* Chunks Tab */}
          {activeTab === 'chunks' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Document Chunks</h2>
              
              {chunks ? (
                <div className="space-y-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-blue-600 font-medium">Total Chunks</p>
                        <p className="text-blue-800">{chunks.total_chunks}</p>
                      </div>
                      <div>
                        <p className="text-blue-600 font-medium">Chunk Size</p>
                        <p className="text-blue-800">{chunks.chunk_size} words</p>
                      </div>
                      <div>
                        <p className="text-blue-600 font-medium">Overlap</p>
                        <p className="text-blue-800">{chunks.chunk_overlap} words</p>
                      </div>
                      <div>
                        <p className="text-blue-600 font-medium">Showing</p>
                        <p className="text-blue-800">{chunks.sample_chunks?.length || 0} samples</p>
                      </div>
                    </div>
                  </div>

                  {chunks.sample_chunks?.map((chunk, index) => (
                    <div key={index} className="bg-white border border-gray-200 rounded-xl p-6">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-lg font-semibold text-gray-800">
                          Chunk {chunk.id}
                        </h4>
                        <div className="text-sm text-gray-500">
                          {chunk.word_count} words • {chunk.length} chars
                        </div>
                      </div>
                      <p className="text-gray-700 leading-relaxed">
                        {chunk.preview}
                      </p>
                    </div>
                  ))}
                </div>
              ) : documentStats ? (
                <div className="text-center py-12">
                  <button
                    onClick={getChunksInfo}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 mx-auto"
                  >
                    <Database size={20} />
                    Load Chunks Information
                  </button>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Database size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Document chunks will appear here after processing</p>
                  <p className="text-sm mt-2">Upload a document to see how it's chunked for RAG</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VerdictRAG;