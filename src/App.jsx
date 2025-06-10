import React, { useState, useRef } from 'react';
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
  XCircle
} from 'lucide-react';

const VerdictRAG = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [caseUrl, setCaseUrl] = useState('');
  const [caseText, setCaseText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const fileInputRef = useRef(null);

  // Mock API calls - replace with actual Flask backend calls
  const uploadFile = async (file) => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setAnalysis({
        title: "Brown v. Board of Education of Topeka",
        summary: "This landmark case declared state laws establishing separate public schools for black and white students to be unconstitutional, overturning the 'separate but equal' doctrine.",
        verdict: "The Supreme Court unanimously ruled that segregated public schools are unconstitutional under the Equal Protection Clause of the 14th Amendment.",
        implications: [
          "Ended legal segregation in public schools",
          "Paved the way for the Civil Rights Movement",
          "Established precedent for challenging discriminatory laws"
        ],
        legalTerms: [
          { term: "Equal Protection Clause", definition: "Part of the 14th Amendment that requires states to provide equal protection under the law to all people" },
          { term: "Precedent", definition: "A legal principle established in a previous court case that guides future decisions" }
        ]
      });
      setLoading(false);
    }, 2000);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      uploadFile(file);
    }
  };

  const handleUrlSubmit = async () => {
    if (caseUrl) {
      setLoading(true);
      // Simulate API call for URL processing
      setTimeout(() => {
        setAnalysis({
          title: "Case from URL: " + caseUrl,
          summary: "Analysis of the case from the provided URL...",
          verdict: "Court ruling extracted from the document...",
          implications: ["Implication 1", "Implication 2"],
          legalTerms: [{ term: "Sample Term", definition: "Sample definition" }]
        });
        setLoading(false);
      }, 2000);
    }
  };

  const handleTextSubmit = async () => {
    if (caseText) {
      setLoading(true);
      // Simulate API call for text processing
      setTimeout(() => {
        setAnalysis({
          title: "Direct Text Analysis",
          summary: "Analysis of the provided case text...",
          verdict: "Court ruling extracted from the text...",
          implications: ["Implication 1", "Implication 2"],
          legalTerms: [{ term: "Sample Term", definition: "Sample definition" }]
        });
        setLoading(false);
      }, 2000);
    }
  };

  const handleChatSubmit = async () => {
    if (chatInput.trim()) {
      const userMessage = { type: 'user', content: chatInput };
      setChatMessages(prev => [...prev, userMessage]);
      setChatInput('');
      
      // Simulate AI response
      setTimeout(() => {
        const aiMessage = { 
          type: 'ai', 
          content: `Based on the case analysis, here's what I found regarding your question: "${chatInput}". This relates to the legal principles established in the case...` 
        };
        setChatMessages(prev => [...prev, aiMessage]);
      }, 1500);
    }
  };

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
                <p className="text-blue-100 text-sm">AI-Powered Case Law Explainer</p>
              </div>
            </div>
            <div className="text-white text-right">
              <p className="text-sm opacity-75">Making Legal Judgments</p>
              <p className="text-sm opacity-75">Accessible to Everyone</p>
            </div>
          </div>
        </div>
      </div>

      <div className="px-6 py-8 w-full">
        {/* Navigation Tabs */}
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-6 mb-8 shadow-xl w-full">
          <div className="flex flex-wrap gap-2 mb-6 w-full">
            <TabButton 
              id="upload" 
              icon={Upload} 
              label="Upload Case" 
              isActive={activeTab === 'upload'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="search" 
              icon={Search} 
              label="Search Cases" 
              isActive={activeTab === 'search'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="analysis" 
              icon={FileText} 
              label="Case Analysis" 
              isActive={activeTab === 'analysis'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="chat" 
              icon={MessageCircle} 
              label="Ask Questions" 
              isActive={activeTab === 'chat'} 
              onClick={setActiveTab} 
            />
            <TabButton 
              id="glossary" 
              icon={Book} 
              label="Legal Terms" 
              isActive={activeTab === 'glossary'} 
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
                  accept=".pdf,.doc,.docx,.txt"
                  className="hidden"
                />
                <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-lg text-gray-600 mb-2">Upload PDF, DOC, or TXT file</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Choose File
                </button>
                {uploadedFile && (
                  <p className="mt-2 text-green-600 font-medium">{uploadedFile.name}</p>
                )}
              </div>

              {/* URL Input */}
              <div className="bg-gray-50 rounded-xl p-6 w-full">
                <h3 className="text-lg font-semibold mb-3">Or enter case URL</h3>
                <div className="flex gap-3 w-full">
                  <input
                    type="url"
                    value={caseUrl}
                    onChange={(e) => setCaseUrl(e.target.value)}
                    placeholder="https://example.com/case-document"
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-full"
                  />
                  <button
                    onClick={handleUrlSubmit}
                    className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors flex-shrink-0"
                  >
                    Process URL
                  </button>
                </div>
              </div>

              {/* Text Input */}
              <div className="bg-gray-50 rounded-xl p-6 w-full">
                <h3 className="text-lg font-semibold mb-3">Or paste case text directly</h3>
                <textarea
                  value={caseText}
                  onChange={(e) => setCaseText(e.target.value)}
                  rows={6}
                  placeholder="Paste the legal document text here..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  onClick={handleTextSubmit}
                  className="mt-3 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Analyze Text
                </button>
              </div>
            </div>
          )}

          {/* Search Tab */}
          {activeTab === 'search' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Search Case Database</h2>
              <div className="flex gap-3 w-full">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for cases, keywords, or legal concepts..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-full"
                />
                <button className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors flex-shrink-0">
                  <Search size={20} />
                </button>
              </div>
              <div className="text-center text-gray-500 py-8">
                Enter a search term to find relevant cases
              </div>
            </div>
          )}

          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Case Analysis</h2>
              
              {loading && (
                <div className="text-center py-12">
                  <Loader2 className="animate-spin mx-auto text-blue-600 mb-4" size={48} />
                  <p className="text-gray-600">Analyzing case document...</p>
                </div>
              )}

              {analysis && !loading && (
                <div className="space-y-6 w-full">
                  {/* Case Title */}
                  <div className="bg-blue-50 border-l-4 border-blue-600 p-6 rounded-r-xl w-full">
                    <h3 className="text-xl font-bold text-blue-900">{analysis.title}</h3>
                  </div>

                  {/* Summary */}
                  <div className="bg-white border border-gray-200 rounded-xl p-6 w-full">
                    <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                      <FileText size={20} />
                      Case Summary
                    </h4>
                    <p className="text-gray-700 leading-relaxed">{analysis.summary}</p>
                  </div>

                  {/* Verdict */}
                  <div className="bg-green-50 border border-green-200 rounded-xl p-6 w-full">
                    <h4 className="text-lg font-semibold text-green-800 mb-3 flex items-center gap-2">
                      <CheckCircle size={20} />
                      Court Verdict
                    </h4>
                    <p className="text-green-700 leading-relaxed">{analysis.verdict}</p>
                  </div>

                  {/* Implications */}
                  <div className="bg-orange-50 border border-orange-200 rounded-xl p-6 w-full">
                    <h4 className="text-lg font-semibold text-orange-800 mb-3 flex items-center gap-2">
                      <AlertCircle size={20} />
                      Real-World Implications
                    </h4>
                    <ul className="space-y-2">
                      {analysis.implications.map((implication, index) => (
                        <li key={index} className="text-orange-700 flex items-start gap-2">
                          <span className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></span>
                          {implication}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {!analysis && !loading && (
                <div className="text-center py-12 text-gray-500">
                  Upload or process a case document to see the analysis
                </div>
              )}
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Ask Questions About the Case</h2>
              
              {/* Chat Messages */}
              <div className="bg-gray-50 rounded-xl p-6 h-96 overflow-y-auto w-full">
                {chatMessages.length === 0 ? (
                  <div className="text-center text-gray-500 mt-20">
                    <MessageCircle size={48} className="mx-auto mb-4 opacity-50" />
                    <p>Ask any questions about the legal case</p>
                    <p className="text-sm">Examples: "What was the main issue?", "How does this affect citizens?"</p>
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
                          {message.content}
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
                  placeholder="Ask a question about the case..."
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-full"
                />
                <button
                  onClick={handleChatSubmit}
                  disabled={!chatInput.trim()}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                >
                  Send
                </button>
              </div>
            </div>
          )}

          {/* Glossary Tab */}
          {activeTab === 'glossary' && (
            <div className="space-y-6 w-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Legal Terms Explained</h2>
              
              {analysis?.legalTerms ? (
                <div className="space-y-4 w-full">
                  {analysis.legalTerms.map((term, index) => (
                    <div key={index} className="bg-white border border-gray-200 rounded-xl p-6 w-full">
                      <h4 className="text-lg font-semibold text-blue-900 mb-2">{term.term}</h4>
                      <p className="text-gray-700">{term.definition}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Book size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Legal terms will appear here after analyzing a case</p>
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