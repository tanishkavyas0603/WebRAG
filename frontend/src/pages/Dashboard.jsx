import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bot, Link as LinkIcon, Loader2, ArrowRight, AlertCircle, CheckCircle2 } from 'lucide-react';
import { documentsApi } from '../api/documents';
import { conversationsApi } from '../api/conversations';
import clsx from 'clsx';

const STATUS_MESSAGES = {
  pending: "Pending ingestion...",
  fetching: "Fetching webpage...",
  extracting: "Extracting content...",
  chunking: "Creating chunks...",
  indexing: "Building search index...",
  ready: "Ready to chat",
  failed: "Ingestion failed"
};

export default function Dashboard() {
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState(null); // Document object
  const navigate = useNavigate();

  // Poll for document status
  useEffect(() => {
    if (!ingestStatus) return;

    let isMounted = true;
    let intervalId;

    if (ingestStatus.status === 'ready') {
      const transitionToChat = async () => {
        try {
          const conv = await conversationsApi.create(ingestStatus.id);
          if (isMounted) {
            navigate(`/app/chat/${conv.id}`);
            window.location.reload(); 
          }
        } catch (err) {
          if (isMounted) {
            console.error("Failed to start chat", err);
            setError("Failed to start chat session.");
            setLoading(false);
          }
        }
      };
      transitionToChat();
    } else if (ingestStatus.status !== 'failed') {
      intervalId = setInterval(async () => {
        try {
          const updatedDoc = await documentsApi.getStatus(ingestStatus.id);
          if (isMounted) {
            setIngestStatus(updatedDoc);
          }
        } catch (err) {
          if (isMounted) {
            console.error("Polling error", err);
            setError("Lost connection while checking status.");
            clearInterval(intervalId);
          }
        }
      }, 2000);
    }

    return () => {
      isMounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [ingestStatus, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    
    try {
      // Basic client-side validation
      new URL(url);
    } catch {
      setError("Please enter a valid URL including http:// or https://");
      return;
    }

    setError('');
    setLoading(true);
    setIngestStatus(null);

    try {
      const doc = await documentsApi.ingest(url);
      setIngestStatus(doc);
    } catch (err) {
      if (err.response?.status === 400 || err.response?.status === 422) {
        setError(err.response.data.detail || "Invalid or unsupported URL.");
      } else {
        setError("An error occurred while communicating with the server.");
      }
      setLoading(false);
    }
  };

  const exampleQuestions = [
    "Summarize this page",
    "What are the key points?",
    "Explain this in simple terms"
  ];

  return (
    <div className="h-full flex flex-col items-center justify-center p-4 md:p-8">
      <div className="w-full max-w-2xl mx-auto flex flex-col items-center text-center space-y-8">
        
        {/* Header */}
        <div className="space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-primary-50 rounded-2xl text-primary-600 mb-2">
            <Bot className="w-10 h-10" />
          </div>
          <h1 className="text-4xl font-bold text-slate-900 tracking-tight">WebRAG</h1>
          <p className="text-lg text-slate-600">Chat with any webpage.</p>
        </div>

        {/* URL Input Form */}
        {!ingestStatus && (
          <div className="w-full space-y-6">
            <p className="text-slate-500">Paste a public webpage and start asking questions.</p>
            
            <form onSubmit={handleSubmit} className="relative">
              <div className="relative flex items-center">
                <LinkIcon className="absolute left-4 w-5 h-5 text-slate-400" />
                <input
                  type="url"
                  required
                  placeholder="https://example.com/article"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={loading}
                  className="w-full pl-12 pr-32 py-4 bg-white border border-slate-200 rounded-2xl shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all text-slate-900 disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={loading || !url.trim()}
                  className="absolute right-2 top-2 bottom-2 px-6 flex items-center gap-2 bg-slate-900 text-white font-medium rounded-xl hover:bg-slate-800 disabled:opacity-50 transition-colors"
                >
                  {loading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <>Start Chat <ArrowRight className="w-4 h-4" /></>
                  )}
                </button>
              </div>
            </form>

            {error && (
              <div className="flex items-center justify-center gap-2 text-red-600 text-sm mt-4 bg-red-50 py-2 px-4 rounded-lg inline-flex">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}

            <div className="pt-8">
              <p className="text-sm font-medium text-slate-500 mb-4 uppercase tracking-wider">Example Questions</p>
              <div className="flex flex-wrap justify-center gap-3">
                {exampleQuestions.map((q, i) => (
                  <button 
                    key={i} 
                    className="px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-600 hover:bg-slate-100 transition-colors"
                    onClick={() => {}} // Disabled for dashboard, just visual
                    disabled
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Ingestion Progress */}
        {ingestStatus && (
          <div className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-6 shadow-sm text-left">
            <h3 className="font-semibold text-slate-900 mb-6 flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-primary-500" />
              Processing Webpage...
            </h3>
            
            <div className="space-y-4">
              {Object.keys(STATUS_MESSAGES).map((key) => {
                const statusOrder = ['pending', 'fetching', 'extracting', 'chunking', 'indexing', 'ready'];
                const currentIndex = statusOrder.indexOf(ingestStatus.status);
                const stepIndex = statusOrder.indexOf(key);
                
                if (key === 'failed' && ingestStatus.status !== 'failed') return null;
                if (ingestStatus.status === 'failed' && key !== 'failed' && stepIndex > statusOrder.indexOf('fetching')) return null;

                const isComplete = currentIndex > stepIndex;
                const isCurrent = currentIndex === stepIndex;
                const isFailed = ingestStatus.status === 'failed' && key === 'failed';

                if (stepIndex > currentIndex && !isFailed && ingestStatus.status !== 'failed') {
                   // Show pending steps faintly
                   return (
                     <div key={key} className="flex items-center gap-3 text-slate-300">
                       <div className="w-5 h-5 rounded-full border-2 border-slate-200 shrink-0" />
                       <span className="text-sm">{STATUS_MESSAGES[key]}</span>
                     </div>
                   );
                }

                return (
                  <div 
                    key={key} 
                    className={clsx(
                      "flex items-center gap-3",
                      isFailed ? "text-red-600" : (isComplete || isCurrent ? "text-slate-700" : "text-slate-300")
                    )}
                  >
                    {isFailed ? (
                      <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
                    ) : isComplete ? (
                      <CheckCircle2 className="w-5 h-5 text-primary-500 shrink-0" />
                    ) : isCurrent ? (
                      <Loader2 className="w-5 h-5 animate-spin text-primary-500 shrink-0" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-slate-200 shrink-0" />
                    )}
                    <span className={clsx("text-sm", isCurrent && "font-medium")}>
                      {isFailed && ingestStatus.error_message ? ingestStatus.error_message : STATUS_MESSAGES[key]}
                    </span>
                  </div>
                );
              })}
            </div>
            
            {ingestStatus.status === 'failed' && (
              <button 
                onClick={() => { setIngestStatus(null); setLoading(false); setError(''); }}
                className="mt-6 w-full py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
              >
                Try Again
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
