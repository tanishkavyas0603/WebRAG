import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bot, Link as LinkIcon, Loader2, ArrowRight, AlertCircle, CheckCircle2 } from 'lucide-react';
import { documentsApi } from '../api/documents';
import { conversationsApi } from '../api/conversations';
import clsx from 'clsx';

// BUG-01 FIX: statusOrder must match ACTUAL backend status values.
// Backend only ever returns: 'pending', 'processing', 'ready', 'failed'
// The previous statusOrder included 'fetching', 'extracting', 'chunking', 'indexing'
// which NEVER appear in the DB. When status was 'processing', indexOf returned -1,
// causing ALL steps to show as unfilled circles (broken progress UI).
const STATUS_STEPS = [
  { key: 'pending',    label: 'Received — queued for processing...' },
  { key: 'processing', label: 'Processing webpage...' },
  { key: 'ready',      label: 'Ready to chat' },
];

// BUG-07: Maximum polling attempts before declaring timeout (150 × 2s = 5 minutes)
const MAX_POLL_ATTEMPTS = 150;

export default function Dashboard() {
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState(null); // Document object
  const [pollCount, setPollCount] = useState(0);
  const navigate = useNavigate();

  // Poll for document status
  useEffect(() => {
    if (!ingestStatus) return;

    let isMounted = true;
    let intervalId;

    if (ingestStatus.status === 'ready') {
      // Document is ready — create a conversation and navigate
      const transitionToChat = async () => {
        try {
          const conv = await conversationsApi.create(ingestStatus.id);
          if (isMounted) {
            // BUG-05 FIX: Removed window.location.reload() — it was a full browser
            // reload that destroyed React state and caused a jarring user experience.
            // The sidebar re-fetches automatically on route change via its own useEffect.
            navigate(`/app/chat/${conv.id}`);
          }
        } catch (err) {
          if (isMounted) {
            console.error("Failed to start chat", err);
            setError("Failed to start chat session. Please try again.");
            setLoading(false);
          }
        }
      };
      transitionToChat();
    } else if (ingestStatus.status === 'failed') {
      // Stopped — do not set up polling
      setLoading(false);
    } else {
      // BUG-07 FIX: Cap polling at MAX_POLL_ATTEMPTS to prevent infinite loops
      // when a document gets stuck in 'pending' or 'processing' after a server restart.
      if (pollCount >= MAX_POLL_ATTEMPTS) {
        if (isMounted) {
          setError(
            "The webpage is taking longer than expected to process. " +
            "This may happen after a server restart. Please try submitting the URL again."
          );
          setLoading(false);
        }
        return;
      }

      intervalId = setInterval(async () => {
        try {
          const updatedDoc = await documentsApi.getStatus(ingestStatus.id);
          if (isMounted) {
            setIngestStatus(updatedDoc);
            setPollCount(prev => prev + 1);
          }
        } catch (err) {
          if (isMounted) {
            console.error("Polling error", err);
            setError("Lost connection while checking status.");
            clearInterval(intervalId);
            setLoading(false);
          }
        }
      }, 2000);
    }

    return () => {
      isMounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [ingestStatus, navigate, pollCount]);

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
    setPollCount(0);

    try {
      const doc = await documentsApi.ingest(url);
      setIngestStatus(doc);
    } catch (err) {
      if (err.response?.status === 400 || err.response?.status === 422) {
        setError(err.response.data.detail || "Invalid or unsupported URL.");
      } else if (err.response?.status === 429) {
        setError("Too many requests. Please wait a moment and try again.");
      } else {
        setError("An error occurred while communicating with the server.");
      }
      setLoading(false);
    }
  };

  const handleReset = () => {
    setIngestStatus(null);
    setLoading(false);
    setError('');
    setPollCount(0);
  };

  const exampleQuestions = [
    "Summarize this page",
    "What are the key points?",
    "Explain this in simple terms"
  ];

  // BUG-01 FIX: Correctly determine step state based on actual backend status values.
  const getStepState = (stepKey, currentStatus) => {
    const order = ['pending', 'processing', 'ready'];
    const currentIdx = order.indexOf(currentStatus);
    const stepIdx = order.indexOf(stepKey);

    if (currentStatus === 'failed') {
      return stepIdx === 0 ? 'complete' : 'failed';
    }
    if (currentIdx === -1) return 'pending'; // Unknown status — show as pending
    if (stepIdx < currentIdx) return 'complete';
    if (stepIdx === currentIdx) return 'current';
    return 'pending';
  };

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
                    onClick={() => {}} // Visual only on dashboard
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
        {/* BUG-01 FIX: Status steps now correctly reflect backend states */}
        {ingestStatus && ingestStatus.status !== 'failed' && (
          <div className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-6 shadow-sm text-left">
            <h3 className="font-semibold text-slate-900 mb-6 flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-primary-500" />
              Processing Webpage...
            </h3>
            
            <div className="space-y-4">
              {STATUS_STEPS.map(({ key, label }) => {
                const state = getStepState(key, ingestStatus.status);

                return (
                  <div
                    key={key}
                    className={clsx(
                      "flex items-center gap-3",
                      state === 'complete' || state === 'current' ? "text-slate-700" : "text-slate-300"
                    )}
                  >
                    {state === 'complete' ? (
                      <CheckCircle2 className="w-5 h-5 text-primary-500 shrink-0" />
                    ) : state === 'current' ? (
                      <Loader2 className="w-5 h-5 animate-spin text-primary-500 shrink-0" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-slate-200 shrink-0" />
                    )}
                    <span className={clsx("text-sm", state === 'current' && "font-medium")}>
                      {label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Failed State */}
        {ingestStatus && ingestStatus.status === 'failed' && (
          <div className="w-full max-w-md bg-white border border-red-200 rounded-2xl p-6 shadow-sm text-left">
            <h3 className="font-semibold text-red-700 mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
              Unable to process this webpage
            </h3>
            {ingestStatus.error_message && (
              <p className="text-sm text-slate-600 mb-4">{ingestStatus.error_message}</p>
            )}
            <button
              onClick={handleReset}
              className="w-full py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Error message (non-document errors) */}
        {error && !ingestStatus && (
          <div className="flex items-center gap-2 text-red-600 text-sm bg-red-50 py-2 px-4 rounded-lg">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Timeout error with reset */}
        {error && ingestStatus && (
          <div className="w-full max-w-md space-y-3">
            <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50 py-3 px-4 rounded-lg">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
            <button
              onClick={handleReset}
              className="w-full py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
