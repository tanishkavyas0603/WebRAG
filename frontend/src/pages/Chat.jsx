import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { conversationsApi } from '../api/conversations';
import { documentsApi } from '../api/documents';
import { Send, Loader2, ExternalLink, Bot, User, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import clsx from 'clsx';
import { formatDistanceToNow } from 'date-fns';

const parseUtcDate = (dateStr) => {
  if (!dateStr) return new Date();
  return new Date(dateStr.endsWith('Z') ? dateStr : dateStr + 'Z');
};

const getCleanTitle = (title) => {
  return (title || '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*/gi, '')
    .trim() || 'Untitled';
};

const cleanMessageContent = (content) => {
  if (!content) return '';
  return content
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*/gi, '')
    .trim();
};

export default function Chat() {
  const { id } = useParams();
  const [conversation, setConversation] = useState(null);
  const [document, setDocument] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    fetchConversation();
  }, [id]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchConversation = async () => {
    setLoading(true);
    setError('');
    try {
      const conversationData = await conversationsApi.get(id);
      setConversation(conversationData);
      
      if (conversationData.document_id) {
        try {
          const docData = await documentsApi.getStatus(conversationData.document_id);
          setDocument(docData);
        } catch (docErr) {
          console.error("Failed to load document", docErr);
        }
      }
      
      // Also fetch messages
      const msgs = await conversationsApi.getMessages(id);
      setMessages(msgs);
    } catch (err) {
      if (err.response?.status === 404) {
        setError('Conversation not found.');
      } else {
        setError('Failed to load conversation.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async (e) => {
    e?.preventDefault();
    if (!input.trim() || sending) return;

    const messageText = input;
    setInput('');
    setSending(true);
    setError('');

    // Optimistically add user message
    const tempUserMsg = { id: Date.now(), role: 'user', content: messageText, created_at: new Date().toISOString() };
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const assistantMessage = await conversationsApi.sendMessage(id, messageText);
      // Re-fetch all messages to get exact state or append assistant msg
      // For safety, re-fetch all
      const msgs = await conversationsApi.getMessages(id);
      setMessages(msgs);
    } catch (err) {
      setError('Sorry, I couldn\'t generate an answer right now. Please try again.');
      // Revert optimistic update on fail
      setMessages(prev => prev.filter(m => m.id !== tempUserMsg.id));
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-white">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (error || !conversation) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-white p-4">
        <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
        <h2 className="text-xl font-semibold text-slate-900 mb-2">{error || "Something went wrong"}</h2>
        <p className="text-slate-500">Please try selecting a conversation from the sidebar.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white relative">
      
      {/* Header */}
      <header className="h-16 border-b border-slate-200 bg-white/80 backdrop-blur-sm flex items-center justify-between px-6 shrink-0 sticky top-0 z-10 hidden lg:flex">
        <div className="flex flex-col overflow-hidden mr-4">
          <h2 className="font-semibold text-slate-900 truncate">
            {document?.title || getCleanTitle(conversation.title)}
          </h2>
          <p className="text-xs text-slate-500 truncate mt-0.5 flex items-center gap-2">
            <span className="bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded font-medium">Single Page Scope</span>
            <span>Chatting with this webpage ({(() => {
              try { return new URL(document?.url || 'https://example.com').hostname; }
              catch(e) { return 'web'; }
            })()})</span>
          </p>
        </div>
        <a 
          href={document?.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-lg transition-colors shrink-0"
        >
          <ExternalLink className="w-4 h-4" />
          Open Webpage
        </a>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4">
            <Bot className="w-12 h-12 text-slate-300" />
            <p>Ask a question about this webpage.</p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isUser = msg.role === 'user';
            
            return (
              <div key={msg.id || idx} className={clsx("flex gap-4 max-w-4xl mx-auto w-full", isUser ? "flex-row-reverse" : "")}>
                
                {/* Avatar */}
                <div className={clsx(
                  "w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1",
                  isUser ? "bg-slate-900 text-white" : "bg-primary-100 text-primary-600"
                )}>
                  {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                </div>

                {/* Message Content */}
                <div className={clsx(
                  "flex flex-col gap-2 max-w-[85%]",
                  isUser ? "items-end" : "items-start"
                )}>
                  <div className={clsx(
                    "px-4 py-3 rounded-2xl",
                    isUser ? "bg-slate-900 text-white rounded-tr-sm" : "bg-slate-50 border border-slate-100 text-slate-800 rounded-tl-sm"
                  )}>
                    {isUser ? (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div className="prose prose-sm md:prose-base prose-slate max-w-none">
                        <ReactMarkdown>
                          {cleanMessageContent(msg.content)}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                  
                  {/* Citations */}
                  {!isUser && msg.citations && msg.citations.length > 0 && (
                    <div className="w-full mt-4 space-y-3">
                      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider pl-1">Sources</p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
                        {msg.citations.map((cite, i) => {
                          let hostname = cite.source;
                          try {
                            hostname = new URL(cite.source).hostname;
                          } catch(e) {}
                          
                          return (
                            <div key={i} className="bg-white border border-slate-200 rounded-xl p-3 flex flex-col gap-1 shadow-sm">
                              <h4 className="font-semibold text-sm text-slate-900 line-clamp-1">{cite.title}</h4>
                              <p className="text-xs text-slate-500 truncate">{hostname}</p>
                              <a 
                                href={cite.source} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="text-xs font-medium text-primary-600 hover:text-primary-700 flex items-center mt-2 pt-2 border-t border-slate-100"
                              >
                                Open webpage <ExternalLink className="w-3 h-3 ml-1" />
                              </a>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  
                  <span className="text-[10px] text-slate-400 mt-1 px-1">
                    {formatDistanceToNow(parseUtcDate(msg.created_at), { addSuffix: true })}
                  </span>
                </div>
              </div>
            );
          })
        )}
        
        {sending && (
          <div className="flex gap-4 max-w-4xl mx-auto w-full">
            <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center shrink-0 mt-1">
              <Bot className="w-5 h-5" />
            </div>
            <div className="bg-slate-50 border border-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2 text-slate-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm font-medium">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 sm:p-6 bg-white border-t border-slate-100 shrink-0">
        <div className="max-w-4xl mx-auto relative">
          <form onSubmit={handleSend} className="relative flex items-end gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question..."
              disabled={sending}
              rows={1}
              className="w-full max-h-[200px] bg-slate-50 border border-slate-200 rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none text-slate-900 disabled:opacity-50"
              style={{ minHeight: '48px' }}
            />
            <button
              type="submit"
              disabled={!input.trim() || sending}
              className="absolute right-2 bottom-2 p-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg disabled:opacity-50 disabled:hover:bg-primary-500 transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-center mt-2">
            <span className="text-[10px] text-slate-400">Press Enter to send, Shift + Enter for new line.</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Small helper since I imported ArrowRight but didn't put it in the import list
import { ArrowRight } from 'lucide-react';
