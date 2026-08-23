import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Bot, LogOut, Plus, Search, MessageSquare, Trash2, Edit2, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useAuth } from '../../context/AuthContext';
import { conversationsApi } from '../../api/conversations';
import clsx from 'clsx';

// Helpers
const getCleanTitle = (title) => {
  return (title || '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*/gi, '')
    .trim() || 'Untitled';
};

const parseUtcDate = (dateStr) => {
  if (!dateStr) return new Date();
  return new Date(dateStr.endsWith('Z') ? dateStr : dateStr + 'Z');
};

export default function Sidebar({ isOpen, setIsOpen }) {
  const [conversations, setConversations] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    try {
      const data = await conversationsApi.list();
      setConversations(data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch conversations', err);
      setError('Unable to load conversations.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e, id) => {
    e.preventDefault();
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      try {
        await conversationsApi.delete(id);
        setConversations(conversations.filter(c => c.id !== id));
        navigate('/app');
      } catch (err) {
        console.error('Failed to delete', err);
      }
    }
  };

  const filteredConversations = conversations.filter(c => 
    getCleanTitle(c.title).toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-slate-900/50 backdrop-blur-sm lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar container */}
      <div className={clsx(
        "fixed inset-y-0 left-0 z-50 flex w-72 flex-col bg-white border-r border-slate-200 transition-transform duration-300 ease-in-out lg:static lg:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        
        {/* Header */}
        <div className="flex h-16 shrink-0 items-center justify-between px-4 border-b border-slate-100">
          <Link to="/app" className="flex items-center gap-2 text-slate-900 font-semibold" onClick={() => setIsOpen(false)}>
            <div className="bg-primary-100 p-1.5 rounded-lg text-primary-600">
              <Bot className="w-5 h-5" />
            </div>
            WebRAG
          </Link>
          <button onClick={() => setIsOpen(false)} className="lg:hidden p-2 text-slate-500 hover:text-slate-700">
            <PanelLeftClose className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <Link 
            to="/app"
            onClick={() => setIsOpen(false)}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800 transition-colors shadow-sm"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </Link>
        </div>

        {/* Search */}
        <div className="px-4 pb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input 
              type="text"
              placeholder="Search conversations..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
            />
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-1">
          {loading ? (
             <div className="px-4 py-3 text-sm text-slate-500">Loading conversations...</div>
          ) : error ? (
             <div className="px-4 py-3 text-sm text-red-500">{error}</div>
          ) : filteredConversations.length === 0 ? (
             <div className="px-4 py-8 text-center flex flex-col items-center">
               <MessageSquare className="w-8 h-8 text-slate-300 mb-3" />
               <p className="text-sm font-medium text-slate-700">
                 {search ? "No matching conversations" : "No conversations yet"}
               </p>
               <p className="text-xs text-slate-500 mt-1">
                 {search ? "Try adjusting your search term." : "Start a new chat to get started."}
               </p>
             </div>
          ) : (
            filteredConversations.map(conv => (
              <Link
                key={conv.id}
                to={`/app/chat/${conv.id}`}
                onClick={() => setIsOpen(false)}
                className="group flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <div className="flex items-center gap-3 overflow-hidden">
                  <MessageSquare className="w-4 h-4 text-slate-400 shrink-0" />
                  <div className="flex flex-col overflow-hidden">
                    <span className="truncate font-medium text-sm">{getCleanTitle(conv.title)}</span>
                    <span className="truncate text-xs text-slate-500 mt-0.5">
                      {conv.document_url ? (
                        (() => {
                          try { return new URL(conv.document_url).hostname; }
                          catch(e) { return 'web'; }
                        })()
                      ) : 'web'}
                    </span>
                    <span className="truncate text-[10px] text-slate-400 mt-0.5">
                      {formatDistanceToNow(parseUtcDate(conv.created_at), { addSuffix: true })}
                    </span>
                  </div>
                </div>
                <button 
                  onClick={(e) => handleDelete(e, conv.id)}
                  className="opacity-0 group-hover:opacity-100 p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-all shrink-0"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </Link>
            ))
          )}
        </div>

        {/* User / Footer */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/50 mt-auto">
          <button 
            onClick={logout}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-700 hover:bg-slate-200 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign out
          </button>
        </div>
      </div>
    </>
  );
}
