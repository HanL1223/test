/**
 * ============================================================================
 * Jira Ticket Generator - React Frontend
 * ============================================================================
 * 
 * A clean, modern interface for generating Jira tickets using the RAG API.
 * 
 * FEATURES:
 * - Natural language ticket input
 * - Fast/Full mode toggle
 * - Generated ticket display
 * - Copy to clipboard
 * - Retrieved context viewer
 * 
 * ============================================================================
 */

import React, { useState } from 'react';
import axios from 'axios';

// API base URL - use proxy in development
const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * Main App Component
 */
function App() {
  // State
  const [request, setRequest] = useState('');
  const [fastMode, setFastMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showContext, setShowContext] = useState(false);

  /**
   * Generate a ticket via the API.
   */
  const handleGenerate = async () => {
    if (!request.trim()) {
      setError('Please enter a ticket description');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setCopied(false);

    try {
      const response = await axios.post(`${API_BASE}/api/generate`, {
        request: request.trim(),
        fast_mode: fastMode,
      });
      
      setResult(response.data);
    } catch (err) {
      console.error('Generation failed:', err);
      setError(
        err.response?.data?.detail || 
        'Failed to generate ticket. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  /**
   * Copy ticket to clipboard.
   */
  const handleCopy = async () => {
    if (!result?.ticket_text) return;
    
    try {
      await navigator.clipboard.writeText(result.ticket_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };

  /**
   * Handle Enter key to submit.
   */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.metaKey) {
      handleGenerate();
    }
  };

  // Styles
  const styles = {
    container: {
      maxWidth: '900px',
      margin: '0 auto',
      padding: '40px 20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    header: {
      textAlign: 'center',
      marginBottom: '40px',
    },
    title: {
      fontSize: '2rem',
      fontWeight: '600',
      color: '#1a1a1a',
      marginBottom: '8px',
    },
    subtitle: {
      fontSize: '1rem',
      color: '#666',
    },
    card: {
      backgroundColor: '#fff',
      borderRadius: '12px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
      padding: '24px',
      marginBottom: '24px',
    },
    textarea: {
      width: '100%',
      minHeight: '120px',
      padding: '16px',
      fontSize: '1rem',
      border: '1px solid #e0e0e0',
      borderRadius: '8px',
      resize: 'vertical',
      fontFamily: 'inherit',
      outline: 'none',
      transition: 'border-color 0.2s',
    },
    controls: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginTop: '16px',
      gap: '16px',
    },
    toggle: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      cursor: 'pointer',
      userSelect: 'none',
    },
    checkbox: {
      width: '18px',
      height: '18px',
      cursor: 'pointer',
    },
    button: {
      padding: '12px 32px',
      fontSize: '1rem',
      fontWeight: '500',
      color: '#fff',
      backgroundColor: '#1a73e8',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      transition: 'background-color 0.2s',
    },
    buttonDisabled: {
      backgroundColor: '#93c5fd',
      cursor: 'not-allowed',
    },
    error: {
      backgroundColor: '#fef2f2',
      color: '#dc2626',
      padding: '12px 16px',
      borderRadius: '8px',
      marginBottom: '24px',
    },
    resultHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '16px',
    },
    resultTitle: {
      fontSize: '1.1rem',
      fontWeight: '600',
      color: '#1a1a1a',
    },
    copyButton: {
      padding: '8px 16px',
      fontSize: '0.875rem',
      color: copied ? '#16a34a' : '#1a73e8',
      backgroundColor: copied ? '#f0fdf4' : '#eff6ff',
      border: 'none',
      borderRadius: '6px',
      cursor: 'pointer',
    },
    ticketBox: {
      backgroundColor: '#f8f9fa',
      padding: '20px',
      borderRadius: '8px',
      fontFamily: 'monospace',
      fontSize: '0.9rem',
      whiteSpace: 'pre-wrap',
      lineHeight: '1.6',
      maxHeight: '400px',
      overflow: 'auto',
    },
    metadata: {
      display: 'flex',
      gap: '16px',
      marginTop: '16px',
      fontSize: '0.875rem',
      color: '#666',
    },
    metaItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '4px',
    },
    contextToggle: {
      marginTop: '16px',
      padding: '8px 16px',
      fontSize: '0.875rem',
      color: '#666',
      backgroundColor: 'transparent',
      border: '1px solid #e0e0e0',
      borderRadius: '6px',
      cursor: 'pointer',
    },
    contextBox: {
      marginTop: '16px',
      padding: '16px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
    },
    contextItem: {
      marginBottom: '12px',
      padding: '12px',
      backgroundColor: '#fff',
      borderRadius: '6px',
      border: '1px solid #e0e0e0',
    },
    contextKey: {
      fontWeight: '600',
      color: '#1a73e8',
      marginBottom: '4px',
    },
    contextScore: {
      fontSize: '0.75rem',
      color: '#666',
    },
    contextText: {
      fontSize: '0.875rem',
      color: '#333',
      marginTop: '8px',
    },
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>🎫 Jira Ticket Generator</h1>
        <p style={styles.subtitle}>
          Generate professional Jira tickets using AI-powered RAG
        </p>
      </header>

      {/* Input Card */}
      <div style={styles.card}>
        <textarea
          style={styles.textarea}
          placeholder="Describe the ticket you want to create...&#10;&#10;Example: Create a ticket for implementing user authentication with OAuth2"
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        
        <div style={styles.controls}>
          <label style={styles.toggle}>
            <input
              type="checkbox"
              style={styles.checkbox}
              checked={fastMode}
              onChange={(e) => setFastMode(e.target.checked)}
              disabled={loading}
            />
            <span>Fast mode (skip refinement)</span>
          </label>
          
          <button
            style={{
              ...styles.button,
              ...(loading ? styles.buttonDisabled : {}),
            }}
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? 'Generating...' : 'Generate Ticket'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={styles.error}>
          {error}
        </div>
      )}

      {/* Result Card */}
      {result && (
        <div style={styles.card}>
          <div style={styles.resultHeader}>
            <span style={styles.resultTitle}>Generated Ticket</span>
            <button style={styles.copyButton} onClick={handleCopy}>
              {copied ? '✓ Copied!' : 'Copy'}
            </button>
          </div>
          
          <div style={styles.ticketBox}>
            {result.ticket_text}
          </div>
          
          <div style={styles.metadata}>
            <span style={styles.metaItem}>
              📊 Style: {result.style_detected}
            </span>
            <span style={styles.metaItem}>
              🔧 Refined: {result.refinement_applied ? 'Yes' : 'No'}
            </span>
            <span style={styles.metaItem}>
              ⏱️ Time: {result.metadata?.elapsed_seconds?.toFixed(2)}s
            </span>
            <span style={styles.metaItem}>
              📚 Context: {result.retrieved_chunks?.length || 0} tickets
            </span>
          </div>
          
          {/* Context Toggle */}
          {result.retrieved_chunks?.length > 0 && (
            <>
              <button
                style={styles.contextToggle}
                onClick={() => setShowContext(!showContext)}
              >
                {showContext ? '▼ Hide' : '▶ Show'} Retrieved Context
              </button>
              
              {showContext && (
                <div style={styles.contextBox}>
                  {result.retrieved_chunks.map((chunk, i) => (
                    <div key={i} style={styles.contextItem}>
                      <div style={styles.contextKey}>
                        {chunk.issue_key}
                        <span style={styles.contextScore}>
                          {' '}(score: {chunk.score?.toFixed(3)})
                        </span>
                      </div>
                      <div style={styles.contextText}>
                        {chunk.text}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Footer */}
      <footer style={{ textAlign: 'center', marginTop: '40px', color: '#666', fontSize: '0.875rem' }}>
        Powered by LangChain + Gemini • Built with ❤️ for better ticket creation
      </footer>
    </div>
  );
}

export default App;
