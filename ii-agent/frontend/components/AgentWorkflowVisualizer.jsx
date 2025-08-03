import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const AgentWorkflowVisualizer = ({ sessionId }) => {
  const [thoughtTrail, setThoughtTrail] = useState([]);
  const [agentStatus, setAgentStatus] = useState('idle');
  
  const wsUrl = sessionId ? `ws://localhost:9000/ws/agent/${sessionId}` : null;
  
  const { 
    connectionStatus, 
    lastMessage, 
    sendMessage,
    error 
  } = useWebSocket(wsUrl);

  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'connected':
          // Request initial status
          sendMessage({ type: 'get_status' });
          sendMessage({ type: 'get_thoughts' });
          break;
          
        case 'status_update':
          setAgentStatus(lastMessage.status);
          break;
          
        case 'thought_trail':
          setThoughtTrail(lastMessage.thoughts || []);
          break;
          
        case 'agent_completed':
          setAgentStatus('completed');
          sendMessage({ type: 'get_thoughts' });
          break;
      }
    }
  }, [lastMessage, sendMessage]);

  const getThoughtIcon = (type) => {
    switch (type) {
      case 'observation': return '👁️';
      case 'thought': return '💭';
      case 'action': return '⚡';
      case 'reflection': return '🤔';
      default: return '📝';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'thinking': return 'text-blue-600 bg-blue-50';
      case 'planning': return 'text-purple-600 bg-purple-50';
      case 'executing': return 'text-orange-600 bg-orange-50';
      case 'reflecting': return 'text-green-600 bg-green-50';
      case 'completed': return 'text-green-600 bg-green-50';
      case 'error': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">
          Agent Workflow Visualizer
        </h2>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(agentStatus)}`}>
          {agentStatus.charAt(0).toUpperCase() + agentStatus.slice(1)}
        </div>
      </div>

      {/* Connection Status */}
      <div className="mb-4 p-3 bg-gray-50 rounded-md">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-400' : 
            connectionStatus === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'
          }`}></div>
          <span className="text-sm text-gray-600">
            WebSocket: {connectionStatus}
          </span>
        </div>
        {error && (
          <div className="mt-2 text-sm text-red-600">{error}</div>
        )}
      </div>

      {/* Thought Trail */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {thoughtTrail.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">🧠</div>
            <p>No thoughts yet. Agent is ready to start thinking!</p>
          </div>
        ) : (
          thoughtTrail.map((thought, index) => (
            <div 
              key={thought.id || index} 
              className="border border-gray-200 rounded-md p-3 hover:bg-gray-50"
            >
              <div className="flex items-start gap-3">
                <div className="text-2xl">{getThoughtIcon(thought.type)}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-gray-700 capitalize">
                      {thought.type}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(thought.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-800">{thought.content}</p>
                  {thought.metadata && Object.keys(thought.metadata).length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs text-gray-500 cursor-pointer">
                        Metadata
                      </summary>
                      <pre className="mt-1 text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                        {JSON.stringify(thought.metadata, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Refresh Button */}
      <div className="mt-4 flex justify-center">
        <button
          onClick={() => {
            sendMessage({ type: 'get_status' });
            sendMessage({ type: 'get_thoughts' });
          }}
          className="px-4 py-2 bg-blue-600 text-foreground rounded hover:bg-blue-700 transition-colors"
          disabled={connectionStatus !== 'connected'}
        >
          Refresh Thoughts
        </button>
      </div>
    </div>
  );
};

export default AgentWorkflowVisualizer;