import React, { useState, useEffect, useRef, useMemo } from 'react';
import { registerTool } from '../../agentic-ui/ag-ui-client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

// ===== Types =====
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agent?: string;
  timestamp: Date;
  metadata?: {
    thinking?: boolean;
    handoff?: { from: string; to: string };
    tools?: string[];
  };
}

interface WorkflowState {
  currentAgent: string | null;
  activeAgents: string[];
  processedSteps: number;
  totalSteps?: number;
}

// ===== API Service for Chat =====
class ROWBOATChatService {
  private baseUrl: string;

  constructor(baseUrl: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9001') {
    this.baseUrl = baseUrl;
  }

  async chatWithWorkflow(
    workflowId: string,
    message: string,
    conversationHistory: ChatMessage[] = []
  ): Promise<Response> {
    const response = await fetch(`${this.baseUrl}/rowboat/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        workflow_id: workflowId,
        message,
        conversation_history: conversationHistory.map(msg => ({
          role: msg.role,
          content: msg.content,
        })),
        stream: true,
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    
    return response;
  }
}

// ===== Chat Components =====

// Message Component
const ChatMessage: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const isUser = message.role === 'user';
  const isThinking = message.metadata?.thinking;
  const isHandoff = !!message.metadata?.handoff;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[70%] ${isUser ? 'order-2' : 'order-1'}`}>
        {!isUser && message.agent && (
          <div className="text-xs text-gray-500 mb-1 flex items-center gap-2">
            <span className="font-medium">{message.agent}</span>
            {isThinking && (
              <span className="inline-flex items-center gap-1">
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                thinking...
              </span>
            )}
          </div>
        )}
        
        {isHandoff && message.metadata?.handoff ? (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-2 rounded-lg text-sm">
            <span className="font-medium">{message.metadata.handoff.from}</span>
            {' → '}
            <span className="font-medium">@{message.metadata.handoff.to}</span>
          </div>
        ) : (
          <div
            className={`px-4 py-2 rounded-lg ${
              isUser
                ? 'bg-blue-500 text-white'
                : isThinking
                ? 'bg-gray-100 text-gray-600 italic'
                : 'bg-gray-200 text-gray-800'
            }`}
          >
            <div className="whitespace-pre-wrap">{message.content}</div>
            {message.metadata?.tools && message.metadata.tools.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-300 text-xs opacity-75">
                🔧 Used: {message.metadata.tools.join(', ')}
              </div>
            )}
          </div>
        )}
        
        <div className="text-xs text-gray-400 mt-1">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

// Agent Status Indicator
const AgentStatusIndicator: React.FC<{ state: WorkflowState }> = ({ state }) => {
  if (!state.currentAgent && state.activeAgents.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-50 border-t px-4 py-2">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-4">
          {state.currentAgent && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-gray-600">
                Active: <span className="font-medium text-gray-800">{state.currentAgent}</span>
              </span>
            </div>
          )}
          {state.activeAgents.length > 1 && (
            <div className="text-gray-500">
              {state.activeAgents.filter(a => a !== state.currentAgent).join(', ')} waiting
            </div>
          )}
        </div>
        {state.totalSteps && (
          <div className="text-gray-500">
            Step {state.processedSteps} of {state.totalSteps}
          </div>
        )}
      </div>
    </div>
  );
};

// Main Chat Interface
const WorkflowChat: React.FC<{ workflowId: string; workflowName?: string; onBack?: () => void }> = ({
  workflowId,
  workflowName = 'Workflow',
  onBack,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [workflowState, setWorkflowState] = useState<WorkflowState>({
    currentAgent: null,
    activeAgents: [],
    processedSteps: 0,
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatService = useMemo(() => new ROWBOATChatService(), []);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Stream response handler
  const handleStreamResponse = async (response: Response) => {
    const reader = response.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              handleStreamEvent(data);
            } catch (e) {
              console.error('Failed to parse stream event:', e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  };

  const handleStreamEvent = (event: any) => {
    switch (event.type) {
      case 'agent_thinking':
        setWorkflowState(prev => ({
          ...prev,
          currentAgent: event.agent,
          activeAgents: [...new Set([...prev.activeAgents, event.agent])],
        }));
        
        setMessages(prev => [...prev, {
          id: `${Date.now()}-thinking`,
          role: 'assistant' as const,
          content: event.content,
          agent: event.agent,
          timestamp: new Date(),
          metadata: { thinking: true },
        }]);
        break;

      case 'agent_response':
        // Remove thinking message and add actual response
        setMessages(prev => {
          const filtered = prev.filter(m => !m.metadata?.thinking || m.agent !== event.agent);
          return [...filtered, {
            id: `${Date.now()}-response`,
            role: 'assistant' as const,
            content: event.content,
            agent: event.agent,
            timestamp: new Date(),
            metadata: { tools: event.tools_used },
          }];
        });
        break;

      case 'agent_handoff':
        setMessages(prev => [...prev, {
          id: `${Date.now()}-handoff`,
          role: 'system' as const,
          content: `Handoff: ${event.from_agent} → @${event.to_agent}`,
          timestamp: new Date(),
          metadata: {
            handoff: { from: event.from_agent, to: event.to_agent },
          },
        }]);
        
        setWorkflowState(prev => ({
          ...prev,
          currentAgent: event.to_agent,
          processedSteps: prev.processedSteps + 1,
        }));
        break;

      case 'workflow_complete':
        setWorkflowState(prev => ({
          ...prev,
          currentAgent: null,
          activeAgents: [],
        }));
        setIsLoading(false);
        break;

      case 'error':
        setMessages(prev => [...prev, {
          id: `${Date.now()}-error`,
          role: 'system' as const,
          content: `Error: ${event.message}`,
          timestamp: new Date(),
        }]);
        setIsLoading(false);
        break;
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await chatService.chatWithWorkflow(
        workflowId,
        input,
        messages
      );
      
      await handleStreamResponse(response);
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        id: `${Date.now()}-error`,
        role: 'system' as const,
        content: 'Failed to send message. Please try again.',
        timestamp: new Date(),
      }]);
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold">{workflowName}</h2>
            <p className="text-sm opacity-80">Multi-agent conversation</p>
          </div>
          {onBack && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onBack}
              className="text-white hover:bg-white/20"
            >
              Back
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">
            <p className="text-lg mb-2">Start a conversation</p>
            <p className="text-sm">
              Type a message below to interact with the {workflowName} workflow
            </p>
          </div>
        ) : (
          messages.map(message => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Agent Status */}
      <AgentStatusIndicator state={workflowState} />

      {/* Input */}
      <div className="border-t px-6 py-4">
        <div className="flex gap-3">
          <Input
            type="text"
            className="flex-1"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            disabled={isLoading}
          />
          <Button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? (
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              'Send'
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

// Template Gallery Component
const WorkflowTemplateGallery: React.FC<{
  onSelectTemplate: (template: any) => void;
}> = ({ onSelectTemplate }) => {
  const templates = [
    {
      id: 'customer-support',
      name: 'Customer Support System',
      description: 'Routes customer inquiries to specialized agents',
      agents: ['intake', 'billing_specialist', 'technical_support', 'escalation'],
      icon: '🎧',
    },
    {
      id: 'research-assistant',
      name: 'Research Assistant',
      description: 'Researches topics and creates comprehensive reports',
      agents: ['researcher', 'fact_checker', 'writer', 'editor'],
      icon: '🔬',
    },
    {
      id: 'code-review',
      name: 'Code Review Pipeline',
      description: 'Automated code review with multiple specialized reviewers',
      agents: ['syntax_checker', 'security_reviewer', 'performance_analyst', 'architect'],
      icon: '💻',
    },
    {
      id: 'content-creation',
      name: 'Content Creation Workflow',
      description: 'Create and optimize content across multiple formats',
      agents: ['content_planner', 'writer', 'seo_optimizer', 'publisher'],
      icon: '✍️',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {templates.map(template => (
        <button
          key={template.id}
          className="bg-white border-2 border-gray-200 rounded-lg p-6 hover:border-blue-500 hover:shadow-lg transition-all text-left"
          onClick={() => onSelectTemplate(template)}
        >
          <div className="flex items-start gap-4">
            <div className="text-4xl">{template.icon}</div>
            <div className="flex-1">
              <h3 className="font-semibold text-lg mb-1">{template.name}</h3>
              <p className="text-gray-600 text-sm mb-3">{template.description}</p>
              <div className="flex flex-wrap gap-1">
                {template.agents.map(agent => (
                  <span
                    key={agent}
                    className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
                  >
                    {agent}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
};

// Register components
registerTool('workflow-chat', WorkflowChat);
registerTool('workflow-template-gallery', WorkflowTemplateGallery);

// Export components
export { WorkflowChat, WorkflowTemplateGallery, ROWBOATChatService };
export default WorkflowChat;