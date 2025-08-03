import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Activity, 
  Clock, 
  Database, 
  AlertCircle, 
  Search,
  ChevronRight,
  ChevronDown,
  CheckCircle,
  XCircle,
  Loader2,
  Terminal as TerminalIcon,
  Globe,
  Code,
  FileText,
  Minimize2,
  Maximize2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";

interface DebugEvent {
  id: string;
  timestamp: string;
  type: 'agent_thinking' | 'tool_call' | 'tool_result' | 'error' | 'user_message' | 'agent_response';
  content: {
    text?: string;
    tool_name?: string;
    tool_input?: Record<string, unknown>;
    result?: string | Record<string, unknown>;
    error?: string;
    [key: string]: unknown;
  };
  duration?: number;
  tokens?: number;
  status?: 'success' | 'error' | 'pending';
  toolName?: string;
}

interface AgentMetrics {
  totalTokens: number;
  totalDuration: number;
  toolCallCount: number;
  thoughtSteps: number;
  errorCount: number;
  successRate: number;
}

interface ThinkingMessage {
  id: string;
  content: string;
  timestamp: number;
}

interface AgentDebuggerPanelProps {
  agentEvents: DebugEvent[];
  thinkingMessages: ThinkingMessage[];
  toolCalls: DebugEvent[];
}

const AgentDebuggerPanel: React.FC<AgentDebuggerPanelProps> = ({ 
  agentEvents, 
  thinkingMessages, 
  toolCalls 
}) => {
  const [activeTab, setActiveTab] = useState('timeline');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  const [isMinimized, setIsMinimized] = useState(false);

  // Calculate metrics from real data
  const metrics: AgentMetrics = {
    totalTokens: agentEvents.reduce((sum, e) => sum + (e.tokens || 0), 0),
    totalDuration: agentEvents.reduce((sum, e) => sum + (e.duration || 0), 0),
    toolCallCount: toolCalls.length,
    thoughtSteps: thinkingMessages.length,
    errorCount: agentEvents.filter(e => e.status === 'error').length,
    successRate: toolCalls.length > 0 
      ? Math.round((toolCalls.filter(e => e.status === 'success').length / toolCalls.length) * 100)
      : 100
  };

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ctrl/Cmd + D to toggle minimize
      if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        setIsMinimized(prev => !prev);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const toggleEventExpansion = (eventId: string) => {
    const newExpanded = new Set(expandedEvents);
    if (newExpanded.has(eventId)) {
      newExpanded.delete(eventId);
    } else {
      newExpanded.add(eventId);
    }
    setExpandedEvents(newExpanded);
  };

  const filteredEvents = agentEvents.filter(event => {
    if (filterType !== 'all' && event.type !== filterType) return false;
    if (searchTerm && !JSON.stringify(event).toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'agent_thinking': return <Brain className="h-4 w-4 text-purple-400" />;
      case 'tool_call': return <Database className="h-4 w-4 text-blue-400" />;
      case 'tool_result': return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'error': return <XCircle className="h-4 w-4 text-red-400" />;
      case 'user_message': return <Activity className="h-4 w-4 text-yellow-400" />;
      default: return <Activity className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getToolIcon = (toolName?: string) => {
    if (!toolName) return <Database className="h-4 w-4" />;
    if (toolName.includes('terminal')) return <TerminalIcon className="h-4 w-4 text-green-400" />;
    if (toolName.includes('browser')) return <Globe className="h-4 w-4 text-blue-400" />;
    if (toolName.includes('code')) return <Code className="h-4 w-4 text-purple-400" />;
    if (toolName.includes('file')) return <FileText className="h-4 w-4 text-yellow-400" />;
    return <Database className="h-4 w-4 text-orange-400" />;
  };

  // Minimized view
  if (isMinimized) {
    return (
      <div className="fixed bottom-4 left-4 bg-card border border-border rounded-lg shadow-2xl z-50 animate-in slide-in-from-bottom-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsMinimized(false)}
          className="flex items-center gap-2 p-3 hover:bg-muted"
          title="Maximize debugger (Ctrl+D)"
        >
          <Brain className="h-5 w-5 text-purple-400" />
          <span className="text-foreground font-semibold">Agent Debugger</span>
          <div className="flex items-center gap-2 ml-4">
            <span className="text-xs text-muted-foreground">
              {metrics.thoughtSteps} thoughts • {metrics.toolCallCount} tools
            </span>
            {metrics.errorCount > 0 && (
              <span className="text-xs text-red-400">
                {metrics.errorCount} errors
              </span>
            )}
          </div>
          <Maximize2 className="h-4 w-4 ml-2" />
        </Button>
      </div>
    );
  }

  // Full view
  return (
    <div className="fixed bottom-0 left-0 right-0 h-96 bg-card border-t border-border shadow-2xl z-50 flex flex-col animate-in slide-in-from-bottom-4">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-400" />
          <span className="text-foreground font-semibold">Agent Debugger</span>
        </div>
        
        {/* Metrics Bar */}
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1">
            <Activity className="h-4 w-4 text-blue-400" />
            <span className="text-muted-foreground">Tokens:</span>
            <span className="text-foreground font-mono">{metrics.totalTokens.toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-4 w-4 text-green-400" />
            <span className="text-muted-foreground">Time:</span>
            <span className="text-foreground font-mono">{(metrics.totalDuration / 1000).toFixed(1)}s</span>
          </div>
          <div className="flex items-center gap-1">
            <Database className="h-4 w-4 text-yellow-400" />
            <span className="text-muted-foreground">Tools:</span>
            <span className="text-foreground font-mono">{metrics.toolCallCount}</span>
          </div>
          {metrics.errorCount > 0 && (
            <div className="flex items-center gap-1">
              <AlertCircle className="h-4 w-4 text-red-400" />
              <span className="text-red-400 font-mono">{metrics.errorCount}</span>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsMinimized(true)}
            className="h-8 w-8"
            title="Minimize debugger (Ctrl+D)"
          >
            <Minimize2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-border bg-muted/50">
        {['timeline', 'network', 'console', 'memory'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium capitalize transition-all duration-200 ${
              activeTab === tab
                ? 'text-foreground bg-background border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Controls Bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-muted/30">
        <div className="flex items-center gap-2 flex-1">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search events..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="h-8 flex-1"
          />
        </div>
        <Select value={filterType} onValueChange={setFilterType}>
          <SelectTrigger className="w-40 h-8">
            <SelectValue placeholder="Filter by type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Events</SelectItem>
            <SelectItem value="agent_thinking">Thinking</SelectItem>
            <SelectItem value="tool_call">Tool Calls</SelectItem>
            <SelectItem value="tool_result">Tool Results</SelectItem>
            <SelectItem value="error">Errors</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'timeline' && (
          <div className="space-y-2">
            {filteredEvents.map(event => (
              <div 
                key={event.id} 
                className="bg-muted/30 rounded-lg overflow-hidden hover:bg-muted/50 transition-all duration-200 border border-transparent hover:border-border"
              >
                <div 
                  className="flex items-start gap-3 p-3 cursor-pointer"
                  onClick={() => toggleEventExpansion(event.id)}
                >
                  <div className="mt-0.5">
                    {expandedEvents.has(event.id) ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </div>
                  {getEventIcon(event.type)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm capitalize">
                        {event.type.replace('_', ' ')}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-sm text-muted-foreground mt-1 line-clamp-2">
                      {event.content.text || event.toolName || 'Event details'}
                    </div>
                    {(event.duration || event.tokens) && (
                      <div className="flex items-center gap-3 mt-2 text-xs">
                        {event.duration && (
                          <span className="text-muted-foreground">
                            <Clock className="inline h-3 w-3 mr-1" />
                            {event.duration}ms
                          </span>
                        )}
                        {event.tokens && (
                          <span className="text-muted-foreground">
                            <Activity className="inline h-3 w-3 mr-1" />
                            {event.tokens} tokens
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  {event.status && (
                    <div className={`px-2 py-1 rounded text-xs font-medium ${
                      event.status === 'success' ? 'bg-green-400/20 text-green-400' :
                      event.status === 'error' ? 'bg-red-400/20 text-red-400' :
                      'bg-yellow-400/20 text-yellow-400'
                    }`}>
                      {event.status}
                    </div>
                  )}
                </div>
                
                {expandedEvents.has(event.id) && (
                  <div className="border-t border-border bg-background/50 p-3">
                    <pre className="text-xs font-mono whitespace-pre-wrap break-all text-muted-foreground">
                      {JSON.stringify(event.content, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'network' && (
          <div className="space-y-2">
            {toolCalls.length > 0 ? (
              toolCalls.map(event => (
                <div key={event.id} className="bg-muted/30 rounded-lg p-3 hover:bg-muted/50 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getToolIcon(event.toolName)}
                      <div>
                        <span className="font-medium text-sm">{event.toolName || 'Unknown Tool'}</span>
                        <div className="text-xs text-muted-foreground mt-1">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {event.duration && (
                        <span className="text-xs text-muted-foreground">{event.duration}ms</span>
                      )}
                      {event.status === 'success' ? (
                        <CheckCircle className="h-4 w-4 text-green-400" />
                      ) : event.status === 'error' ? (
                        <XCircle className="h-4 w-4 text-red-400" />
                      ) : (
                        <Loader2 className="h-4 w-4 text-yellow-400 animate-spin" />
                      )}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No network requests yet</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'console' && (
          <div className="font-mono text-xs space-y-1">
            {agentEvents.map(event => (
              <div key={event.id} className={`flex items-start gap-2 ${
                event.type === 'agent_thinking' ? 'text-purple-400' :
                event.type === 'tool_call' ? 'text-blue-400' :
                event.type === 'tool_result' && event.status === 'success' ? 'text-green-400' :
                event.type === 'error' ? 'text-red-400' :
                'text-yellow-400'
              }`}>
                <span className="text-muted-foreground opacity-60">
                  [{new Date(event.timestamp).toLocaleTimeString()}]
                </span>
                <span className="font-semibold">{event.type}:</span>
                <span className="text-muted-foreground">
                  {typeof event.content?.text === 'string' 
                    ? event.content.text.substring(0, 100) + (event.content.text.length > 100 ? '...' : '')
                    : event.toolName || JSON.stringify(event.content).substring(0, 100)
                  }
                </span>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'memory' && (
          <div className="space-y-4">
            <div className="bg-muted/30 rounded-lg p-4">
              <div className="flex justify-between items-center mb-3">
                <span className="text-sm font-medium">Context Window Usage</span>
                <span className="text-sm font-mono">
                  {metrics.totalTokens.toLocaleString()} / 200K
                </span>
              </div>
              <Progress 
                value={(metrics.totalTokens / 200000) * 100} 
                className="h-2"
              />
              <div className="text-xs text-muted-foreground mt-2">
                {((metrics.totalTokens / 200000) * 100).toFixed(1)}% utilized
              </div>
            </div>
            
            <div className="bg-muted/30 rounded-lg p-4">
              <h3 className="text-sm font-medium mb-3">Session Statistics</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Thinking Messages:</span>
                  <span className="font-mono">{thinkingMessages.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Tool Calls:</span>
                  <span className="font-mono">{toolCalls.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Events:</span>
                  <span className="font-mono">{agentEvents.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Success Rate:</span>
                  <span className={`font-mono ${
                    metrics.successRate >= 90 ? 'text-green-400' : 
                    metrics.successRate >= 70 ? 'text-yellow-400' : 
                    'text-red-400'
                  }`}>
                    {metrics.successRate}%
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-muted/30 rounded-lg p-4">
              <h3 className="text-sm font-medium mb-3">Performance Metrics</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avg Response Time:</span>
                  <span className="font-mono">
                    {toolCalls.length > 0 
                      ? (metrics.totalDuration / toolCalls.length / 1000).toFixed(2) 
                      : '0.00'
                    }s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Processing:</span>
                  <span className="font-mono">{(metrics.totalDuration / 1000).toFixed(1)}s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Error Rate:</span>
                  <span className={`font-mono ${
                    metrics.errorCount === 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {toolCalls.length > 0 
                      ? ((metrics.errorCount / toolCalls.length) * 100).toFixed(1) 
                      : '0.0'
                    }%
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentDebuggerPanel;