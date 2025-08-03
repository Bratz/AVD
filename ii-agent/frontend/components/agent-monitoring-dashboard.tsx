import React, { useState, useEffect, useMemo } from 'react';
import { useAppContext } from "@/context/app-context";
import { 
  Activity, 
  Brain, 
  Zap, 
  TrendingUp, 
  Clock, 
  CheckCircle,
  XCircle,
  BarChart3,
  FileText,
  Globe,
  Terminal,
  Code,
  AlertCircle,
  RefreshCw,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";

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

interface DebugEvent {
  id: string;
  timestamp: string;
  type: string;
  content: {
    text?: string;
    tool_name?: string;
    tool_input?: Record<string, unknown>;
    result?: string | Record<string, unknown>;
    [key: string]: unknown;
  };
  duration?: number;
  tokens?: number;
  status?: 'success' | 'error' | 'pending';
  toolName?: string;
}

interface AgentData {
  metrics: AgentMetrics;
  events: DebugEvent[];
  thinkingMessages: ThinkingMessage[];
}

interface AgentMonitoringDashboardProps {
  sessionId: string | null;
  agentData: AgentData;
  onClose?: () => void;
}

const AgentMonitoringDashboard: React.FC<AgentMonitoringDashboardProps> = ({ sessionId, agentData, onClose }) => {
  const { state } = useAppContext();
  const [timeRange, setTimeRange] = useState('live');
  const [refreshInterval, setRefreshInterval] = useState(1000);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  
  // Use REAL data from props and state
  const metrics = agentData?.metrics || state.agentMetrics;
  const events = agentData?.events || state.debugEvents;
  const thinkingMessages = agentData?.thinkingMessages || state.thinkingMessages;

  // Handle ESC key to close
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && onClose) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Auto-refresh
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  // Calculate real-time statistics
  const realtimeStats = useMemo(() => {
    const now = Date.now();
    const timeRanges = {
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      'live': 0
    };
    
    const cutoff = timeRanges[timeRange as keyof typeof timeRanges];
    const recentEvents = cutoff === 0 ? events : events.filter(e => 
      now - new Date(e.timestamp).getTime() < cutoff
    );

    // Calculate tool statistics
    const toolStats = recentEvents.reduce((acc, event) => {
      if (event.type === 'tool_call' && event.content?.tool_name) {
        const toolName = event.content.tool_name as string;
        if (!acc[toolName]) {
          acc[toolName] = { name: toolName, count: 0, totalTime: 0, failures: 0, lastUsed: event.timestamp };
        }
        acc[toolName].count++;
        acc[toolName].lastUsed = event.timestamp;
      } else if (event.type === 'tool_result') {
        const toolName = event.toolName || event.content?.tool_name as string;
        if (toolName && acc[toolName]) {
          if (event.duration) acc[toolName].totalTime += event.duration;
          if (event.status === 'error') acc[toolName].failures++;
        }
      }
      return acc;
    }, {} as Record<string, { name: string; count: number; totalTime: number; failures: number; lastUsed: string }>);

    return {
      recentEvents,
      toolStats: Object.values(toolStats).sort((a, b) => b.count - a.count),
      eventRate: recentEvents.length / (cutoff / 60000 || 1), // events per minute
      avgResponseTime: recentEvents.reduce((sum, e) => sum + (e.duration || 0), 0) / (recentEvents.length || 1)
    };
  }, [events, timeRange]);

  // Get tool icon
  const getToolIcon = (toolName: string) => {
    if (toolName.includes('terminal')) return <Terminal className="h-5 w-5 text-green-400" />;
    if (toolName.includes('browser')) return <Globe className="h-5 w-5 text-blue-400" />;
    if (toolName.includes('code')) return <Code className="h-5 w-5 text-purple-400" />;
    if (toolName.includes('file')) return <FileText className="h-5 w-5 text-yellow-400" />;
    return <Zap className="h-5 w-5 text-orange-400" />;
  };

  // Format recent thoughts
  const recentThoughts = thinkingMessages
    .slice(-5)
    .reverse()
    .map(msg => ({
      id: msg.id,
      content: msg.content.length > 100 ? msg.content.substring(0, 100) + '...' : msg.content,
      timestamp: new Date(msg.timestamp).toLocaleTimeString()
    }));

  return (
    <div 
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-7xl h-[90vh] bg-background rounded-xl overflow-hidden shadow-2xl border border-border"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="h-full overflow-auto">
          <div className="p-6">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <Brain className="h-8 w-8 text-purple-400" />
                <div>
                  <h1 className="text-2xl font-bold text-foreground">Agent Monitoring Dashboard</h1>
                  <p className="text-muted-foreground text-sm">
                    Session: {sessionId || 'current'} | Last update: {lastUpdate.toLocaleTimeString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <Select value={timeRange} onValueChange={setTimeRange}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="live">Live</SelectItem>
                    <SelectItem value="1m">Last 1m</SelectItem>
                    <SelectItem value="5m">Last 5m</SelectItem>
                    <SelectItem value="15m">Last 15m</SelectItem>
                  </SelectContent>
                </Select>
                
                <div className="flex items-center gap-2">
                  <RefreshCw className={`h-4 w-4 ${refreshInterval === 1000 ? 'animate-spin' : ''}`} />
                  <Select value={refreshInterval.toString()} onValueChange={(v) => setRefreshInterval(Number(v))}>
                    <SelectTrigger className="w-24">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1000">1s</SelectItem>
                      <SelectItem value="5000">5s</SelectItem>
                      <SelectItem value="10000">10s</SelectItem>
                      <SelectItem value="30000">30s</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                {onClose && (
                  <Button variant="ghost" size="icon" onClick={onClose}>
                    <X className="h-5 w-5" />
                  </Button>
                )}
              </div>
            </div>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
              <div className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <Brain className="h-5 w-5 text-purple-400" />
                  <span className="text-xs text-muted-foreground px-2 py-1 bg-purple-400/10 rounded">
                    {realtimeStats.eventRate.toFixed(1)}/min
                  </span>
                </div>
                <div className="text-2xl font-bold text-foreground">{metrics.thoughtSteps}</div>
                <div className="text-xs text-muted-foreground">Active Thoughts</div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <Zap className="h-5 w-5 text-yellow-400" />
                  {metrics.toolCallCount > 0 && (
                    <TrendingUp className="h-3 w-3 text-green-400" />
                  )}
                </div>
                <div className="text-2xl font-bold text-foreground">{metrics.toolCallCount}</div>
                <div className="text-xs text-muted-foreground">Tools Executed</div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className={`text-xs px-2 py-1 rounded ${
                    metrics.successRate >= 90 ? 'bg-green-400/10 text-green-400' :
                    metrics.successRate >= 70 ? 'bg-yellow-400/10 text-yellow-400' :
                    'bg-red-400/10 text-red-400'
                  }`}>
                    {metrics.successRate}%
                  </span>
                </div>
                <div className="text-2xl font-bold text-foreground">{metrics.successRate}%</div>
                <div className="text-xs text-muted-foreground">Success Rate</div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <Clock className="h-5 w-5 text-blue-400" />
                  <span className="text-xs text-muted-foreground">avg</span>
                </div>
                <div className="text-2xl font-bold text-foreground">
                  {realtimeStats.avgResponseTime > 0 ? (realtimeStats.avgResponseTime / 1000).toFixed(1) : 0}s
                </div>
                <div className="text-xs text-muted-foreground">Response Time</div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <Activity className="h-5 w-5 text-orange-400" />
                  <Progress value={(metrics.totalTokens / 200000) * 100} className="w-12 h-1" />
                </div>
                <div className="text-2xl font-bold text-foreground">{metrics.totalTokens.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">Tokens Used</div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <BarChart3 className="h-5 w-5 text-emerald-400" />
                  <span className="text-xs text-muted-foreground">USD</span>
                </div>
                <div className="text-2xl font-bold text-foreground">
                  ${((metrics.totalTokens / 1000000) * 3).toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">Est. Cost</div>
              </div>
            </div>

            {/* Error Alert */}
            {metrics.errorCount > 0 && (
              <div className="bg-red-400/10 border border-red-400/30 rounded-lg p-4 mb-6 flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-red-400" />
                <div className="flex-1">
                  <div className="font-medium text-red-400">
                    {metrics.errorCount} Error{metrics.errorCount > 1 ? 's' : ''} Detected
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Check the console for details
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Tool Usage Statistics */}
              <div className="bg-card border border-border rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-foreground">
                  <Zap className="h-5 w-5 text-yellow-400" />
                  Tool Usage Statistics
                </h2>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {realtimeStats.toolStats.length > 0 ? (
                    realtimeStats.toolStats.map((tool, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          {getToolIcon(tool.name)}
                          <div>
                            <div className="font-medium text-foreground">{tool.name}</div>
                            <div className="text-xs text-muted-foreground">
                              Avg: {tool.count > 0 ? (tool.totalTime / tool.count / 1000).toFixed(1) : 0}s
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <div className="font-mono text-sm text-foreground">{tool.count} calls</div>
                            <div className="text-xs text-muted-foreground">
                              {tool.failures === 0 ? (
                                <span className="text-green-400">100% success</span>
                              ) : (
                                <span className="text-red-400">{tool.failures} failures</span>
                              )}
                            </div>
                          </div>
                          {tool.failures === 0 ? (
                            <CheckCircle className="h-4 w-4 text-green-400" />
                          ) : (
                            <XCircle className="h-4 w-4 text-red-400" />
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      <Zap className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p>No tool usage data yet</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Thought Stream */}
              <div className="bg-card border border-border rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-foreground">
                  <Brain className="h-5 w-5 text-purple-400" />
                  Thought Stream ({thinkingMessages.length})
                </h2>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {recentThoughts.length > 0 ? (
                    recentThoughts.map((thought) => (
                      <div key={thought.id} className="bg-muted/30 rounded-lg p-3 hover:bg-muted/50 transition-colors">
                        <div className="text-xs text-muted-foreground mb-1">{thought.timestamp}</div>
                        <div className="text-sm text-foreground">{thought.content}</div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p>No thoughts captured yet</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Event Timeline */}
            <div className="bg-card border border-border rounded-lg p-4 mt-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-foreground">
                <Activity className="h-5 w-5 text-blue-400" />
                Recent Events ({realtimeStats.recentEvents.length})
              </h2>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {realtimeStats.recentEvents.slice(-10).reverse().map(event => (
                  <div key={event.id} className="flex items-center gap-3 text-sm">
                    <span className="text-xs text-muted-foreground w-16">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                    <div className={`w-2 h-2 rounded-full ${
                      event.type === 'error' ? 'bg-red-400' :
                      event.type === 'tool_result' && event.status === 'success' ? 'bg-green-400' :
                      'bg-blue-400'
                    }`} />
                    <span className="text-foreground flex-1">
                      {event.type.replace('_', ' ')}
                      {event.toolName && ` - ${event.toolName}`}
                    </span>
                    {event.duration && (
                      <span className="text-xs text-muted-foreground">{event.duration}ms</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentMonitoringDashboard;