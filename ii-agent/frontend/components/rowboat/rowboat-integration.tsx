import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  ReactFlowProvider,
  Panel,
  Handle,
  Position,
  MarkerType,
  NodeProps,
  getSmoothStepPath,
} from 'reactflow';
import 'reactflow/dist/style.css';
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
  X,
  Send,
  Loader2,
  Play,
  Pause,
  Plus,
  Settings,
  Save,
  Upload,
  Download,
  Copy,
  Eye,
  EyeOff,
  MessageSquare,
  Network,
  Cpu,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Sparkles,
  Wifi,
  WifiOff,
  Server,
  GitBranch,
  Layers
} from 'lucide-react';

// ===== Enhanced TypeScript Interfaces =====
interface WorkflowDefinition {
  id?: string; // Add this field
  name: string;
  description: string;
  agents: AgentConfig[];
  edges: WorkflowEdge[];
  entry_point: string;
  metadata: Record<string, any>;
  version?: string;
  created_at?: string;
  updated_at?: string;
}

interface AgentConfig {
  name: string;
  role: string;
  instructions: string;
  tools?: string[];
  temperature?: number;
  model?: string;
  outputVisibility?: 'user_facing' | 'internal';
  controlType?: 'retain' | 'relinquish_to_parent' | 'start_agent';
  connected_agents?: string[];
  examples?: { input: string; output: string }[];
  metadata?: Record<string, any>;
  mcp_servers?: string[];
  description?: string;  // Add this
  max_tokens?: number;   // Add this
}

interface WorkflowEdge {
  from_agent: string;
  to_agent: string;
  condition?: string;
  isMentionBased?: boolean;
}

interface WorkflowVisualization {
  nodes: Array<{
    id: string;
    type: string;
    data: {
      label: string;
      role: string;
      hasInstructions: boolean;
      connectedAgents: string[];
      tools?: string[];
      model?: string;
      status?: 'idle' | 'thinking' | 'executing' | 'error' | 'complete';
      metrics?: AgentMetrics;
      outputVisibility?: 'user_facing' | 'internal';
      name?: string;
      instructions?: string;
      visibility?: 'user_facing' | 'internal'; // Add this - from API response
      controlType?: string; // Add this - from API response
    };
    position: { x: number; y: number };
    style?: { // Add this - from API response
      backgroundColor?: string;
      [key: string]: any;
    };
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    type: string;
    animated: boolean;
    label?: string;
    data?: WorkflowEdge;
  }>;
  metadata: Record<string, any>;
}

interface AgentMetrics {
  totalTokens: number;
  totalDuration: number;
  toolCallCount: number;
  thoughtSteps: number;
  errorCount: number;
  successRate: number;
}

interface WorkflowExecution {
  workflow_id: string;
  execution_id: string;
  status: string;
  current_agent?: string;
  messages: ExecutionMessage[];
  metrics?: Record<string, AgentMetrics>;
}

interface ExecutionMessage {
  id?: string;
  agent: string;
  content: string;
  timestamp: string;
  type: 'thinking' | 'response' | 'handoff' | 'tool_call' | 'tool_result' | 'error';
  metadata?: Record<string, any>;
  visibility?: 'internal' | 'external';
}

interface MCPServer {
  name: string;
  url: string;
  tools: string[];
  status: 'connected' | 'disconnected' | 'error' | 'available' | 'unknown';
  error?: string;
}

interface StreamEvent {
  type: 'agent' | 'tool' | 'prompt' | 'edge' | 'progress' | 'complete' | 'error' | 'message' | 'done';
  data: any;
  timestamp?: number;
}

// Enhanced ROWBOAT Node Data
interface ROWBOATNodeData {
  name: string;
  role: string;
  model?: string;
  tools: string[];
  mcpServers?: string[];
  instructions: string;
  examples?: string[];
  isParallel?: boolean;
  isSubworkflow?: boolean;
  hasApproval?: boolean;
  suggestedTools?: string[];
  mentionedAgents?: string[];
  status?: 'idle' | 'thinking' | 'executing' | 'error' | 'complete';
  metrics?: AgentMetrics;
  outputVisibility?: 'user_facing' | 'internal';
  label?: string;
  hasInstructions?: boolean;
  connectedAgents?: string[];
  description?: string;      // Add this
  temperature?: number;      // Add this
  max_tokens?: number;       // Add this
  controlType?: string;      // Add this
  metadata?: Record<string, any>; // Add this
}

interface ROWBOATEdgeData {
  condition?: string | null;
  isMentionBased?: boolean;
}


type ROWBOATNode = Node<ROWBOATNodeData>;
type ROWBOATEdge = Edge<ROWBOATEdgeData>;




// ===== Enhanced WebSocket Service =====
class EnhancedROWBOATWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<Function>> = new Map();
  private authToken: string | null = null;
  private workflowId: string | null = null;

  constructor(private baseUrl: string = '') {
    this.baseUrl = baseUrl || process.env.NEXT_PUBLIC_ROWBOAT_API_URL || 'ws://localhost:9000';
    console.log('WebSocket service initialized with baseUrl:', this.baseUrl);
  }

      // Add method to set authentication token
  setAuthToken(token: string) {
    this.authToken = token;
    console.log('Auth token set');
  }

  connect(workflowId: string) {
    const wsUrl = `${this.baseUrl.replace('http', 'ws')}/rowboat/ws/${workflowId}`;
    
    console.log('Attempting WebSocket connection to:', wsUrl);

    try {
    // If you have an auth token, include it in the URL as a query parameter
    // since WebSocket doesn't support custom headers in the browser
    const urlWithAuth = this.authToken 
      ? `${wsUrl}?token=${encodeURIComponent(this.authToken)}`
      : wsUrl;
    
    this.ws = new WebSocket(urlWithAuth);
      
      this.ws.onopen = () => {
        console.log('ROWBOAT WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected', {});
      };
      
      this.ws.onmessage = (event) => {
        console.log('WebSocket message received:', event.data);
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error details:', {
          error,
          readyState: this.ws?.readyState,
          url: wsUrl,
          hasToken: !!this.authToken
        });
        this.emit('error', error);
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        this.emit('disconnected', {});
        // Only attempt reconnect if it wasn't a clean close
        if (!event.wasClean && event.code !== 1000) {
          this.attemptReconnect(workflowId);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.emit('error', error);
    }
  }

  private handleMessage(data: any) {
    const { type, ...restData } = data;
    
    switch (type) {
      case 'agent_message':
        this.emit('agent_message', restData);
        break;
      case 'agent_status':
        this.emit('agent_status', restData);
        break;
      case 'execution_complete':
        this.emit('execution_complete', restData);
        break;
      case 'error':
        this.emit('error', restData);
        break;
      default:
        this.emit(type, restData);
    }
  }
  private attemptReconnect(workflowId: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect(workflowId);
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)?.add(callback);
  }

  off(event: string, callback: Function) {
    this.listeners.get(event)?.delete(callback);
  }

  removeAllListeners() {
    this.listeners.clear();
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }
}

// ===== Enhanced API Service =====
class EnhancedROWBOATApiService {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl?: string, apiKey?: string) {
    this.baseUrl = baseUrl || process.env.NEXT_PUBLIC_ROWBOAT_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9000';
    this.apiKey = apiKey || process.env.NEXT_PUBLIC_ROWBOAT_API_KEY;
  }

  private getHeaders() {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  // Streaming workflow creation from natural language
  async *createWorkflowFromDescriptionStream(
    description: string,
    context?: Record<string, any>
  ): AsyncGenerator<StreamEvent> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/from-description/stream`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ description, context }),
      });

      if (!response.ok) {
        // Fallback to non-streaming endpoint if streaming is not available
        console.warn('Streaming endpoint not available, falling back to standard creation');
        const workflow = await this.createWorkflowFromDescription(description);
        yield { 
          type: 'progress', 
          data: { 
            message: 'Creating workflow (non-streaming mode)...', 
            progress: 50 
          } 
        };
        yield { 
          type: 'complete', 
          data: { 
            workflow_id: workflow.name || workflow.metadata?.workflow_id || 'unknown',
            summary: {
              agents: workflow.agents.length,
              tools: 0,
              edges: workflow.edges.length
            }
          } 
        };
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              yield event;
            } catch (e) {
              console.error('Failed to parse SSE event:', e);
            }
          } else if (line.startsWith('event: ')) {
            // Handle named events
            const eventType = line.slice(7);
            // Next line should be data
            const dataLine = lines.shift();
            if (dataLine?.startsWith('data: ')) {
              try {
                const data = JSON.parse(dataLine.slice(6));
                yield { type: eventType as any, data };
              } catch (e) {
                console.error('Failed to parse named event data:', e);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream creation error:', error);
      // Fallback to mock streaming
      yield* this.mockStreamingCreation(description);
    }
  }

  // Mock streaming for demo/fallback
  private async *mockStreamingCreation(description: string): AsyncGenerator<StreamEvent> {
    yield { type: 'progress', data: { message: 'Analyzing requirements...', progress: 20 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'agent', data: { name: 'coordinator', role: 'coordinator', instructions: 'Coordinate the workflow' } };
    yield { type: 'progress', data: { message: 'Creating coordinator agent...', progress: 40 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'agent', data: { name: 'specialist', role: 'specialist', instructions: 'Process specialized tasks' } };
    yield { type: 'progress', data: { message: 'Creating specialist agent...', progress: 60 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'tool', data: { name: 'data_processor', agent: 'specialist' } };
    yield { type: 'progress', data: { message: 'Adding tools...', progress: 80 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'edge', data: { from_agent: 'coordinator', to_agent: 'specialist', isMentionBased: true } };
    yield { type: 'progress', data: { message: 'Connecting agents...', progress: 90 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { 
      type: 'complete', 
      data: { 
        workflow_id: `workflow-${Date.now()}`,
        summary: { agents: 2, tools: 3, edges: 1 }
      } 
    };
  }

  async createWorkflowFromDescription(description: string, context?: Record<string, any>): Promise<WorkflowDefinition> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/from-description`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ description, context }),
      });
      if (!response.ok) throw new Error('Failed to create workflow');
      return response.json();
    } catch (error) {
      // Return mock workflow for demo
      return {
        name: 'Mock Workflow',
        description: description,
        agents: [
          { name: 'coordinator', role: 'coordinator', instructions: 'Coordinate tasks' },
          { name: 'worker', role: 'specialist', instructions: 'Execute tasks' }
        ],
        edges: [{ from_agent: 'coordinator', to_agent: 'worker' }],
        entry_point: 'coordinator',
        metadata: {}
      };
    }
  }

  async listWorkflows(): Promise<WorkflowDefinition[]> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows`, {
        headers: this.getHeaders(),
      });
      if (!response.ok) {
        console.warn('Failed to list workflows, returning empty array');
        return [];
      }
      
      const data = await response.json();
      return data.workflows || [];
    } catch (error) {
      console.error('Failed to list workflows:', error);
      return [];
    }
  }

  async getWorkflow(workflowId: string): Promise<WorkflowDefinition> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/${workflowId}`, {
        headers: this.getHeaders(),
      });
      if (!response.ok) {
        console.warn('Failed to get workflow details, using basic structure');
        return {
          name: workflowId,
          description: 'Workflow created via ROWBOAT',
          agents: [],
          edges: [],
          entry_point: '',
          metadata: { workflow_id: workflowId }
        };
      }
      return response.json();
    } catch (error) {
      console.error('Failed to get workflow:', error);
      return {
        name: workflowId,
        description: 'Workflow created via ROWBOAT',
        agents: [],
        edges: [],
        entry_point: '',
        metadata: { workflow_id: workflowId }
      };
    }
  }

  async executeWorkflow(workflowId: string, input: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/rowboat/workflows/${workflowId}/execute`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ input, stream: false }),
    });
    if (!response.ok) throw new Error('Failed to execute workflow');
    return response.json();
  }

  // Streaming execution
  async *executeWorkflowStream(workflowId: string, input: string): AsyncGenerator<any> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ input, stream: true }),
      });

      if (!response.ok || !response.body) {
        throw new Error('Failed to execute workflow');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              yield event;
            } catch (e) {
              console.error('Failed to parse execution event:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Execution stream error:', error);
      throw error;
    }
  }

  async testWorkflow(workflowId: string, testInput: string): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/${workflowId}/test`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ test_input: testInput }),
      });
      if (!response.ok) throw new Error('Failed to test workflow');
      return response.json();
    } catch (error) {
      // Return mock test results
      return {
        success: true,
        trace: [
          { agent: 'coordinator', action: 'Received input', output: 'Processing request...' },
          { agent: 'specialist', action: 'Executing task', output: 'Task completed successfully' },
        ],
      };
    }
  }

  async getVisualization(workflowId: string): Promise<WorkflowVisualization> {
    try {
      console.log(`Fetching visualization for workflow: ${workflowId}`);
      const url = `${this.baseUrl}/rowboat/workflows/${workflowId}/visualization`;
      console.log('Visualization URL:', url);
      

      const response = await fetch(url, {
        headers: this.getHeaders(),
      });
      
      console.log('Visualization response status:', response.status);
      
      // Even with 200 status, check for error in response
      const text = await response.text();
      console.log('Raw response:', text);
      
      // Try to parse JSON
      let data;
      try {
        data = JSON.parse(text);
      } catch (e) {
        console.error('Failed to parse visualization response:', e);
        throw new Error('Invalid JSON response from visualization endpoint');
      }
      
      // Check if response contains an error
      if (data.error) {
        console.error('Backend error:', data.error);
        
        // If workflow not found, create a new one or return empty
        if (data.error === 'Workflow not found') {
          console.log('Workflow not found, returning empty visualization');
          return { nodes: [], edges: [], metadata: {} };
        }
        
        throw new Error(data.error);
      }
      // const data = JSON.parse(text);
      // console.log('Raw visualization from API:', text);
      console.log('Parsed visualization data:', data);
      
      // Validate the response structure
      if (!data || typeof data !== 'object') {
        console.error('Invalid visualization data structure:', data);
        return { nodes: [], edges: [], metadata: {} };
      }
      
      // Check if it's wrapped in another object
      if (data.visualization) {
        data = data.visualization;
      }
      
      // Ensure required fields exist
      const visualization: WorkflowVisualization = {
        nodes: Array.isArray(data.nodes) ? data.nodes : [],
        edges: Array.isArray(data.edges) ? data.edges : [],
        metadata: data.metadata || {}
      };
      
      console.log('Final visualization object:', visualization);
      
      return visualization;
    } catch (error) {
      console.error('Failed to get visualization:', error);
      return { nodes: [], edges: [], metadata: {} };
    }
  }

  async getWorkflowMetrics(workflowId: string, startDate?: Date, endDate?: Date): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (startDate) params.append('start_date', startDate.toISOString());
      if (endDate) params.append('end_date', endDate.toISOString());
      
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/${workflowId}/metrics?${params}`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        console.warn('Metrics endpoint not available, returning default metrics');
        return {
          workflow_id: workflowId,
          total_executions: 0,
          successful_executions: 0,
          failed_executions: 0,
          avg_duration: 0,
          success_rate: 0,
          execution_history: [],
          agent_metrics: {},
          time_range: {
            start: startDate?.toISOString() || null,
            end: endDate?.toISOString() || null
          }
        };
      }
      
      return response.json();
    } catch (error) {
      console.error('Failed to get metrics:', error);
      return {
        workflow_id: workflowId,
        total_executions: 0,
        successful_executions: 0,
        failed_executions: 0,
        avg_duration: 0,
        success_rate: 0,
        error: (error as Error).message
      };
    }
  }

  async getMCPServers(): Promise<MCPServer[]> {
    try {
      // First try to get MCP servers from the backend
      const response = await fetch(`${this.baseUrl}/rowboat/mcp/servers`, {
        headers: this.getHeaders(),
      });
      
      if (response.ok) {
        const data = await response.json();
        return data.servers || [];
      }
      
      // If backend endpoint fails, check MCP servers directly
      console.warn('Failed to get MCP servers from backend, checking directly...');
      return this.checkMCPServersDirectly();
      
    } catch (error) {
      console.error('Failed to get MCP servers:', error);
      return this.checkMCPServersDirectly();
    }
  }

  private async checkMCPServersDirectly(): Promise<MCPServer[]> {
    const servers: MCPServer[] = [];
    
    // Check base MCP server
    if (process.env.NEXT_PUBLIC_MCP_BASE_URL) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout
        
        const response = await fetch(`${process.env.NEXT_PUBLIC_MCP_BASE_URL}/health`, {
          headers: {
            'Authorization': `Bearer ${process.env.NEXT_PUBLIC_MCP_API_KEY}`,
          },
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        servers.push({
          name: 'mcp_base_server',
          url: process.env.NEXT_PUBLIC_MCP_BASE_URL,
          tools: ['list_banking_apis', 'get_api_structure', 'invoke_banking_api'],
          status: response.ok ? 'connected' : 'disconnected'
        });
      } catch (error) {
        servers.push({
          name: 'mcp_base_server',
          url: process.env.NEXT_PUBLIC_MCP_BASE_URL,
          tools: ['list_banking_apis', 'get_api_structure', 'invoke_banking_api'],
          status: 'error',
          error: (error as Error).message
        });
      }
    }
    
    // Check SSE server (different endpoint path)
    if (process.env.NEXT_PUBLIC_MCP_SSE_ENDPOINT) {
      servers.push({
        name: 'mcp_sse_server',
        url: process.env.NEXT_PUBLIC_MCP_SSE_ENDPOINT,
        tools: ['event_stream', 'real_time_updates'],
        status: 'connected' // SSE doesn't have a simple health check
      });
    }
    
    return servers;
  }

  async getMCPStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/mcp/status`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        return {
          mcp_enabled: false,
          connection_status: 'disconnected',
          available_tools: [],
          error: 'Failed to get MCP status'
        };
      }
      
      return response.json();
    } catch (error) {
      return {
        mcp_enabled: false,
        connection_status: 'disconnected',
        available_tools: [],
        error: (error as Error).message
      };
    }
  }

  async suggestTools(agentRole: string, instructions: string): Promise<string[]> {
    const suggestions: Record<string, string[]> = {
      researcher: ['web_search', 'document_parser', 'mcp:research_server:deep_search'],
      analyzer: ['data_analyzer', 'chart_generator', 'mcp:analytics_server:analyze'],
      writer: ['markdown_formatter', 'grammar_checker', 'mcp:writing_server:enhance'],
      coordinator: ['task_router', 'status_tracker', 'mcp:workflow_server:coordinate'],
      specialist: ['data_processor', 'api_caller', 'database_query'],
      reviewer: ['quality_checker', 'approval_tool', 'feedback_generator'],
    };
    return suggestions[agentRole] || ['generic_tool'];
  }
}

// ===== Enhanced Agent Node Component =====
const EnhancedAgentNode: React.FC<NodeProps> = ({ data, isConnectable, selected }) => {
  
  console.log('AgentNode data:', data);
  const roleColors: Record<string, string> = {
    coordinator: '#3b82f6',
    researcher: '#10b981',
    analyzer: '#f59e0b',
    writer: '#8b5cf6',
    specialist: '#ef4444',
    reviewer: '#6366f1',
    customer_support: '#ec4899',
    generic: '#6b7280',
    custom: '#6b7280',
  };

  const bgColor = roleColors[data.role] || '#6b7280';
  const status = data.status || 'idle';

  const statusIcons: Record<string, React.ReactNode> = {
    idle: null,
    thinking: <Brain className="animate-pulse" size={14} />,
    executing: <Zap className="animate-spin" size={14} />,
    error: <XCircle size={14} />,
    complete: <CheckCircle size={14} />,
  };

  const statusColors: Record<string, string> = {
    idle: 'transparent',
    thinking: '#f59e0b',
    executing: '#3b82f6',
    error: '#ef4444',
    complete: '#10b981',
  };

    // Add this - Control type badges definition
  const controlTypeBadges: Record<string, { color: string; text: string }> = {
    retain: { color: 'rgba(59, 130, 246, 0.2)', text: 'Retain' },
    relinquish_to_parent: { color: 'rgba(239, 68, 68, 0.2)', text: 'Relinquish' },
    start_agent: { color: 'rgba(16, 185, 129, 0.2)', text: 'Start Agent' },
  };

  const roleIcons: Record<string, React.ReactNode> = {
    coordinator: <Network size={16} />,
    researcher: <Globe size={16} />,
    analyzer: <BarChart3 size={16} />,
    writer: <FileText size={16} />,
    specialist: <Code size={16} />,
    reviewer: <CheckCircle size={16} />,
    custom: <Cpu size={16} />,
  };

  const isExecuting = status === 'executing' || status === 'thinking';

  return (
    <div
      style={{
        background: '#1e1e1e',
        border: `2px solid ${selected ? '#0b57d0' : bgColor}`,
        borderRadius: '12px',
        minWidth: '180px',
        maxWidth: '220px',
        boxShadow: selected 
          ? '0 0 0 2px rgba(11, 87, 208, 0.3), 0 4px 12px rgba(0, 0, 0, 0.3)' 
          : '0 2px 8px rgba(0, 0, 0, 0.2)',
        transition: 'all 0.2s ease',
        position: 'relative',
        overflow: 'visible',
      }}
    >
      {/* Status indicator */}
      {isExecuting && (
        <div
          style={{
            position: 'absolute',
            top: -8,
            right: -8,
            width: 16,
            height: 16,
            background: '#0b57d0',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(11, 87, 208, 0.4)',
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              background: 'white',
              borderRadius: '50%',
              animation: 'pulse 1.5s ease-in-out infinite',
            }}
          />
        </div>
      )}

      <Handle
        type="target"
        position={Position.Left}
        style={{
          background: bgColor,
          width: 12,
          height: 12,
          border: '2px solid #1e1e1e',
          left: -6,
        }}
        isConnectable={isConnectable}
      />

      {/* Header with icon */}
      <div
        style={{
          background: bgColor,
          padding: '8px 12px',
          borderRadius: '10px 10px 0 0',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <div style={{ color: 'white', opacity: 0.9 }}>
          {roleIcons[data.role] || roleIcons.custom}
        </div>
        <span
          style={{
            color: 'white',
            fontSize: '13px',
            fontWeight: 600,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {data.name || data.label || 'Unnamed'}
        </span>
      </div>

           {/* Body */}
      <div style={{ padding: '10px 12px' }}>
        {/* Model info */}
        {data.model && (
          <div
            style={{
              fontSize: '11px',
              color: '#9aa0a6',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Cpu size={10} />
            <span>{data.model.split('/').pop()}</span>
          </div>
        )}

        {/* Output visibility badge */}
        {data.outputVisibility && (
          <div
            style={{
              fontSize: '10px',
              padding: '3px 8px',
              backgroundColor: data.outputVisibility === 'internal' ? '#2e2e2e' : bgColor + '20',
              color: data.outputVisibility === 'internal' ? '#9aa0a6' : bgColor,
              borderRadius: '4px',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              marginBottom: '8px',
              border: `1px solid ${bgColor}30`,
            }}
          >
            {data.outputVisibility === 'internal' ? <EyeOff size={10} /> : <Eye size={10} />}
            {data.outputVisibility}
          </div>
        )}

        {/* Tools count */}
        {data.tools && data.tools.length > 0 && (
          <div
            style={{
              fontSize: '11px',
              color: '#9aa0a6',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Terminal size={10} />
            <span>{data.tools.length} tools</span>
          </div>
        )}

        {/* Connected agents count */}
        {data.connectedAgents && data.connectedAgents.length > 0 && (
          <div
            style={{
              fontSize: '11px',
              color: '#9aa0a6',
              marginTop: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Network size={10} />
            <span>{data.connectedAgents.length} connections</span>
          </div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        style={{
          background: bgColor,
          width: 12,
          height: 12,
          border: '2px solid #1e1e1e',
          right: -6,
        }}
        isConnectable={isConnectable}
      />
    </div>
  );
}; 


      {/* Add Model Display 
      {data.model && (
        <div style={{ 
          fontSize: '11px', 
          opacity: 0.8,
          marginBottom: '6px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          <Cpu size={12} />
          <span style={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}>
            {data.model.split('/').pop() || data.model}
          </span>
        </div>
      )}

     Add Temperature Display 
      {data.temperature !== undefined && (
        <div style={{ 
          fontSize: '10px', 
          opacity: 0.7,
          marginBottom: '4px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          <span>Temp: {data.temperature}</span>
          {data.max_tokens && <span>• Max tokens: {data.max_tokens}</span>}
        </div>
      )}

      Control Type Badge
      {data.controlType && controlTypeBadges[data.controlType] && (
        <div style={{ 
          fontSize: '10px', 
          padding: '2px 6px', 
          backgroundColor: controlTypeBadges[data.controlType].color,
          borderRadius: '3px',
          display: 'inline-block',
          marginBottom: '4px',
          border: '1px solid rgba(0,0,0,0.1)'
        }}>
          {controlTypeBadges[data.controlType].text}
        </div>
      )} */}

      {/* Output Visibility
      {data.outputVisibility && (
        <div style={{ 
          fontSize: '10px', 
          padding: '2px 6px', 
          backgroundColor: data.outputVisibility === 'internal' ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.2)',
          borderRadius: '3px',
          display: 'inline-block',
          marginBottom: '4px'
        }}>
          {data.outputVisibility === 'internal' ? <EyeOff size={10} className="inline mr-1" /> : <Eye size={10} className="inline mr-1" />}
          {data.outputVisibility}
        </div>
      )} 

      Connected Agents
      {data.connectedAgents && data.connectedAgents.length > 0 && (
        <div style={{ 
          fontSize: '11px', 
          marginTop: '8px',
          padding: '4px 0',
          borderTop: '1px solid rgba(0,0,0,0.1)'
        }}>
          <div style={{ marginBottom: '4px', opacity: 0.8 }}>
            Connected to: {data.connectedAgents.join(', ')}
          </div>
        </div>
      )}

       Tools section 
      {data.tools && data.tools.length > 0 && (
        <div style={{ 
          fontSize: '11px', 
          marginTop: '8px',
          padding: '4px 0',
          borderTop: '1px solid rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
            <Terminal size={12} />
            <span>Tools ({data.tools.length}):</span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {data.tools.slice(0, 3).map((tool: string, index: number) => {
              const isMCPTool = tool.startsWith('mcp:');
              const toolName = tool.split(':').pop() || tool;
              
              return (
                <span
                  key={index}
                  style={{
                    backgroundColor: isMCPTool ? 'rgba(34, 197, 94, 0.2)' : 'rgba(0,0,0,0.1)',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    fontSize: '10px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '2px',
                    border: isMCPTool ? '1px solid rgba(34, 197, 94, 0.3)' : '1px solid rgba(0,0,0,0.2)',
                  }}
                  title={tool}
                >
                  {isMCPTool && <Server size={8} style={{ flexShrink: 0 }} />}
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '80px' }}>
                    {toolName}
                  </span>
                </span>
              );
            })}
            {data.tools.length > 3 && (
              <span style={{ fontSize: '10px', opacity: 0.7 }}>+{data.tools.length - 3} more</span>
            )}
          </div>
        </div>
      )} */}

      {/* Status indicator
      {status !== 'idle' && (
        <div
          style={{
            position: 'absolute',
            bottom: '-4px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '60%',
            height: '2px',
            backgroundColor: statusColors[status],
            borderRadius: '1px',
          }}
        />
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#555' }}
        isConnectable={isConnectable}
      />
    </div>
  );
};*/}

// ===== MCP Status Indicator Component =====
const MCPStatusIndicator: React.FC = () => {
  const [mcpStatus, setMcpStatus] = useState<'checking' | 'connected' | 'partial' | 'disconnected'>('checking');
  const [showDetails, setShowDetails] = useState(false);
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  useEffect(() => {
    const checkMCPStatus = async () => {
      try {
        const [status, servers] = await Promise.all([
          api.getMCPStatus(),
          api.getMCPServers()
        ]);
        
        setMcpServers(servers);
        
        if (status.connection_status === 'connected') {
          setMcpStatus('connected');
        } else if (status.mcp_enabled) {
          setMcpStatus('partial');
        } else {
          setMcpStatus('disconnected');
        }
      } catch (error) {
        console.error('Failed to check MCP status:', error);
        setMcpStatus('disconnected');
      }
    };

    checkMCPStatus();
    const interval = setInterval(checkMCPStatus, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [api]);

  const statusColors = {
    connected: 'bg-green-500',
    partial: 'bg-yellow-500',
    disconnected: 'bg-red-500',
    checking: 'bg-gray-500'
  };

  const statusIcons = {
    connected: <Wifi className="w-4 h-4" />,
    partial: <WifiOff className="w-4 h-4" />,
    disconnected: <WifiOff className="w-4 h-4" />,
    checking: <Loader2 className="w-4 h-4 animate-spin" />
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
      >
        <div className={`w-2 h-2 rounded-full ${statusColors[mcpStatus]}`} />
        <span className="text-sm font-medium">MCP</span>
        {statusIcons[mcpStatus]}
        <ChevronDown className={`w-3 h-3 transition-transform ${showDetails ? 'rotate-180' : ''}`} />
      </button>

      {showDetails && (
        <div className="absolute top-full right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 p-4 z-50">
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <Server className="w-4 h-4" />
            MCP Server Status
          </h4>
          
          <div className="space-y-2">
            {mcpServers.length > 0 ? (
              mcpServers.map((server) => (
                <div key={server.name} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${
                      server.status === 'connected' ? 'bg-green-500' :
                      server.status === 'available' ? 'bg-yellow-500' :
                      server.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
                    }`} />
                    <span className="text-sm font-medium">{server.name}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-gray-500">{server.tools.length} tools</span>
                    {server.error && (
                      <span className="text-xs text-red-500" title={server.error}>
                        <AlertCircle className="w-3 h-3" />
                      </span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500">No MCP servers configured</p>
            )}
          </div>

          <div className="mt-3 pt-3 border-t border-gray-200">
            <p className="text-xs text-gray-500">
              Status: {mcpStatus === 'connected' ? 'All systems operational' :
                       mcpStatus === 'partial' ? 'Some servers unavailable' :
                       mcpStatus === 'disconnected' ? 'No connection' : 'Checking...'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

// ===== MCP Tool Selector Component =====
const MCPToolSelector: React.FC<{ 
  selectedTools: string[], 
  onToolsChange: (tools: string[]) => void 
}> = ({ selectedTools, onToolsChange }) => {
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  useEffect(() => {
    const loadMCPServers = async () => {
      try {
        const servers = await api.getMCPServers();
        setMcpServers(servers);
      } catch (error) {
        console.error('Failed to load MCP servers:', error);
      } finally {
        setLoading(false);
      }
    };
    loadMCPServers();
  }, [api]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-4">
        <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-2">
        <Server className="w-4 h-4 text-gray-600" />
        <span className="font-medium text-gray-700">MCP Tools</span>
      </div>
      
      {mcpServers.length === 0 ? (
        <p className="text-sm text-gray-500 italic">No MCP servers available</p>
      ) : (
        mcpServers.map(server => (
          <div key={server.name} className="border rounded-lg p-3 bg-gray-50">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${
                server.status === 'connected' ? 'bg-green-500' : 
                server.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
              }`} />
              <span className="font-medium text-sm">{server.name}</span>
              <span className="text-xs text-gray-500">({server.status})</span>
            </div>
            
            {server.status === 'connected' ? (
              <div className="grid grid-cols-2 gap-2">
                {server.tools.map(tool => {
                  const toolId = `mcp:${server.name}:${tool}`;
                  return (
                    <label key={tool} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-white p-1 rounded">
                      <input
                        type="checkbox"
                        checked={selectedTools.includes(toolId)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            onToolsChange([...selectedTools, toolId]);
                          } else {
                            onToolsChange(selectedTools.filter(t => t !== toolId));
                          }
                        }}
                        className="rounded border-gray-300"
                      />
                      <span className="text-gray-700">{tool}</span>
                    </label>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-gray-500 italic">
                {server.error || 'Server not available'}
              </p>
            )}
          </div>
        ))
      )}
    </div>
  );
};

// ===== Workflow Test Panel Component =====
const WorkflowTestPanel: React.FC<{ workflowId: string }> = ({ workflowId }) => {
  const [testInput, setTestInput] = useState('');
  const [testResults, setTestResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  const runTest = async () => {
    if (!testInput.trim()) return;
    
    setIsRunning(true);
    setTestResults(null);
    
    try {
      const results = await api.testWorkflow(workflowId, testInput);
      setTestResults(results);
      setShowTrace(true);
    } catch (error) {
      setTestResults({
        success: false,
        error: (error as Error).message
      });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-gray-900 p-4 rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Play className="w-4 h-4" />
          Test Workflow
        </h3>
        {testResults && (
          <button
            onClick={() => setShowTrace(!showTrace)}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            {showTrace ? 'Hide' : 'Show'} Trace
          </button>
        )}
      </div>
      
      <div className="flex gap-2">
        <input 
          value={testInput}
          onChange={(e) => setTestInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && runTest()}
          placeholder="Enter test input..."
          className="flex-1 p-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
          disabled={isRunning}
        />
        <button 
          onClick={runTest} 
          disabled={isRunning || !testInput.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {isRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run Test
            </>
          )}
        </button>
      </div>
      
      {testResults && (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-2">
            {testResults.success ? (
              <CheckCircle className="text-green-500 w-5 h-5" />
            ) : (
              <XCircle className="text-red-500 w-5 h-5" />
            )}
            <span className="text-white font-medium">
              Test {testResults.success ? 'Passed' : 'Failed'}
            </span>
          </div>
          
          {showTrace && testResults.trace && (
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {testResults.trace.map((step: any, i: number) => (
                <div key={i} className="bg-gray-800 p-3 rounded-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-blue-400 font-medium">{step.agent}</span>
                    <span className="text-gray-500 text-sm">→</span>
                    <span className="text-gray-400 text-sm">{step.action}</span>
                  </div>
                  <div className="text-gray-300 text-sm pl-4">{step.output}</div>
                </div>
              ))}
            </div>
          )}
          
          {testResults.error && (
            <div className="bg-red-900/20 border border-red-500/50 rounded p-3 mt-2">
              <p className="text-red-400 text-sm">{testResults.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ===== 1. Enhanced Streaming Workflow Builder =====
interface StreamingWorkflowBuilderProps {
  onWorkflowCreated?: (workflow: WorkflowDefinition) => void;
}

const StreamingWorkflowBuilder: React.FC<StreamingWorkflowBuilderProps> = ({ onWorkflowCreated }) => {
  const [description, setDescription] = useState('');
  const [isBuilding, setIsBuilding] = useState(false);
  const [createdWorkflow, setCreatedWorkflow] = useState<WorkflowDefinition | null>(null);
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [buildProgress, setBuildProgress] = useState(0);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  const buildWorkflow = async () => {
    if (!description.trim()) return;
    
    setIsBuilding(true);
    setStreamEvents([]);
    setBuildProgress(0);
    setCreatedWorkflow(null);
    
    try {
      const events: StreamEvent[] = [];
      
      for await (const event of api.createWorkflowFromDescriptionStream(description)) {
        events.push({ ...event, timestamp: Date.now() });
        setStreamEvents([...events]);
        
        if (event.type === 'progress' && event.data.progress) {
          setBuildProgress(event.data.progress);
        }
        
        if (event.type === 'complete' && event.data.workflow_id) {
          const workflow = await api.getWorkflow(event.data.workflow_id);
          setCreatedWorkflow(workflow);
          onWorkflowCreated?.(workflow);
        }
      }
      
      setDescription('');
    } catch (error) {
      console.error('Failed to build workflow:', error);
      alert('Failed to build workflow. Please try again.');
    } finally {
      setIsBuilding(false);
      setBuildProgress(0);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <Sparkles className="text-yellow-500" />
        Natural Language Workflow Builder
      </h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Describe your workflow in plain English
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Example: Create a customer support workflow that classifies incoming requests and routes them to billing or technical support agents based on the content..."
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={4}
          />
        </div>
        
        <button
          onClick={buildWorkflow}
          disabled={isBuilding || !description.trim()}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {isBuilding ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              Building Workflow...
            </>
          ) : (
            <>
              <Zap size={20} />
              Build Workflow
            </>
          )}
        </button>

        {/* Progress Bar */}
        {buildProgress > 0 && buildProgress < 100 && (
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${buildProgress}%` }}
            />
          </div>
        )}

        {/* Stream Events Display */}
        {streamEvents.length > 0 && (
          <div className="mt-4 bg-gray-100 rounded-lg p-4 max-h-40 overflow-y-auto">
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Activity size={16} />
              Build Events
            </h4>
            <div className="space-y-1 text-sm">
              {streamEvents.map((event, i) => (
                <div key={i} className="flex items-start gap-2">
                  <span className="text-gray-500 text-xs">
                    {event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : ''}
                  </span>
                  <div className="flex-1">
                    {event.type === 'progress' && (
                      <span className="text-blue-600">
                        {event.data.message}
                      </span>
                    )}
                    {event.type === 'agent' && (
                      <span className="text-green-600">
                        Created agent: <strong>{event.data.name}</strong> ({event.data.role})
                      </span>
                    )}
                    {event.type === 'tool' && (
                      <span className="text-purple-600">
                        Added tool: {event.data.name} to {event.data.agent}
                      </span>
                    )}
                    {event.type === 'edge' && (
                      <span className="text-orange-600">
                        Connected: {event.data.from_agent} → {event.data.to_agent}
                      </span>
                    )}
                    {event.type === 'complete' && (
                      <span className="text-green-600 font-semibold">
                        ✓ Workflow created successfully!
                      </span>
                    )}
                    {event.type === 'error' && (
                      <span className="text-red-600">
                        Error: {event.data.error}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {createdWorkflow && (
        <div className="mt-6 p-4 bg-green-50 rounded-lg">
          <h3 className="font-semibold mb-2 text-green-800">
            ✓ Created Workflow: {createdWorkflow.name}
          </h3>
          <p className="text-gray-600 mb-4">{createdWorkflow.description}</p>
          <div className="space-y-2">
            <h4 className="font-medium">Agents ({createdWorkflow.agents.length}):</h4>
            {createdWorkflow.agents.map((agent, index) => (
              <div key={index} className="pl-4 text-sm flex items-center gap-2">
                <Brain size={14} className="text-blue-500" />
                <span className="font-medium">{agent.name}</span> - {agent.role}
                {agent.tools && agent.tools.length > 0 && (
                  <span className="text-gray-500">({agent.tools.length} tools)</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ===== Error Boundary for ReactFlow =====
class ReactFlowErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true };
  }

  componentDidCatch(error: any, errorInfo: any) {
    console.error('ReactFlow Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
          <div className="text-center">
            <AlertCircle className="mx-auto text-red-500 mb-4" size={48} />
            <p className="text-gray-600">Failed to load workflow editor</p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

  // Memoize node types
const nodeTypes = {
  agent: EnhancedAgentNode,
};

// Add custom edge type for n8n-style connections
const customEdgeTypes = {
  smoothstep: (props: any) => {
    const { sourceX, sourceY, targetX, targetY, markerEnd } = props;
    const [edgePath] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
    });

    return (
      <>
        <path
          id={props.id}
          style={props.style}
          className="react-flow__edge-path"
          d={edgePath}
          markerEnd={markerEnd}
        />
      </>
    );
  },
};


// ===== Part 1: Complete EnhancedWorkflowVisualEditor =====
const EnhancedWorkflowVisualEditor: React.FC<{ workflowId?: string }> = ({ workflowId }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const [showEnhancements, setShowEnhancements] = useState(false);
  const [enhancements, setEnhancements] = useState<any>(null);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);



  // Initialize with default nodes if no workflow
  useEffect(() => {
    const initializeEditor = async () => {
      setIsLoading(true);
      
      if (workflowId) {
        await loadWorkflowVisualization();
      } else if (nodes.length === 0) {
        // Create default nodes for new workflow
        const defaultNodes: ROWBOATNode[] = [
          {
            id: '1',
            type: 'agent',
            position: { x: 250, y: 50 },
            data: {
              name: 'coordinator',
              role: 'coordinator',
              hasInstructions: false,
              connectedAgents: [],
              tools: [],
              status: 'idle',
              label: 'Coordinator Agent',
              instructions: 'Coordinate workflow tasks and delegate to appropriate agents',
              outputVisibility: 'user_facing',
              model: 'claude-3-opus-20240229'
            }
          },
          {
            id: '2',
            type: 'agent',
            position: { x: 250, y: 200 },
            data: {
              name: 'worker',
              role: 'specialist',
              hasInstructions: false,
              connectedAgents: [],
              tools: [],
              status: 'idle',
              label: 'Worker Agent',
              instructions: 'Execute specialized tasks as assigned by the coordinator',
              outputVisibility: 'user_facing',
              model: 'claude-3-opus-20240229'
            }
          }
        ];
        
        const defaultEdges: ROWBOATEdge[] = [
          {
            id: 'e1-2',
            source: '1',
            target: '2',
            type: 'smoothstep',
            animated: true,
            markerEnd: {
              type: MarkerType.ArrowClosed,
            }
          }
        ];
        
        setNodes(defaultNodes);
        setEdges(defaultEdges);
      }
      
      await loadMCPServers();
      setIsLoading(false);
    };
    
    initializeEditor();
  }, [workflowId]);

  const loadWorkflowVisualization = async () => {
    if (!workflowId) return;
    
    try {
      const viz = await api.getVisualization(workflowId);
      
      console.log('Visualization response:', viz);
      
      if (!viz.nodes || viz.nodes.length === 0) {
        console.warn('Empty visualization received');
        // Create default nodes
        const defaultNodes: ROWBOATNode[] = [
          {
            id: '1',
            type: 'agent',
            position: { x: 250, y: 50 },
            data: {
              name: 'coordinator',
              role: 'coordinator',
              hasInstructions: true,
              connectedAgents: [],
              tools: [],
              status: 'idle',
              label: 'Coordinator Agent',
              instructions: 'Coordinate workflow tasks',
              outputVisibility: 'user_facing',
              model: 'claude-3-opus-20240229'
            }
          }
        ];
        setNodes(defaultNodes);
        return;
      }
      
      // Get the full workflow details to enrich the visualization
      let agentDetails: Record<string, AgentConfig> = {};
      try {
        const workflow = await api.getWorkflow(workflowId);
        agentDetails = workflow.agents.reduce((acc, agent) => {
          acc[agent.name] = agent;
          return acc;
        }, {} as Record<string, AgentConfig>);
      } catch (workflowError) {
        console.error('Failed to get workflow details, using visualization data only:', workflowError);
      }
      
      // Process nodes with enriched data
      const processedNodes = viz.nodes.map((n, index) => {
        // Get the full agent details from the workflow
        const agentDetail = agentDetails[n.data.label] || {};
        
        const processedNode: ROWBOATNode = {
          id: n.id,
          type: 'agent', // Force type to 'agent' instead of 'default'
          position: n.position,
          style: n.style, // Now TypeScript knows about this property
          data: {
            // Basic fields from visualization
            name: n.data.label,
            label: n.data.label,
            role: n.data.role || 'custom',
            hasInstructions: n.data.hasInstructions || false,
            connectedAgents: n.data.connectedAgents || [],
            
            // Enrich with data from workflow details
            instructions: agentDetail.instructions || '',
            tools: agentDetail.tools || [],
            model: agentDetail.model || 'claude-3-opus-20240229',
            // Fixed: use visibility from node data, outputVisibility from agent details
            outputVisibility: n.data.visibility || agentDetail.outputVisibility || 'user_facing',
            description: agentDetail.description || '',
            temperature: agentDetail.temperature,
            max_tokens: agentDetail.max_tokens,
            
            // Control type from visualization
            controlType: n.data.controlType,
            
            // Default values
            status: 'idle',
            metrics: undefined,
            
            // Include any metadata
            metadata: {
              ...agentDetail.metadata,
              style: n.style // Preserve the color styling
            }
          }
        };
        
        console.log(`Processed node ${index}:`, processedNode);
        return processedNode;
      });
      
      setNodes(processedNodes);
      setEdges(viz.edges || []);
    } catch (error) {
      console.error('Failed to load visualization:', error);
      
      // Set some default nodes on complete failure
      const errorNodes: ROWBOATNode[] = [
        {
          id: 'error',
          type: 'agent',
          position: { x: 250, y: 150 },
          data: {
            name: 'Error',
            label: 'Error Loading Workflow',
            role: 'generic',
            hasInstructions: false,
            connectedAgents: [],
            tools: [],
            status: 'error',
            instructions: 'Failed to load workflow visualization',
            outputVisibility: 'user_facing',
            model: ''
          }
        }
      ];
      setNodes(errorNodes);
    }
  };

  const loadMCPServers = async () => {
    try {
      const servers = await api.getMCPServers();
      setMcpServers(servers);
    } catch (error) {
      console.error('Failed to load MCP servers:', error);
    }
  };

  const addNewAgent = () => {
    const newNode: ROWBOATNode = {
      id: `agent-${Date.now()}`,
      type: 'agent',
      position: { x: 250, y: 100 + nodes.length * 150 },
      data: {
        name: `agent_${nodes.length + 1}`,
        role: 'specialist',
        instructions: '',
        tools: [],
        status: 'idle',
        label: `Agent ${nodes.length + 1}`,
        hasInstructions: false,
        connectedAgents: [],
        outputVisibility: 'user_facing',
        model: 'claude-3-opus-20240229'
      }
    };
    setNodes(nds => [...nds, newNode]);
  };

  const detectMentionsAndCreateEdges = useCallback(() => {
    const mentionRegex = /@(\w+)/g;
    const newEdges: Edge[] = [];
    
    nodes.forEach(node => {
      const matches = (node.data.instructions || '').matchAll(mentionRegex);
      for (const match of matches) {
        const targetAgentName = match[1];
        const targetNode = nodes.find(n => n.data.name === targetAgentName);
        
        if (targetNode && !edges.some(e => 
          e.source === node.id && 
          e.target === targetNode.id && 
          e.data?.isMentionBased
        )) {
          newEdges.push({
            id: `mention-${node.id}-${targetNode.id}`,
            source: node.id,
            target: targetNode.id,
            type: 'smoothstep',
            animated: true,
            style: { stroke: '#00a67e', strokeWidth: 2 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: '#00a67e',
            },
            data: { isMentionBased: true }
          });
        }
      }
    });
    
    if (newEdges.length > 0) {
      setEdges((eds) => {
        const nonMentionEdges = eds.filter(e => !e.data?.isMentionBased);
        return [...nonMentionEdges, ...newEdges];
      });
    }
  }, [nodes, edges, setEdges]);

  useEffect(() => {
    detectMentionsAndCreateEdges();
  }, [nodes, detectMentionsAndCreateEdges]);

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            type: 'smoothstep',
            animated: true,
            markerEnd: {
              type: MarkerType.ArrowClosed,
            },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event: any, node: any) => {
    setSelectedNode(node.id);
  }, []);

  const updateNodeData = (nodeId: string, data: Partial<ROWBOATNodeData>) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: {
              ...node.data,
              ...data,
            },
          };
        }
        return node;
      })
    );
  };

  const saveWorkflow = async () => {
    const workflow: WorkflowDefinition = {
      name: workflowId || `workflow-${Date.now()}`,
      description: 'Visual workflow created with ROWBOAT',
      agents: nodes.map(node => ({
        name: node.data.name,
        role: node.data.role,
        instructions: node.data.instructions || '',
        tools: node.data.tools || [],
        model: node.data.model,
        outputVisibility: node.data.outputVisibility,
        mcp_servers: node.data.mcpServers
      })),
      edges: edges.map(edge => ({
        from_agent: nodes.find(n => n.id === edge.source)?.data.name || '',
        to_agent: nodes.find(n => n.id === edge.target)?.data.name || '',
        condition: edge.data?.condition,
        isMentionBased: edge.data?.isMentionBased
      })),
      entry_point: nodes[0]?.data.name || 'coordinator',
      metadata: {
        visualization: { nodes, edges }
      }
    };

    console.log('Saving workflow:', workflow);
    // TODO: Implement actual save API call
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg shadow-lg overflow-hidden" style={{ minHeight: '600px', height: '600px' }}>
      <div className="p-4 border-b border-gray-700 bg-gray-800 flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2 text-white">
          <GitBranch className="text-blue-400" />
          Visual Workflow Editor
        </h3>
        <div className="flex items-center gap-2">
          <MCPStatusIndicator />
          <button
            onClick={addNewAgent}
            className="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Node
          </button>
          <button
            onClick={() => setShowTestPanel(!showTestPanel)}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Test
          </button>
          <button
            onClick={saveWorkflow}
            className="px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
        </div>
      </div>

      <div className="flex h-full" style={{ height: 'calc(100% - 64px)' }}>
        {/* <div className="flex-1 relative bg-white" style={{ 
          backgroundImage: 'radial-gradient(circle at 1px 1px, rgb(50, 50, 50) 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}> */}
        <div className="flex-1 relative" style={{ minHeight: '500px', height: '100%' }}>
          <ReactFlowErrorBoundary>
            <div style={{ width: '100%', height: '100%' }}> {/* Add this wrapper */}
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              edgeTypes={customEdgeTypes}  // Add this line
              fitView={false}
              fitViewOptions={{ 
                padding: 0.2,
                includeHiddenNodes: false,
                minZoom: 0.5,
                maxZoom: 1.5
              }}
              defaultViewport={{ x: 100, y: 100, zoom: 0.8 }}
              minZoom={0.1}
              maxZoom={2}
              // style={{ background: 'grey' }}
              defaultEdgeOptions={{
                type: 'smoothstep',
                animated: true,
                style: {
                  stroke: '#4b5563',
                  strokeWidth: 2,
                },
                markerEnd: {
                  type: MarkerType.ArrowClosed,
                  color: '#4b5563',
                  width: 20,
                  height: 20,
                },
              }}
              connectionLineStyle={{
                stroke: '#4b5563',
                strokeWidth: 2,
              }}
              proOptions={{ hideAttribution: true }}
              className="react-flow-n8n-style"
            >
              <Background 
                variant={BackgroundVariant.Dots} 
                gap={12} 
                size={1} 
                color="#374151"
              />
              <Controls 
                className="react-flow-controls-n8n"
                showZoom={true}
                showFitView={true}
                showInteractive={false}
              />
              {/* Use Panel for better positioning */}
              <Panel position="top-right" style={{ margin: '10px' }}>
                <MiniMap 
                  style={{
                    width: 150,
                    height: 100,
                    backgroundColor: '#f8f8f8',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  }}
                  nodeColor={(node) => {
                    const roleColors: Record<string, string> = {
                      coordinator: '#3b82f6',
                      researcher: '#10b981',
                      analyzer: '#f59e0b',
                      writer: '#8b5cf6',
                      specialist: '#ef4444',
                      reviewer: '#6366f1',
                      customer_support: '#ec4899',
                      custom: '#9333ea',
                      generic: '#6b7280',
                    };
                    return roleColors[node.data?.role] || '#6b7280';
                  }}
                  nodeStrokeWidth={3}
                  zoomable
                  pannable
                />
              </Panel>
            </ReactFlow>
            </div>
          </ReactFlowErrorBoundary>
        </div>

        {selectedNode && (
          <AgentPropertyPanel
            node={nodes.find(n => n.id === selectedNode)!}
            onUpdate={(data) => updateNodeData(selectedNode, data)}
            onClose={() => setSelectedNode(null)}
            mcpServers={mcpServers}
          />
        )}
      </div>

      {showTestPanel && workflowId && (
        <div className="absolute bottom-4 right-4 w-96 z-50">
          <WorkflowTestPanel workflowId={workflowId} />
        </div>
      )}
    </div>
  );
};

// ===== Part 2: Agent Property Panel =====
interface AgentPropertyPanelProps {
  node: ROWBOATNode;
  onUpdate: (data: Partial<ROWBOATNodeData>) => void;
  onClose: () => void;
  mcpServers: MCPServer[];
}

const AgentPropertyPanel: React.FC<AgentPropertyPanelProps> = ({ 
  node, 
  onUpdate, 
  onClose,
  mcpServers 
}) => {
  const [localData, setLocalData] = useState(node.data);
  const [showToolSelector, setShowToolSelector] = useState(false);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  useEffect(() => {
    setLocalData(node.data);
  }, [node]);

  const handleChange = (field: keyof ROWBOATNodeData, value: any) => {
    const newData = { ...localData, [field]: value };
    setLocalData(newData);
    onUpdate({ [field]: value });
  };

  const suggestTools = async () => {
    const suggested = await api.suggestTools(localData.role, localData.instructions);
    const currentTools = localData.tools || [];
    const newTools = [...new Set([...currentTools, ...suggested])];
    handleChange('tools', newTools);
  };

    // Common model options
  const modelOptions = [
    { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
    { value: 'claude-3-sonnet-20240229', label: 'Claude 3 Sonnet' },
    { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku' },
    { value: 'deepseek-ai/DeepSeek-V3-0324', label: 'DeepSeek V3' },
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
  ];
  
  // Check if current model is in the list, if not add it
  const currentModel = localData.model || '';
  const isCustomModel = !modelOptions.find(opt => opt.value === currentModel);
  
  return (
    <div className="w-96 bg-white border-l shadow-lg p-4 overflow-y-auto" style={{ maxHeight: 'calc(100% - 64px)' }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Agent Properties</h3>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 rounded transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Agent Name</label>
          <input
            type="text"
            value={localData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
          <select
            value={localData.role}
            onChange={(e) => handleChange('role', e.target.value)}
            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="coordinator">Coordinator</option>
            <option value="researcher">Researcher</option>
            <option value="analyzer">Analyzer</option>
            <option value="writer">Writer</option>
            <option value="specialist">Specialist</option>
            <option value="reviewer">Reviewer</option>
            <option value="custom">Custom</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
          <select
            value={localData.model || 'claude-3-opus-20240229'}
            onChange={(e) => handleChange('model', e.target.value)}
            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {/* Add custom model if it's not in the standard list */}
            {isCustomModel && currentModel && (
              <option value={currentModel}>{currentModel}</option>
            )}
            
            {/* Standard model options */}
            {modelOptions.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
            
            {/* Option to enter custom model */}
            <option value="custom">Custom Model...</option>
          </select>
          
          {/* Show input for custom model */}
          {localData.model === 'custom' && (
            <input
              type="text"
              placeholder="Enter custom model name"
              className="w-full mt-2 p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              onChange={(e) => handleChange('model', e.target.value)}
            />
          )}
        </div>

        {/* Temperature and Max Tokens */}
        {(localData.temperature !== undefined || localData.max_tokens !== undefined) && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Temperature</label>
              <input
                type="number"
                value={localData.temperature || 0.7}
                onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                min="0"
                max="2"
                step="0.1"
                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Tokens</label>
              <input
                type="number"
                value={localData.max_tokens || 4096}
                onChange={(e) => handleChange('max_tokens', parseInt(e.target.value))}
                min="1"
                max="32000"
                className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Output Visibility</label>
          <select
            value={localData.outputVisibility || 'user_facing'}
            onChange={(e) => handleChange('outputVisibility', e.target.value as 'user_facing' | 'internal')}
            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="user_facing">User Facing</option>
            <option value="internal">Internal Only</option>
          </select>
        </div>

        {/* Control Type */}
        {localData.controlType && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Control Type</label>
            <select
              value={localData.controlType || 'retain'}
              onChange={(e) => handleChange('controlType', e.target.value)}
              className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="retain">Retain</option>
              <option value="relinquish_to_parent">Relinquish to Parent</option>
              <option value="start_agent">Start Agent</option>
            </select>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Instructions</label>
          <textarea
            value={localData.instructions || ''}
            onChange={(e) => handleChange('instructions', e.target.value)}
            rows={4}
            className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            placeholder="Enter agent instructions... Use @agentname to mention other agents"
          />
          <p className="text-xs text-gray-500 mt-1">
            Tip: Use @agentname to create connections to other agents
          </p>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700">Tools</label>
            <div className="flex gap-2">
              <button
                onClick={suggestTools}
                className="text-xs px-2 py-1 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded transition-colors flex items-center gap-1"
              >
                <Sparkles className="w-3 h-3" />
                Suggest
              </button>
              <button
                onClick={() => setShowToolSelector(!showToolSelector)}
                className="text-xs px-2 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded transition-colors"
              >
                {showToolSelector ? 'Hide' : 'Add'} Tools
              </button>
            </div>
          </div>

          <div className="space-y-2">
            {(localData.tools || []).map((tool, index) => {
              const isMCPTool = tool.startsWith('mcp:');
              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded border"
                >
                  <div className="flex items-center gap-2">
                    {isMCPTool && <Server className="w-3 h-3 text-green-600" />}
                    <span className="text-sm">{tool}</span>
                  </div>
                  <button
                    onClick={() => {
                      const newTools = [...(localData.tools || [])];
                      newTools.splice(index, 1);
                      handleChange('tools', newTools);
                    }}
                    className="text-red-500 hover:text-red-700"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              );
            })}
          </div>

          {showToolSelector && (
            <div className="mt-3 p-3 border rounded-lg bg-gray-50">
              <MCPToolSelector
                selectedTools={localData.tools || []}
                onToolsChange={(tools) => handleChange('tools', tools)}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ===== Part 3: Workflow Execution Panel =====
interface WorkflowExecutionPanelProps {
  workflowId: string;
  onClose?: () => void;
}

const WorkflowExecutionPanel: React.FC<WorkflowExecutionPanelProps> = ({ 
  workflowId, 
  onClose 
}) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ExecutionMessage[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [showInternalMessages, setShowInternalMessages] = useState(false);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<EnhancedROWBOATWebSocket | null>(null);
  const apiRef = useRef<EnhancedROWBOATApiService | null>(null);
  const isConnectedRef = useRef(false);
  const messageIdCounter = useRef(0);
  
  // Generate unique message IDs
  const generateMessageId = (prefix: string) => {
    messageIdCounter.current += 1;
    return `${prefix}-${Date.now()}-${messageIdCounter.current}-${Math.random().toString(36).substr(2, 9)}`;
  };
  
  useEffect(() => {
    let cleanup = false;
    
    // Initialize services only once
    if (!apiRef.current) {
      apiRef.current = new EnhancedROWBOATApiService();
    }
    
    if (!wsRef.current) {
      wsRef.current = new EnhancedROWBOATWebSocket();
      
      const apiKey = process.env.NEXT_PUBLIC_ROWBOAT_API_KEY;
      if (apiKey) {
        wsRef.current.setAuthToken(apiKey);
        console.log('Auth token set on WebSocket instance');
      }
    }
    
    // Prevent double connection
    if (isConnectedRef.current || cleanup) return;
    
    const ws = wsRef.current;
    
    // Remove any existing listeners before adding new ones
    ws.removeAllListeners();
    
    // Connect WebSocket
    ws.connect(workflowId);
    isConnectedRef.current = true; // ADD THIS LINE
    
    // Set up event listeners
    ws.on('execution_started', (data: any) => {
      if (cleanup) return;
      console.log('Workflow execution started:', data);
      setIsExecuting(true);
    });

    ws.on('execution_result', (data: any) => {
      if (cleanup) return;
      console.log('Workflow execution result:', data);
      
      const responseData = data.data;
      if (!responseData) return;
      
      const response = responseData.response || '';
      const agentsUsed = responseData.agents_used || [];
      const messageContent = response || responseData.result?.last_external_message || '';
      
      // Fix: Consistent agent name assignment
      let agentName = 'System';
      if (agentsUsed.length > 0) {
        agentName = agentsUsed[agentsUsed.length - 1];
      } else if (responseData.result?.current_agent) {
        agentName = responseData.result.current_agent;
      }
      
      // Add message to the UI with unique ID
      const message: ExecutionMessage = {
        id: generateMessageId('msg'),
        agent: agentName, // Use the consistently assigned agent name
        content: messageContent, // Use the properly extracted content
        timestamp: new Date().toISOString(),
        type: 'response',
        visibility: 'external',
        metadata: responseData.metadata
      };
      
    // Prevent duplicate messages
    setMessages(prev => {
      // Check if a similar message was just added (within 100ms)
      const recentMessage = prev[prev.length - 1];
      if (recentMessage && 
          recentMessage.content === messageContent &&
          recentMessage.agent === agentName) {
        // Parse timestamp from the message timestamp, not the ID
        const recentTime = new Date(recentMessage.timestamp).getTime();
        const currentTime = new Date().getTime();
        
        if (currentTime - recentTime < 100) {
          console.warn('Duplicate message detected, skipping');
          return prev;
        }
      }
      return [...prev, message];
    });
    });

    ws.on('complete', (data: any) => {
      if (cleanup) return;
      console.log('Workflow execution complete:', data);
      setIsExecuting(false);
      setCurrentAgent(null);
    });

    ws.on('error', (data: any) => {
      if (cleanup) return;
      console.error('WebSocket error:', data);
      setIsExecuting(false);
      
      const errorMessage: ExecutionMessage = {
        id: generateMessageId('error'),
        agent: 'System',
        content: `Error: ${data.error || 'Unknown error occurred'}`,
        timestamp: new Date().toISOString(),
        type: 'error',
        visibility: 'external'
      };
      
      setMessages(prev => [...prev, errorMessage]);
    });
    
    // Cleanup function
    return () => {
      cleanup = true;
      if (wsRef.current && isConnectedRef.current) {
        wsRef.current.removeAllListeners();
        wsRef.current.disconnect();
        isConnectedRef.current = false;
      }
    };
  }, [workflowId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const executeWorkflow = async () => {
    if (!input.trim() || isExecuting || !wsRef.current) return;

    // Add user message to the UI with unique ID
    const userMessage: ExecutionMessage = {
      id: generateMessageId('user'),
      agent: 'User',
      content: input,
      timestamp: new Date().toISOString(),
      type: 'response',
      visibility: 'external'
    };
    setMessages(prev => [...prev, userMessage]);

    // Send message via WebSocket
    wsRef.current.send({
      message: input,
      context: {},
      session_id: `session-${Date.now()}`
    });

    setInput('');
  };

  const filteredMessages = showInternalMessages 
    ? messages 
    : messages.filter(m => m.visibility !== 'internal');

  const getMessageIcon = (type: ExecutionMessage['type']) => {
    switch (type) {
      case 'thinking': return <Brain className="w-4 h-4 text-yellow-500" />;
      case 'tool_call': return <Terminal className="w-4 h-4 text-purple-500" />;
      case 'handoff': return <GitBranch className="w-4 h-4 text-blue-500" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-500" />;
      default: return <MessageSquare className="w-4 h-4 text-green-500" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg flex flex-col h-full">
      <div className="p-4 border-b flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Activity className="text-blue-500" />
          Workflow Execution
        </h3>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showInternalMessages}
              onChange={(e) => setShowInternalMessages(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-600">Show Internal</span>
          </label>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-100 rounded transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {filteredMessages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${
              message.visibility === 'internal' ? 'opacity-60' : ''
            }`}
          >
            <div className="flex-shrink-0 mt-1">
              {getMessageIcon(message.type)}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium text-sm">{message.agent}</span>
                <span className="text-xs text-gray-500">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </span>
                {message.visibility === 'internal' && (
                  <span className="text-xs px-2 py-0.5 bg-gray-100 rounded">Internal</span>
                )}
              </div>
              <div className="text-gray-700 whitespace-pre-wrap">{message.content}</div>
            </div>
          </div>
        ))}
        {currentAgent && isExecuting && (
          <div className="flex items-center gap-2 text-blue-600 animate-pulse">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">{currentAgent} is processing...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && executeWorkflow()}
            placeholder="Enter your message..."
            className="flex-1 p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isExecuting}
          />
          <button
            onClick={executeWorkflow}
            disabled={isExecuting || !input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isExecuting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

// ===== Part 4: Main ROWBOAT Component =====
export interface ROWBOATProps {
  workflowId?: string;
  mode?: 'builder' | 'editor' | 'execution' | 'full';
  onWorkflowCreated?: (workflow: WorkflowDefinition) => void;
  apiUrl?: string;
  apiKey?: string;
}

export const ROWBOAT: React.FC<ROWBOATProps> = ({ 
  workflowId,
  mode = 'full',
  onWorkflowCreated,
  apiUrl,
  apiKey
}) => {
  const [activeTab, setActiveTab] = useState<'builder' | 'editor' | 'execution'>('builder');
  const [currentWorkflowId, setCurrentWorkflowId] = useState(workflowId);
  const [workflows, setWorkflows] = useState<WorkflowDefinition[]>([]);
  const api = useMemo(() => new EnhancedROWBOATApiService(apiUrl, apiKey), [apiUrl, apiKey]);

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    try {
      const workflowList = await api.listWorkflows();
      setWorkflows(workflowList);
    } catch (error) {
      console.error('Failed to load workflows:', error);
    }
  };

  const handleWorkflowCreated = (workflow: WorkflowDefinition) => {
    const workflowId = (workflow as any).id || workflow.name;
    setCurrentWorkflowId(workflowId);
    setActiveTab('editor');
    loadWorkflows();
    onWorkflowCreated?.(workflow);
  };

  if (mode !== 'full') {
    // Single mode rendering
    switch (mode) {
      case 'builder':
        return <StreamingWorkflowBuilder onWorkflowCreated={handleWorkflowCreated} />;
      case 'editor':
        return <EnhancedWorkflowVisualEditor workflowId={currentWorkflowId} />;
      case 'execution':
        return currentWorkflowId ? (
          <WorkflowExecutionPanel workflowId={currentWorkflowId} />
        ) : (
          <div className="text-center p-8 text-gray-500">
            No workflow selected for execution
          </div>
        );
    }
  }

  // Full mode with tabs
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Network className="text-blue-500" />
            ROWBOAT Workflow System
          </h1>
          <p className="text-gray-600 mt-2">
            Build, visualize, and execute AI agent workflows
          </p>
        </div>

        {/* Workflow Selector */}
        {workflows.length > 0 && (
          <div className="mb-4 flex items-center gap-4">
            <label className="text-sm font-medium text-gray-700">Active Workflow:</label>
            <select
              value={currentWorkflowId || ''}
              onChange={(e) => setCurrentWorkflowId(e.target.value)}
              className="px-3 py-1.5 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select a workflow...</option>
              {workflows.map((wf) => (
                <option key={wf.id} value={wf.id}>
                  {wf.name} - {wf.description}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('builder')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'builder'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Natural Language Builder
              </div>
            </button>
            <button
              onClick={() => setActiveTab('editor')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'editor'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <GitBranch className="w-4 h-4" />
                Visual Editor
              </div>
            </button>
            <button
              onClick={() => setActiveTab('execution')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'execution'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              disabled={!currentWorkflowId}
            >
              <div className="flex items-center gap-2">
                <Play className="w-4 h-4" />
                Execution
              </div>
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === 'builder' && (
            <StreamingWorkflowBuilder onWorkflowCreated={handleWorkflowCreated} />
          )}
          {activeTab === 'editor' && (
            <EnhancedWorkflowVisualEditor workflowId={currentWorkflowId} />
          )}
          {activeTab === 'execution' && currentWorkflowId && (
            <WorkflowExecutionPanel workflowId={currentWorkflowId} />
          )}
        </div>
      </div>
    </div>
  );
};

// ===== Part 5: Exports and Provider =====

// Export all components for individual use
export {
  StreamingWorkflowBuilder,
  EnhancedWorkflowVisualEditor,
  WorkflowExecutionPanel,
  WorkflowTestPanel,
  MCPStatusIndicator,
  MCPToolSelector,
  EnhancedROWBOATApiService,
  EnhancedROWBOATWebSocket,
  ReactFlowErrorBoundary
};

// Export types
export type {
  WorkflowDefinition,
  AgentConfig,
  WorkflowEdge,
  WorkflowVisualization,
  AgentMetrics,
  WorkflowExecution,
  ExecutionMessage,
  MCPServer,
  StreamEvent,
  ROWBOATNodeData,
  ROWBOATEdgeData,
  ROWBOATNode,
  ROWBOATEdge
};

// Provider component for ReactFlow
export const ROWBOATProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ReactFlowProvider>
      {children}
    </ReactFlowProvider>
  );
};

// Default export with provider
export default function ROWBOATWithProvider(props: ROWBOATProps) {
  return (
    <ROWBOATProvider>
      <ROWBOAT {...props} />
    </ROWBOATProvider>
  );
}

// ===== Part 6: ROWBOAT Integration Components =====
// File: components/rowboat/rowboat-integration.tsx

// import React from 'react';
// import ROWBOATWithProvider, { 
//   ROWBOAT,
//   StreamingWorkflowBuilder,
//   EnhancedWorkflowVisualEditor,
//   WorkflowExecutionPanel,
//   WorkflowDefinition,
//   EnhancedROWBOATApiService
// } from './rowboat'; // The main file we just created


// Streaming Workflow Executor (wrapper around execution panel)
export const StreamingWorkflowExecutor: React.FC<{ workflowId?: string }> = ({ workflowId }) => {
  if (!workflowId) {
    return (
      <div className="p-8 text-center text-gray-500">
        Please select a workflow to execute
      </div>
    );
  }
  
  return <WorkflowExecutionPanel workflowId={workflowId} />;
};

// Enhanced Workflow List
export const EnhancedWorkflowList: React.FC = () => {
  const [workflows, setWorkflows] = React.useState<WorkflowDefinition[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [selectedWorkflow, setSelectedWorkflow] = React.useState<string | null>(null);
  const api = React.useMemo(() => new EnhancedROWBOATApiService(), []);

  React.useEffect(() => {
    loadWorkflows();
  }, []);

  

  const loadWorkflows = async () => {
    try {
      const list = await api.listWorkflows();
      setWorkflows(list);
    } catch (error) {
      console.error('Failed to load workflows:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }
  // const [workflows, setWorkflows] = useState<WorkflowDefinition[]>([]);

  useEffect(() => {
    const loadWorkflows = async () => {
      try {
        const api = new EnhancedROWBOATApiService();
        const workflowList = await api.listWorkflows();
        
        console.log('=== WORKFLOWS LIST OUTPUT ===');
        console.log('Total workflows:', workflowList.length);
        console.log('Full workflow list:', workflowList);
        
        workflowList.forEach((workflow, index) => {
          console.log(`Workflow ${index + 1}:`, {
            name: workflow.name,
            description: workflow.description,
            agentCount: workflow.agents?.length,
            edgeCount: workflow.edges?.length,
            metadata: workflow.metadata
          });
        });
        
        setWorkflows(workflowList);
      } catch (error) {
        console.error('Failed to load workflows:', error);
      }
    };
    
    loadWorkflows();
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4">Workflows</h2>
      
      {workflows.length === 0 ? (
        <p className="text-gray-500 text-center py-8">
          No workflows found. Create your first workflow!
        </p>
      ) : (
        <div className="space-y-3">
          {workflows.map((workflow) => (
            <div
              key={workflow.id || workflow.name}
              className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                selectedWorkflow === (workflow.id || workflow.name)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setSelectedWorkflow(workflow.id || workflow.name)}
            >
              <h3 className="font-semibold">{workflow.name}</h3>
              <p className="text-sm text-gray-600 mt-1">{workflow.description}</p>
              <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
                <span>ID: {workflow.id}</span> {/* Show the actual UUID */}
                <span>{workflow.agents.length} agents</span>
                <span>{workflow.edges.length} connections</span>
                {workflow.created_at && (
                  <span>Created {new Date(workflow.created_at).toLocaleDateString()}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};


// Enhanced ROWBOAT Dashboard
export const EnhancedROWBOATDashboard: React.FC = () => {
  return (
    <ROWBOATWithProvider mode="full" />
  );
};

// Re-export the other components with their expected names
// export { StreamingWorkflowBuilder, EnhancedWorkflowVisualEditor };

