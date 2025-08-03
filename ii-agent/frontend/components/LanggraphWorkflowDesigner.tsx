import React, { useState, useCallback, useEffect, useMemo, useRef, CSSProperties } from 'react';
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
  Database,
  Shield,
  Users,
  Workflow,
  GitBranch,
  Package,
  Server,
  Cloud,
  Lock,
  Unlock,
  TestTube,
  FlaskConical
} from 'lucide-react';

// ===== Enhanced TypeScript Interfaces =====
interface WorkflowDefinition {
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
}

interface WorkflowEdge {
  from_agent: string;
  to_agent: string;
  condition?: string;
  isMentionBased?: boolean;
}

interface AgentMetrics {
  totalTokens: number;
  totalDuration: number;
  toolCallCount: number;
  thoughtSteps: number;
  errorCount: number;
  successRate: number;
}

interface ExecutionMessage {
  id?: string;
  agent: string;
  content: string;
  timestamp: string;
  type: 'thinking' | 'response' | 'handoff' | 'tool_call' | 'tool_result' | 'error';
  metadata?: Record<string, any>;
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
  model: string;
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
}

interface ROWBOATEdgeData {
  condition?: string | null;
  isMentionBased?: boolean;
}

type ROWBOATNode = Node<ROWBOATNodeData>;
type ROWBOATEdge = Edge<ROWBOATEdgeData>;

// ===== Enhanced API Service with Streaming =====
class EnhancedRowboatAPI {
  private baseUrl: string;
  private apiKey?: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_ROWBOAT_API_URL || 
                   process.env.NEXT_PUBLIC_API_URL || 
                   'http://localhost:8000';
    this.apiKey = process.env.NEXT_PUBLIC_ROWBOAT_API_KEY;
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

  // Streaming workflow creation
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
        // Fallback to mock streaming
        yield* this.mockStreamingCreation(description);
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
          }
        }
      }
    } catch (error) {
      console.error('Stream creation error:', error);
      yield* this.mockStreamingCreation(description);
    }
  }

  // Mock streaming for demo/fallback
  private async *mockStreamingCreation(description: string): AsyncGenerator<StreamEvent> {
    yield { type: 'progress', data: { message: 'Analyzing requirements...', progress: 20 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'agent', data: { name: 'coordinator', role: 'coordinator' } };
    yield { type: 'progress', data: { message: 'Creating coordinator agent...', progress: 40 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'agent', data: { name: 'specialist', role: 'specialist' } };
    yield { type: 'progress', data: { message: 'Creating specialist agent...', progress: 60 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'tool', data: { name: 'data_processor', agent: 'specialist' } };
    yield { type: 'progress', data: { message: 'Adding tools...', progress: 80 } };
    await new Promise(resolve => setTimeout(resolve, 500));
    
    yield { type: 'edge', data: { from_agent: 'coordinator', to_agent: 'specialist' } };
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

  async buildFromDescription(description: string) {
    // Legacy non-streaming method
    const agents = [
      {
        name: 'intake_agent',
        role: 'coordinator',
        instructions: 'Receive customer request and route to @specialist_agent for processing.',
        tools: ['request_classifier'],
      },
      {
        name: 'specialist_agent',
        role: 'specialist',
        instructions: 'Process the request and send results to @review_agent for quality check.',
        tools: ['data_processor', 'api_caller'],
      },
      {
        name: 'review_agent',
        role: 'reviewer',
        instructions: 'Review the processed data and finalize the response.',
        tools: ['quality_checker'],
      },
    ];
    
    return { agents, edges: [] };
  }

  async suggestTools(agentRole: string, instructions: string) {
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

  async getMCPServers() {
    return {
      'research_server': {
        name: 'Research MCP Server',
        tools: ['deep_search', 'fact_check', 'summarize'],
        status: 'connected' as const
      },
      'analytics_server': {
        name: 'Analytics MCP Server',
        tools: ['analyze', 'visualize', 'predict'],
        status: 'disconnected' as const
      },
      'writing_server': {
        name: 'Writing MCP Server',
        tools: ['enhance', 'translate', 'format'],
        status: 'connected' as const
      },
    };
  }

  async testWorkflow(workflow: any, testInput: string) {
    return {
      success: true,
      trace: [
        { agent: 'intake_agent', action: 'Received input', output: 'Classified as: data request' },
        { agent: 'specialist_agent', action: 'Processing', output: 'Data processed successfully' },
        { agent: 'review_agent', action: 'Reviewing', output: 'Approved and finalized' },
      ],
    };
  }

  async getWorkflow(workflowId: string): Promise<WorkflowDefinition> {
    // Mock implementation
    return {
      name: workflowId,
      description: 'Workflow created via natural language',
      agents: [
        {
          name: 'coordinator',
          role: 'coordinator',
          instructions: 'Coordinate workflow execution',
          tools: ['request_classifier'],
        },
        {
          name: 'specialist',
          role: 'specialist',
          instructions: 'Process specialized tasks',
          tools: ['data_processor', 'api_caller'],
        }
      ],
      edges: [
        {
          from_agent: 'coordinator',
          to_agent: 'specialist',
          isMentionBased: true
        }
      ],
      entry_point: 'coordinator',
      metadata: { workflow_id: workflowId },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
  }

  async executeWorkflow(workflowId: string, input: string): Promise<any> {
    // Mock execution
    return {
      execution_id: `exec-${Date.now()}`,
      status: 'completed',
      result: 'Workflow executed successfully',
    };
  }

  async getMCPStatus() {
    return {
      mcp_enabled: true,
      connection_status: 'connected',
      available_tools: ['web_search', 'document_parser'],
    };
  }
}

// ===== Enhanced Agent Node Component =====
const EnhancedAgentNode: React.FC<NodeProps> = ({ data, isConnectable, selected }) => {
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

  // const bgColor = roleColors[data.role] || '#6b7280';
  // const status = data.status || 'idle';

  const statusIcons: Record<string, React.ReactNode> = {
    idle: null,
    thinking: <Brain className="animate-pulse" size={14} />,
    executing: <Zap className="animate-spin" size={14} />,
    error: <XCircle size={14} />,
    complete: <CheckCircle size={14} />,
  };

  const roleIcons: Record<string, React.ReactNode> = {
    coordinator: <Network size={16} />,
    researcher: <Globe size={16} />,
    analyzer: <BarChart3 size={16} />,
    writer: <FileText size={16} />,
    specialist: <Cpu size={16} />,
    reviewer: <Shield size={16} />,
    customer_support: <Users size={16} />,
    generic: <Package size={16} />,
  };
  const statusColors: Record<string, string> = {
    idle: 'transparent',
    thinking: '#f59e0b',
    executing: '#3b82f6',
    error: '#ef4444',
    complete: '#10b981',
  };

  const bgColor = roleColors[data.role] || '#6b7280';
  const Icon = roleIcons[data.role] || <Package size={16} />;
  const status = data.status || 'idle';
  
  return (
    <div
      style={{
        background: bgColor,
        color: 'white',
        padding: '12px 16px',
        borderRadius: '8px',
        minWidth: '200px',
        maxWidth: '250px',
        boxShadow: selected ? '0 0 0 2px #fff, 0 0 0 4px #3b82f6' : '0 2px 4px rgba(0,0,0,0.2)',
        border: `2px solid ${status !== 'idle' ? statusColors[status] : 'transparent'}`,
        transition: 'all 0.2s',
      }}
      className="agent-node hover:shadow-lg"
    >
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: '#555', width: 8, height: 8 }}
        isConnectable={isConnectable}
      />
      
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {Icon}
          <div style={{ fontWeight: 'bold', fontSize: '14px' }}>{data.label || data.name}</div>
        </div>
        {status !== 'idle' && statusIcons[status] && (
          <div style={{ color: statusColors[status] }}>
            {statusIcons[status]}
          </div>
        )}
      </div>
      
      <div style={{ fontSize: '12px', opacity: 0.9, marginBottom: '4px' }}>
        Role: {data.role}
      </div>
      
      {data.tools && data.tools.length > 0 && (
        <div style={{ fontSize: '11px', opacity: 0.8, marginTop: '4px' }}>
          🔧 {data.tools.length} tool{data.tools.length > 1 ? 's' : ''}
        </div>
      )}
      
      {data.outputVisibility === 'internal' && (
        <div style={{ fontSize: '10px', marginTop: '4px', opacity: 0.7, display: 'flex', alignItems: 'center', gap: '2px' }}>
          <Lock size={10} />
          Internal only
        </div>
      )}
      
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#555', width: 8, height: 8 }}
        isConnectable={isConnectable}
      />
    </div>
  );
};

// ===== WebSocket Manager for Real-time Updates =====
class EnhancedWebSocket {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(workflowId: string) {
    if (this.ws) {
      this.ws.close();
    }

    const wsUrl = process.env.NEXT_PUBLIC_ROWBOAT_API_URL || 
                  process.env.NEXT_PUBLIC_API_URL || 
                  'http://localhost:8000';
    const wsEndpoint = wsUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    
    try {
      this.ws = new WebSocket(`${wsEndpoint}/rowboat/ws/${workflowId}`);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected', { workflowId });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected', {});
        this.attemptReconnect(workflowId);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.emit('error', { message: 'Failed to connect to WebSocket' });
    }
  }

  private attemptReconnect(workflowId: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}...`);
      setTimeout(() => {
        this.connect(workflowId);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      this.emit('reconnect_failed', {});
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  private handleMessage(data: any) {
    this.emit(data.type, data);
  }

  on(event: string, callback: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: (data: any) => void) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  private emit(event: string, data: any) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
}

// ===== Main Enhanced LangGraph Workflow Designer =====
const EnhancedLangGraphWorkflowDesigner: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [naturalLanguageInput, setNaturalLanguageInput] = useState('');
  const [isBuilding, setIsBuilding] = useState(false);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [nodeIdCounter, setNodeIdCounter] = useState(1);
  const [mcpServers, setMcpServers] = useState<Record<string, any>>({});
  const [testResults, setTestResults] = useState<any>(null);
  const [showTestPanel, setShowTestPanel] = useState(false);
  
  // New enhanced features
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [buildProgress, setBuildProgress] = useState(0);
  const [createdWorkflow, setCreatedWorkflow] = useState<WorkflowDefinition | null>(null);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [messages, setMessages] = useState<ExecutionMessage[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [mcpStatus, setMcpStatus] = useState<'checking' | 'connected' | 'partial' | 'disconnected'>('checking');
  
  const api = useMemo(() => new EnhancedRowboatAPI(), []);
  const ws = useMemo(() => new EnhancedWebSocket(), []);

  const [isCanvasExpanded, setIsCanvasExpanded] = useState(false);
  const [showAgentConfig, setShowAgentConfig] = useState(false);
  const [testInput, setTestInput] = useState('');
  
  // Auto-detect @mentions and create edges
  const detectMentionsAndCreateEdges = useCallback(() => {
    const mentionRegex = /@(\w+)/g;
    const newEdges: ROWBOATEdge[] = [];
    
    nodes.forEach(node => {
      const matches = node.data.instructions.matchAll(mentionRegex);
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
            markerEnd: { type: MarkerType.ArrowClosed },
            data: { isMentionBased: true }
          } as ROWBOATEdge);
        }
      }
    });
    
    if (newEdges.length > 0) {
      setEdges((eds) => {
        const nonMentionEdges = eds.filter(e => !e.data?.isMentionBased);
        return [...nonMentionEdges, ...newEdges] as ROWBOATEdge[];
      });
    }
  }, [nodes, edges, setEdges]);

  // Load MCP servers on mount
  useEffect(() => {
    const loadServers = async () => {
      try {
        const servers = await api.getMCPServers();
        setMcpServers(servers);
        
        // Check MCP status
        const status = await api.getMCPStatus();
        if (status.connection_status === 'connected') {
          setMcpStatus('connected');
        } else if (status.mcp_enabled) {
          setMcpStatus('partial');
        } else {
          setMcpStatus('disconnected');
        }
      } catch (error) {
        console.error('Failed to load MCP servers:', error);
        setMcpStatus('disconnected');
      }
    };
    loadServers();
  }, [api]);

  // Auto-detect @mentions
  useEffect(() => {
    detectMentionsAndCreateEdges();
  }, [nodes, edges, detectMentionsAndCreateEdges]);

  // Enhanced streaming workflow creation
  const buildFromNaturalLanguage = async () => {
    if (!naturalLanguageInput.trim()) return;
    
    setIsBuilding(true);
    setStreamEvents([]);
    setBuildProgress(0);
    setCreatedWorkflow(null);
    setIsCanvasExpanded(true);
    
    try {
      const events: StreamEvent[] = [];
      const newNodes: ROWBOATNode[] = [];
      let nodeCounter = 1;
      
      for await (const event of api.createWorkflowFromDescriptionStream(naturalLanguageInput)) {
        events.push({ ...event, timestamp: Date.now() });
        setStreamEvents([...events]);
        
        if (event.type === 'progress' && event.data.progress) {
          setBuildProgress(event.data.progress);
        }
        
        if (event.type === 'agent') {
          const suggestedTools = await api.suggestTools(event.data.role, '');
          
          const newNode: ROWBOATNode = {
            id: `agent-${nodeCounter}`,
            type: 'rowboat',
            position: { 
              x: 100 + (nodeCounter - 1) * 300, 
              y: 100 + (nodeCounter % 2) * 150 
            },
            data: {
              name: event.data.name,
              role: event.data.role,
              model: 'gpt-4-turbo-preview',
              tools: event.data.tools || [],
              suggestedTools,
              instructions: event.data.instructions || '',
              mcpServers: [],
              status: 'idle',
            },
          };
          
          newNodes.push(newNode);
          setNodes(prev => [...prev, newNode]);
          nodeCounter++;
        }
        
        if (event.type === 'edge' && event.data.from_agent && event.data.to_agent) {
          const sourceNode = newNodes.find(n => n.data.name === event.data.from_agent);
          const targetNode = newNodes.find(n => n.data.name === event.data.to_agent);
          
          if (sourceNode && targetNode) {
            const newEdge: ROWBOATEdge = {
              id: `edge-${Date.now()}`,
              source: sourceNode.id,
              target: targetNode.id,
              type: 'smoothstep',
              animated: true,
              data: { isMentionBased: event.data.isMentionBased }
            } as ROWBOATEdge;
            
            setEdges(prev => [...prev, newEdge]);
          }
        }
        
        if (event.type === 'complete' && event.data.workflow_id) {
          const workflow = await api.getWorkflow(event.data.workflow_id);
          setCreatedWorkflow(workflow);
          setNodeIdCounter(nodeCounter);
        }
      }
      
      setNaturalLanguageInput('');
      console.log('✨ Workflow created from your description!');
    } catch (error) {
      console.error('Failed to build from description:', error);
      alert('Failed to build workflow from description. Please try again.');
    } finally {
      setIsBuilding(false);
    }
  };

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({
        ...params,
        type: 'smoothstep',
        animated: true,
        style: { stroke: '#666' },
        data: { condition: null, isMentionBased: false }
      }, eds));
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
    setShowAgentConfig(true); // Add this line
  }, []);

  const updateNodeData = useCallback((nodeId: string, updates: Partial<ROWBOATNodeData>) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...updates } }
          : node
      )
    );
  }, [setNodes]);

  const applySuggestedTools = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (node?.data.suggestedTools) {
      updateNodeData(nodeId, {
        tools: [...new Set([...node.data.tools, ...node.data.suggestedTools])],
      });
    }
  }, [nodes, updateNodeData]);

  const testWorkflow = async () => {
    if (!testInput.trim()) {
      setTestInput('Test customer billing inquiry');
    }
    
    setIsExecuting(true);
    setShowTestPanel(true);
    
    try {
      // Simulate execution on nodes
      for (const node of nodes) {
        updateNodeStatus(node.data.name, 'thinking');
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateNodeStatus(node.data.name, 'executing');
        await new Promise(resolve => setTimeout(resolve, 800));
        
        updateNodeStatus(node.data.name, 'complete');
      }
      
      const workflow = {
        nodes: nodes.map(n => n.data),
        edges: edges.map(e => ({
          source: nodes.find(n => n.id === e.source)?.data.name,
          target: nodes.find(n => n.id === e.target)?.data.name,
          isMentionBased: e.data?.isMentionBased,
        })),
      };
      
      const results = await api.testWorkflow(workflow, testInput || 'Test customer request');
      setTestResults(results);
      setShowTestPanel(true);
      
      // Reset node statuses after delay
      setTimeout(() => {
        setNodes(nds => nds.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
      }, 2000);
      
    } catch (error) {
      console.error('Failed to test workflow:', error);
      alert('Failed to test workflow. Please try again.');
      setNodes(nds => nds.map(n => ({ ...n, data: { ...n.data, status: 'error' } })));
    } finally {
      setIsExecuting(false);
    }
  };

  const addAgent = useCallback(() => {
    const newNode: ROWBOATNode = {
      id: `agent-${nodeIdCounter}`,
      type: 'rowboat',
      position: { x: 250, y: 100 + nodeIdCounter * 150 },
      data: {
        name: `agent_${nodeIdCounter}`,
        role: 'generic',
        model: 'gpt-4-turbo-preview',
        tools: [],
        instructions: 'Describe what this agent does. Use @agent_name to route to other agents.',
        mcpServers: [],
        status: 'idle',
      },
    };
    setNodes((nds) => [...nds, newNode]);
    setNodeIdCounter((prev) => prev + 1);
  }, [nodeIdCounter, setNodes]);

  // Execute workflow with real-time updates
  const executeWorkflow = async (input: string) => {
    if (!createdWorkflow || isExecuting) return;
    
    setIsExecuting(true);
    setMessages([]);
    
    try {
      // Connect WebSocket for real-time updates
      ws.connect(createdWorkflow.name);
      
      // Set up WebSocket listeners
      ws.on('agent_thinking', (data) => {
        setCurrentAgent(data.agent);
        addMessage({
          agent: data.agent,
          content: data.content || data.text,
          timestamp: new Date().toISOString(),
          type: 'thinking',
        });
        updateNodeStatus(data.agent, 'thinking');
      });

      ws.on('agent_response', (data) => {
        addMessage({
          agent: data.agent,
          content: data.content || data.text,
          timestamp: new Date().toISOString(),
          type: 'response',
        });
        updateNodeStatus(data.agent, 'complete');
      });

      ws.on('workflow_complete', () => {
        setIsExecuting(false);
        setCurrentAgent(null);
        // Reset all node statuses
        setNodes(nds => nds.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
      });

      // Send execution request
      ws.send({ message: input });
      
    } catch (error) {
      console.error('Failed to execute workflow:', error);
      setIsExecuting(false);
    }
  };

  const addMessage = (message: ExecutionMessage) => {
    setMessages(prev => [...prev, { ...message, id: `msg-${Date.now()}-${Math.random()}` }]);
  };

  const updateNodeStatus = (agentName: string, status: ROWBOATNodeData['status']) => {
    setNodes(nds => nds.map(n => 
      n.data.name === agentName 
        ? { ...n, data: { ...n.data, status } }
        : n
    ));
  };

  // Node types with enhanced agent node
  const nodeTypes = useMemo(() => ({
    rowboat: EnhancedAgentNode,
  }), []);

  // Enhanced styles
  const styles: Record<string, CSSProperties> = {
    container: {
      height: '100vh',
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'Inter, system-ui, sans-serif',
    },
    toolbar: {
      padding: '16px',
      backgroundColor: '#1a1a1a',
      borderBottom: '1px solid #333',
      display: 'flex',
      gap: '12px',
      alignItems: 'center',
      flexWrap: 'wrap',
    },
    naturalLanguageInput: {
      flex: 1,
      minWidth: '300px',
      padding: '10px 16px',
      backgroundColor: '#2a2a2a',
      border: '1px solid #444',
      borderRadius: '8px',
      color: '#fff',
      fontSize: '14px',
      outline: 'none',
    },
    button: {
      padding: '10px 20px',
      backgroundColor: '#0084ff',
      color: 'white',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all 0.2s',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    secondaryButton: {
      padding: '10px 20px',
      backgroundColor: '#333',
      color: '#fff',
      border: '1px solid #555',
      borderRadius: '8px',
      cursor: 'pointer',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all 0.2s',
    },
    successButton: {
      padding: '10px 20px',
      backgroundColor: '#00a67e',
      color: 'white',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all 0.2s',
    },
    panel: {
      backgroundColor: '#1a1a1a',
      color: '#fff',
      padding: '20px',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
      maxWidth: '400px',
      border: '1px solid #333',
    },
    streamPanel: {
      position: 'absolute' as const,
      bottom: '20px',
      left: '20px',
      backgroundColor: '#1a1a1a',
      color: '#fff',
      padding: '16px',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
      maxWidth: '400px',
      maxHeight: '300px',
      overflow: 'auto',
      border: '1px solid #333',
    },
    progressBar: {
      width: '100%',
      height: '4px',
      backgroundColor: '#333',
      borderRadius: '2px',
      overflow: 'hidden',
      marginTop: '8px',
    },
    progressFill: {
      height: '100%',
      backgroundColor: '#0084ff',
      transition: 'width 0.3s ease',
    },
    mcpBadge: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '4px',
      padding: '4px 8px',
      fontSize: '12px',
      borderRadius: '4px',
      backgroundColor: '#2a2a2a',
      border: '1px solid #444',
    },
  };

  const mcpStatusConfig = {
    checking: { color: '#666', text: 'Checking...', icon: <Loader2 className="animate-spin" size={12} /> },
    connected: { color: '#10b981', text: 'Connected', icon: <CheckCircle size={12} /> },
    partial: { color: '#f59e0b', text: 'Partial', icon: <AlertCircle size={12} /> },
    disconnected: { color: '#ef4444', text: 'Disconnected', icon: <XCircle size={12} /> }
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.toolbar}>
        <div className="flex items-center gap-3">
          <Workflow className="w-6 h-6 text-blue-500" />
          <h1 className="text-xl font-semibold">ROWBOAT Multi-Agent System</h1>
        </div>
        
        {/* MCP Status Badge */}
        <div style={styles.mcpBadge}>
          <span style={{ color: mcpStatusConfig[mcpStatus].color }}>
            {mcpStatusConfig[mcpStatus].icon}
          </span>
          <span style={{ color: mcpStatusConfig[mcpStatus].color }}>
            MCP: {mcpStatusConfig[mcpStatus].text}
          </span>
        </div>
      </div>

      {/* Natural Language Input */}
      <div style={{ padding: '16px', borderBottom: '1px solid #333', backgroundColor: '#0a0a0a' }}>
        <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles className="w-5 h-5 text-yellow-500" />
          Natural Language Workflow Builder
        </h2>
        <p style={{ fontSize: '14px', color: '#888', marginBottom: '12px' }}>
          Describe your workflow in plain English
        </p>
        <div style={{ display: 'flex', gap: '8px' }}>
          <textarea
            style={{
              ...styles.naturalLanguageInput,
              minHeight: '100px',
              resize: 'none'
            }}
            placeholder="Example: Create a customer support workflow that classifies incoming requests and routes them to billing or technical support agents based on the content..."
            value={naturalLanguageInput}
            onChange={(e) => setNaturalLanguageInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && e.ctrlKey) {
                buildFromNaturalLanguage();
              }
            }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
          <div style={{ fontSize: '12px', color: '#666' }}>
            Press Ctrl+Enter to build
          </div>
          <button 
            style={styles.button}
            onClick={buildFromNaturalLanguage}
            disabled={isBuilding || !naturalLanguageInput.trim()}
          >
            {isBuilding ? (
              <>
                <Loader2 className="animate-spin" size={16} />
                Building...
              </>
            ) : (
              <>
                <Sparkles size={16} />
                Build with AI
              </>
            )}
          </button>
        </div>
      </div>

      {/* Collapsible Canvas Area */}
      <div style={{ flex: 1, backgroundColor: '#0a0a0a', position: 'relative' }}>
        {/* Canvas Header/Toggle */}
        <button
          style={{
            width: '100%',
            padding: '12px 16px',
            backgroundColor: isCanvasExpanded ? '#1a1a1a' : '#111',
            border: 'none',
            borderBottom: '1px solid #333',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            color: '#fff',
            transition: 'all 0.2s',
          }}
          onClick={() => setIsCanvasExpanded(!isCanvasExpanded)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {isCanvasExpanded ? (
                <ChevronDown size={16} />
              ) : (
                <ChevronRight size={16} />
              )}
              <GitBranch size={16} />
            </div>
            <span style={{ fontWeight: '500' }}>Workflow Canvas</span>
            {nodes.length > 0 && (
              <span style={{ fontSize: '14px', color: '#888' }}>
                ({nodes.length} agents, {edges.length} connections)
              </span>
            )}
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {!isCanvasExpanded && nodes.length > 0 && (
              <div style={{ display: 'flex', marginRight: '8px' }}>
                {nodes.slice(0, 3).map((node, i) => (
                  <div
                    key={node.id}
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      backgroundColor: '#3b82f6',
                      border: '2px solid #0a0a0a',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginLeft: i > 0 ? '-8px' : '0',
                      zIndex: nodes.length - i,
                    }}
                  >
                    <span style={{ fontSize: '12px', fontWeight: 'bold' }}>
                      {node.data.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                ))}
                {nodes.length > 3 && (
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      backgroundColor: '#333',
                      border: '2px solid #0a0a0a',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginLeft: '-8px',
                    }}
                  >
                    <span style={{ fontSize: '12px' }}>+{nodes.length - 3}</span>
                  </div>
                )}
              </div>
            )}
            <span style={{ fontSize: '12px', color: '#666' }}>
              {isCanvasExpanded ? 'Click to collapse' : 'Click to expand'}
            </span>
          </div>
        </button>

        {/* Canvas Content */}
        {isCanvasExpanded && (
          <div style={{ position: 'absolute', inset: 0, top: '49px', backgroundColor: '#0a0a0a' }}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              fitView
              nodeTypes={nodeTypes}
              defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
            >
              <MiniMap 
                style={{ backgroundColor: '#1a1a1a' }}
                nodeColor="#0084ff"
              />
              <Controls className="react-flow__controls" />
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#333" />
              
              {/* Toolbar Panel */}
              <Panel position="top-left" style={styles.panel}>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    style={styles.secondaryButton}
                    onClick={addAgent}
                  >
                    <Plus size={16} className="inline mr-1" />
                    Add Agent
                  </button>
                  <button
                    style={styles.successButton}
                    onClick={testWorkflow}
                    disabled={nodes.length === 0 || isExecuting}
                  >
                    {isExecuting ? (
                      <Loader2 className="animate-spin inline mr-1" size={16} />
                    ) : (
                      <FlaskConical size={16} className="inline mr-1" />
                    )}
                    Test Workflow
                  </button>
                  <button
                    style={styles.secondaryButton}
                  >
                    <Save size={16} className="inline mr-1" />
                    Save
                  </button>
                </div>
              </Panel>
              
              {/* Test Results Panel */}
              {showTestPanel && testResults && (
                <Panel position="bottom-left" style={{ ...styles.panel, maxWidth: '400px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <h4 style={{ marginTop: 0, marginBottom: 0 }}>
                      Test Results {testResults.success ? '✅' : '❌'}
                    </h4>
                    <button
                      style={{ ...styles.secondaryButton, padding: '4px 8px', fontSize: '12px' }}
                      onClick={() => setShowTestPanel(false)}
                    >
                      ✕
                    </button>
                  </div>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ fontSize: '12px', color: '#888' }}>Test Input</label>
                    <input
                      type="text"
                      style={{
                        width: '100%',
                        padding: '8px',
                        marginTop: '4px',
                        backgroundColor: '#2a2a2a',
                        border: '1px solid #444',
                        borderRadius: '4px',
                        color: '#fff',
                      }}
                      placeholder="Enter test message..."
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                    />
                  </div>
                  
                  <div style={{ fontSize: '13px' }}>
                    {testResults.trace.map((step: any, i: number) => (
                      <div key={i} style={{ marginBottom: '8px', padding: '8px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
                        <strong style={{ color: '#0084ff' }}>{step.agent}:</strong> {step.action}
                        <div style={{ color: '#888', fontSize: '12px', marginTop: '4px' }}>{step.output}</div>
                      </div>
                    ))}
                  </div>
                </Panel>
              )}

              {/* Agent Configuration Panel */}
              {showAgentConfig && selectedNode && (
                <Panel position="top-right" style={{ ...styles.panel, width: '320px', marginTop: '60px' }}>
                  <h3 style={{ marginTop: 0, marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    Configure Agent
                    <button
                      style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: '4px' }}
                      onClick={() => setShowAgentConfig(false)}
                    >
                      <X size={16} />
                    </button>
                  </h3>
                  {(() => {
                    const node = nodes.find(n => n.id === selectedNode);
                    if (!node) return null;
                    
                    return (
                      <>
                        <div style={{ marginBottom: '12px' }}>
                          <label style={{ fontSize: '12px', color: '#888' }}>Agent Name</label>
                          <input
                            style={{ ...styles.naturalLanguageInput, marginTop: '4px' }}
                            type="text"
                            value={node.data.name}
                            onChange={(e) => updateNodeData(selectedNode, { name: e.target.value })}
                          />
                        </div>
                        
                        <div style={{ marginBottom: '12px' }}>
                          <label style={{ fontSize: '12px', color: '#888' }}>Role</label>
                          <select
                            style={{ ...styles.naturalLanguageInput, marginTop: '4px' }}
                            value={node.data.role}
                            onChange={(e) => updateNodeData(selectedNode, { role: e.target.value })}
                          >
                            <option value="generic">Generic</option>
                            <option value="coordinator">Coordinator</option>
                            <option value="researcher">Researcher</option>
                            <option value="analyzer">Analyzer</option>
                            <option value="writer">Writer</option>
                            <option value="specialist">Specialist</option>
                            <option value="reviewer">Reviewer</option>
                            <option value="customer_support">Customer Support</option>
                          </select>
                        </div>
                        
                        <div style={{ marginBottom: '12px' }}>
                          <label style={{ fontSize: '12px', color: '#888' }}>Instructions</label>
                          <textarea
                            style={{ ...styles.naturalLanguageInput, minHeight: '100px', marginTop: '4px' }}
                            placeholder="Describe what this agent does..."
                            value={node.data.instructions}
                            onChange={(e) => updateNodeData(selectedNode, { instructions: e.target.value })}
                          />
                        </div>
                        
                        {/* MCP Servers */}
                        <div style={{ marginBottom: '12px' }}>
                          <h4 style={{ fontSize: '14px', marginBottom: '8px' }}>MCP Servers</h4>
                          {Object.entries(mcpServers).map(([key, server]) => (
                            <label key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', fontSize: '12px' }}>
                              <input
                                type="checkbox"
                                checked={node.data.mcpServers?.includes(key) || false}
                                onChange={(e) => {
                                  const servers = node.data.mcpServers || [];
                                  updateNodeData(selectedNode, {
                                    mcpServers: e.target.checked 
                                      ? [...servers, key]
                                      : servers.filter((s: string) => s !== key)
                                  });
                                }}
                                style={{ marginRight: '8px' }}
                              />
                              {server.name} ({server.tools.length} tools)
                              <span style={{ 
                                marginLeft: '4px',
                                width: '8px',
                                height: '8px',
                                borderRadius: '50%',
                                backgroundColor: server.status === 'connected' ? '#10b981' : '#ef4444',
                                display: 'inline-block'
                              }} />
                            </label>
                          ))}
                        </div>
                        
                        {/* Tools */}
                        {node.data.tools && node.data.tools.length > 0 && (
                          <div>
                            <h4 style={{ fontSize: '14px', marginBottom: '8px' }}>Tools</h4>
                            {node.data.tools.map((tool: string, i: number) => (
                              <span key={i} style={{ 
                                fontSize: '12px',
                                marginRight: '8px',
                                padding: '4px 8px',
                                backgroundColor: '#374151',
                                borderRadius: '4px',
                                display: 'inline-block',
                                marginBottom: '4px'
                              }}>
                                {tool}
                              </span>
                            ))}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </Panel>
              )}
            </ReactFlow>
          </div>
        )}

        {/* Streaming Progress Panel */}
        {isBuilding && streamEvents.length > 0 && (
          <div style={styles.streamPanel}>
            <h4 style={{ marginTop: 0, marginBottom: '12px', fontSize: '16px' }}>
              Building Workflow...
            </h4>
            <div style={styles.progressBar}>
              <div style={{ ...styles.progressFill, width: `${buildProgress}%` }} />
            </div>
            <div style={{ marginTop: '12px', fontSize: '12px' }}>
              {streamEvents.slice(-5).map((event, idx) => (
                <div key={idx} style={{ marginBottom: '4px', opacity: 0.8 }}>
                  <ChevronRight size={12} className="inline mr-1" />
                  {event.type === 'agent' && (
                    <span style={{ color: '#3b82f6' }}>
                      Created agent: <strong>{event.data.name}</strong>
                    </span>
                  )}
                  {event.type === 'tool' && (
                    <span style={{ color: '#10b981' }}>
                      Added tool: <strong>{event.data.name}</strong>
                    </span>
                  )}
                  {event.type === 'edge' && (
                    <span style={{ color: '#8b5cf6' }}>
                      Connected agents
                    </span>
                  )}
                  {event.type === 'progress' && (
                    <span style={{ color: '#6b7280' }}>{event.data.message}</span>
                  )}
                  {event.type === 'complete' && (
                    <span style={{ color: '#10b981', fontWeight: 'bold' }}>
                      ✓ Workflow created!
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* MCP Server Status Panel (move outside ReactFlow) */}
        <div style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          width: '250px',
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '8px',
          padding: '16px',
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        }}>
          <h3 style={{ marginTop: 0, marginBottom: '12px', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Server size={16} />
            MCP Server Status
          </h3>
          <div style={{ fontSize: '12px' }}>
            {Object.entries(mcpServers).map(([key, server]) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: server.status === 'connected' ? '#10b981' : 
                                    server.status === 'disconnected' ? '#ef4444' : '#f59e0b'
                  }} />
                  <span>{server.name}</span>
                </div>
                <span style={{ color: '#888' }}>{server.tools.length} tools</span>
              </div>
            ))}
          </div>
          {mcpStatus === 'partial' && (
            <p style={{ fontSize: '11px', color: '#888', marginTop: '8px', marginBottom: 0 }}>
              Status: Some servers unavailable
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
// Export wrapped in provider
const EnhancedLangGraphWorkflowDesignerWithProvider: React.FC = () => (
  <ReactFlowProvider>
    <EnhancedLangGraphWorkflowDesigner />
  </ReactFlowProvider>
);

export default EnhancedLangGraphWorkflowDesignerWithProvider;

