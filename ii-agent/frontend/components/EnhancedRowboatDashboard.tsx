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
    };
    position: { x: number; y: number };
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

// Test Result interfaces
interface TestTrace {
  agent: string;
  action?: string;
  output: string;
}

interface TestResult {
  success: boolean;
  execution_time?: number;
  trace?: TestTrace[];
  error?: string;
}

interface EnhancementResult {
  suggestions?: string[];
  tool_recommendations?: string[];
}

// ===== Enhanced Styles (Dark Theme from LangGraph) =====
const styles = {
  container: {
    height: '100vh',
    width: '100%',
    display: 'flex',
    flexDirection: 'column' as const,
    fontFamily: 'Inter, system-ui, sans-serif',
    backgroundColor: '#0a0a0a',
    color: '#fff',
  },
  header: {
    padding: '16px 24px',
    backgroundColor: '#1a1a1a',
    borderBottom: '1px solid #333',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  toolbar: {
    padding: '16px',
    backgroundColor: '#1a1a1a',
    borderBottom: '1px solid #333',
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
    flexWrap: 'wrap' as const,
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
    border: '1px solid #333',
  },
  canvasContainer: {
    position: 'relative' as const,
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  collapsibleHeader: {
    position: 'relative' as const,
    width: '100%',
    padding: '12px 16px',
    backgroundColor: '#1a1a1a',
    border: 'none',
    borderBottom: '1px solid #333',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    color: '#fff',
    transition: 'all 0.2s',
    '&:hover': {
      backgroundColor: '#222',
    },
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
};

// ===== Service Classes =====
class EnhancedROWBOATApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  async createWorkflowFromDescription(description: string): Promise<WorkflowDefinition> {
    // Mock implementation
    return {
      name: 'Generated Workflow',
      description,
      agents: [],
      edges: [],
      entry_point: 'coordinator',
      metadata: {},
    };
  }

  async *createWorkflowFromDescriptionStream(description: string): AsyncGenerator<StreamEvent> {
    // Mock streaming implementation
    const events: StreamEvent[] = [
      { type: 'progress', data: { progress: 10, message: 'Analyzing description...' } },
      { type: 'agent', data: { name: 'coordinator', role: 'coordinator', instructions: 'Coordinate the workflow' } },
      { type: 'progress', data: { progress: 30, message: 'Creating agents...' } },
      { type: 'agent', data: { name: 'processor', role: 'specialist', instructions: 'Process data' } },
      { type: 'edge', data: { from_agent: 'coordinator', to_agent: 'processor', isMentionBased: true } },
      { type: 'progress', data: { progress: 100, message: 'Complete!' } },
      { type: 'complete', data: { workflow_id: 'test-workflow', summary: { agents: 2, edges: 1 } } },
    ];

    for (const event of events) {
      await new Promise(resolve => setTimeout(resolve, 500));
      yield event;
    }
  }

  async getVisualization(workflowId: string): Promise<WorkflowVisualization> {
    // Mock implementation
    return {
      nodes: [
        {
          id: '1',
          type: 'agent',
          data: {
            label: 'Coordinator',
            role: 'coordinator',
            hasInstructions: true,
            connectedAgents: ['processor'],
            tools: ['request_classifier'],
            model: 'gpt-4',
          },
          position: { x: 100, y: 100 },
        },
        {
          id: '2',
          type: 'agent',
          data: {
            label: 'Processor',
            role: 'specialist',
            hasInstructions: true,
            connectedAgents: [],
            tools: ['data_processor'],
            model: 'gpt-4',
          },
          position: { x: 400, y: 100 },
        },
      ],
      edges: [
        {
          id: 'e1-2',
          source: '1',
          target: '2',
          type: 'smoothstep',
          animated: true,
        },
      ],
      metadata: {},
    };
  }

  async getMCPServers(): Promise<MCPServer[]> {
    // Mock implementation
    return [
      {
        name: 'filesystem',
        url: 'mcp://localhost:3000',
        tools: ['read_file', 'write_file', 'list_directory'],
        status: 'connected',
      },
      {
        name: 'web_search',
        url: 'mcp://localhost:3001',
        tools: ['search', 'fetch_page'],
        status: 'available',
      },
    ];
  }

  async enhanceWorkflow(workflowId: string): Promise<EnhancementResult> {
    // Mock implementation
    return {
      suggestions: [
        'Consider adding error handling agents',
        'Add a review agent for quality control',
        'Implement parallel processing for better performance',
      ],
      tool_recommendations: ['data_validator', 'error_handler', 'notification_sender'],
    };
  }

  async testWorkflow(workflowId: string, input: string): Promise<TestResult> {
    // Mock implementation
    return {
      success: true,
      execution_time: 2.5,
      trace: [
        { agent: 'coordinator', action: 'Received input', output: 'Classified as: data processing request' },
        { agent: 'processor', action: 'Processing', output: 'Data processed successfully' },
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
          name: 'processor',
          role: 'specialist',
          instructions: 'Process specialized tasks',
          tools: ['data_processor', 'api_caller'],
        }
      ],
      edges: [
        {
          from_agent: 'coordinator',
          to_agent: 'processor',
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

class EnhancedROWBOATWebSocket {
  private listeners: Map<string, Set<Function>> = new Map();

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)?.add(callback);
  }

  off(event: string, callback: Function) {
    this.listeners.get(event)?.delete(callback);
  }

  emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }

  connect() {
    // Mock connection
    console.log('WebSocket connected');
  }

  disconnect() {
    // Mock disconnection
    console.log('WebSocket disconnected');
  }
}

// ===== Enhanced Agent Node Component with Avatars =====
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

  return (
    <div
      style={{
        background: bgColor,
        color: 'white',
        padding: '12px 16px',
        borderRadius: '8px',
        minWidth: '200px',
        boxShadow: selected ? 
          `0 0 0 2px #fff, 0 0 0 4px ${bgColor}` : 
          '0 2px 8px rgba(0,0,0,0.2)',
        border: status !== 'idle' ? `2px solid ${statusColors[status]}` : 'none',
        position: 'relative',
      }}
    >
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: '#555' }}
        isConnectable={isConnectable}
      />
      
      {/* Avatar */}
      <div
        style={{
          position: 'absolute',
          top: '-16px',
          right: '-16px',
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          backgroundColor: '#0a0a0a',
          border: `2px solid ${bgColor}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          fontSize: '14px',
        }}
      >
        {data.name.charAt(0).toUpperCase()}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <span style={{ fontWeight: 'bold' }}>{data.name}</span>
        {statusIcons[status]}
      </div>
      
      <div style={{ fontSize: '12px', opacity: 0.8, marginBottom: '8px' }}>
        {data.role} • {data.model || 'gpt-4'}
      </div>
      
      {data.tools && data.tools.length > 0 && (
        <div style={{ 
          display: 'flex', 
          gap: '4px', 
          flexWrap: 'wrap',
          marginTop: '8px'
        }}>
          {data.tools.slice(0, 3).map((tool: string, idx: number) => (
            <span
              key={idx}
              style={{
                fontSize: '10px',
                padding: '2px 6px',
                backgroundColor: 'rgba(255,255,255,0.2)',
                borderRadius: '4px',
              }}
            >
              {tool}
            </span>
          ))}
          {data.tools.length > 3 && (
            <span style={{ fontSize: '10px', opacity: 0.6 }}>
              +{data.tools.length - 3}
            </span>
          )}
        </div>
      )}
      
      {data.metrics && (
        <div style={{ 
          marginTop: '8px', 
          fontSize: '10px', 
          opacity: 0.7,
          borderTop: '1px solid rgba(255,255,255,0.2)',
          paddingTop: '8px'
        }}>
          <div>Tokens: {data.metrics.totalTokens}</div>
          <div>Success: {data.metrics.successRate}%</div>
        </div>
      )}
      
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#555' }}
        isConnectable={isConnectable}
      />
    </div>
  );
};

// ===== MCP Status Indicator =====
const MCPStatusIndicator: React.FC = () => {
  const [mcpStatus, setMcpStatus] = useState<'checking' | 'connected' | 'partial' | 'disconnected'>('checking');
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  useEffect(() => {
    const checkMCPStatus = async () => {
      try {
        const status = await api.getMCPStatus();
        if (status.mcp_enabled && status.connection_status === 'connected') {
          setMcpStatus('connected');
        } else if (status.mcp_enabled) {
          setMcpStatus('partial');
        } else {
          setMcpStatus('disconnected');
        }
      } catch (error) {
        setMcpStatus('disconnected');
      }
    };

    checkMCPStatus();
    const interval = setInterval(checkMCPStatus, 30000);
    
    return () => clearInterval(interval);
  }, [api]);

  const statusConfig = {
    checking: { color: 'gray', text: 'Checking MCP...', icon: <Loader2 className="animate-spin" size={14} /> },
    connected: { color: 'green', text: 'MCP Connected', icon: <CheckCircle size={14} /> },
    partial: { color: 'yellow', text: 'MCP Partial', icon: <AlertCircle size={14} /> },
    disconnected: { color: 'red', text: 'MCP Disconnected', icon: <XCircle size={14} /> }
  };

  const config = statusConfig[mcpStatus];

  return (
    <div className={`flex items-center gap-2 text-sm`}>
      <span style={{ color: `${config.color}` }}>{config.icon}</span>
      <span style={{ color: `${config.color}` }}>{config.text}</span>
    </div>
  );
};

// ===== Enhanced Visual Workflow Editor with Collapsible Canvas =====
const EnhancedWorkflowVisualEditor: React.FC<{ workflowId?: string }> = ({ workflowId }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const [showEnhancements, setShowEnhancements] = useState(false);
  const [enhancements, setEnhancements] = useState<EnhancementResult | null>(null);
  const [isCanvasExpanded, setIsCanvasExpanded] = useState(true);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [testResults, setTestResults] = useState<TestResult | null>(null);
  const [testInput, setTestInput] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  // Memoize node types
  const nodeTypes = useMemo(() => ({
    agent: EnhancedAgentNode,
  }), []);

  useEffect(() => {
    if (workflowId) {
      loadWorkflowVisualization();
    }
    loadMCPServers();
  }, [workflowId]);

  const loadWorkflowVisualization = async () => {
    if (!workflowId) return;
    
    try {
      const viz = await api.getVisualization(workflowId);
      setNodes((viz.nodes || []).map(n => ({ ...n, type: 'agent' })));
      setEdges(viz.edges || []);
    } catch (error) {
      console.error('Failed to load visualization:', error);
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
              width: 20,
              height: 20,
              color: '#666',
            },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
  }, []);

  const handleEnhance = async () => {
    if (!workflowId) return;
    
    try {
      const result = await api.enhanceWorkflow(workflowId);
      setEnhancements(result);
      setShowEnhancements(true);
    } catch (error) {
      console.error('Failed to enhance workflow:', error);
    }
  };

  const testWorkflow = async () => {
    if (!workflowId || !testInput.trim()) return;
    
    setIsExecuting(true);
    setShowTestPanel(true);
    
    try {
      const results = await api.testWorkflow(workflowId, testInput);
      setTestResults(results);
    } catch (error) {
      console.error('Test failed:', error);
      setTestResults({ success: false, error: error instanceof Error ? error.message : 'Unknown error' });
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div style={styles.canvasContainer}>
      {/* Collapsible Canvas Header */}
      <button
        style={{
          ...styles.collapsibleHeader,
          backgroundColor: isCanvasExpanded ? '#1a1a1a' : '#111',
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
            
            {/* Top Panel with Actions */}
            <Panel position="top-left" style={styles.panel}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  style={styles.secondaryButton}
                  onClick={handleEnhance}
                >
                  <Sparkles size={16} className="inline mr-1" />
                  AI Enhance
                </button>
                <button
                  style={styles.successButton}
                  onClick={() => setShowTestPanel(!showTestPanel)}
                  disabled={nodes.length === 0}
                >
                  <FlaskConical size={16} className="inline mr-1" />
                  Test
                </button>
              </div>
            </Panel>

            {/* Test Panel */}
            {showTestPanel && (
              <Panel position="bottom-left" style={{ ...styles.panel, maxWidth: '400px' }}>
                <div style={{ marginBottom: '12px' }}>
                  <h4 style={{ marginTop: 0, marginBottom: '8px' }}>Test Workflow</h4>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                      type="text"
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                      placeholder="Enter test input..."
                      style={{
                        flex: 1,
                        padding: '8px',
                        backgroundColor: '#2a2a2a',
                        border: '1px solid #444',
                        borderRadius: '4px',
                        color: '#fff',
                        fontSize: '14px',
                      }}
                      onKeyPress={(e) => e.key === 'Enter' && testWorkflow()}
                    />
                    <button
                      style={styles.button}
                      onClick={testWorkflow}
                      disabled={isExecuting || !testInput.trim()}
                    >
                      {isExecuting ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
                    </button>
                  </div>
                </div>

                {testResults && (
                  <div style={{
                    marginTop: '12px',
                    padding: '12px',
                    backgroundColor: testResults.success ? '#064e3b' : '#7f1d1d',
                    borderRadius: '6px',
                    fontSize: '14px',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      {testResults.success ? <CheckCircle size={16} /> : <XCircle size={16} />}
                      <span style={{ fontWeight: 'bold' }}>
                        {testResults.success ? 'Test Passed' : 'Test Failed'}
                      </span>
                    </div>
                    
                    {testResults.execution_time && (
                      <div style={{ fontSize: '12px', opacity: 0.8 }}>
                        Execution time: {testResults.execution_time}s
                      </div>
                    )}
                    
                    {testResults.trace && (
                      <div style={{ marginTop: '8px' }}>
                        <div style={{ fontSize: '12px', opacity: 0.8, marginBottom: '4px' }}>Trace:</div>
                        {testResults.trace.map((step: TestTrace, idx: number) => (
                          <div key={idx} style={{ fontSize: '12px', marginLeft: '12px', marginTop: '4px' }}>
                            <span style={{ color: '#3b82f6' }}>{step.agent}:</span> {step.output}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </Panel>
            )}

            {/* MCP Server Selection Panel */}
            {selectedNode && (
              <Panel position="top-right" style={styles.panel}>
                <h4 style={{ marginTop: 0 }}>MCP Servers</h4>
                <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '12px' }}>
                  Connect servers to add tools
                </div>
                <div className="space-y-2">
                  {mcpServers.map((server) => (
                    <label
                      key={server.name}
                      style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '8px',
                        cursor: 'pointer' 
                      }}
                    >
                      <input
                        type="checkbox"
                        style={{ cursor: 'pointer' }}
                      />
                      <span>{server.name}</span>
                      <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: server.status === 'connected' ? '#10b981' : 
                          server.status === 'error' ? '#ef4444' : '#6b7280'
                      }} />
                    </label>
                  ))}
                </div>
              </Panel>
            )}
          </ReactFlow>

          {/* Enhancement Suggestions Modal */}
          {showEnhancements && enhancements && (
            <div style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(0,0,0,0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '16px',
            }}>
              <div style={{
                backgroundColor: '#1a1a1a',
                borderRadius: '12px',
                padding: '24px',
                maxWidth: '600px',
                maxHeight: '80vh',
                overflow: 'auto',
                border: '1px solid #333',
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  marginBottom: '16px'
                }}>
                  <h3 style={{ margin: 0 }}>AI Enhancement Suggestions</h3>
                  <button
                    onClick={() => setShowEnhancements(false)}
                    style={{ 
                      background: 'none',
                      border: 'none',
                      color: '#999',
                      cursor: 'pointer'
                    }}
                  >
                    <X size={20} />
                  </button>
                </div>
                
                <div style={{ marginBottom: '16px' }}>
                  {enhancements.suggestions?.map((suggestion: any, idx: number) => (
                    <div key={idx} style={{
                      padding: '12px',
                      backgroundColor: '#2a2a2a',
                      borderRadius: '8px',
                      marginBottom: '8px',
                      fontSize: '14px',
                    }}>
                      {suggestion}
                    </div>
                  ))}
                </div>
                
                {enhancements.tool_recommendations && enhancements.tool_recommendations.length > 0 && (
                  <div>
                    <h4 style={{ marginBottom: '8px' }}>Recommended Tools:</h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {enhancements.tool_recommendations.map((tool: string, idx: number) => (
                        <span key={idx} style={{
                          padding: '6px 12px',
                          backgroundColor: '#064e3b',
                          color: '#10b981',
                          borderRadius: '16px',
                          fontSize: '12px',
                        }}>
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ===== Streaming Workflow Builder =====
const StreamingWorkflowBuilder: React.FC = () => {
  const [description, setDescription] = useState('');
  const [isBuilding, setIsBuilding] = useState(false);
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [createdWorkflow, setCreatedWorkflow] = useState<WorkflowDefinition | null>(null);
  const api = useMemo(() => new EnhancedROWBOATApiService(), []);

  const handleCreateWithStreaming = async () => {
    if (!description.trim()) return;
    
    setIsBuilding(true);
    setStreamEvents([]);
    setProgress(0);
    
    try {
      const events: StreamEvent[] = [];
      
      for await (const event of api.createWorkflowFromDescriptionStream(description)) {
        events.push({ ...event, timestamp: Date.now() });
        setStreamEvents([...events]);
        
        if (event.type === 'progress' && event.data.progress) {
          setProgress(event.data.progress);
        }
        
        if (event.type === 'complete' && event.data.workflow_id) {
          try {
            const workflow = await api.getWorkflow(event.data.workflow_id);
            setCreatedWorkflow(workflow);
          } catch (error) {
            console.error('Failed to get workflow details:', error);
            setCreatedWorkflow({
              name: event.data.workflow_id,
              description: description,
              agents: event.data.summary?.agents ? 
                Array(event.data.summary.agents).fill(null).map((_, i) => ({
                  name: `agent_${i}`,
                  role: 'generic',
                  instructions: 'Generated agent',
                })) : [],
              edges: event.data.summary?.edges ? 
                Array(event.data.summary.edges).fill(null).map((_, i) => ({
                  from_agent: `agent_${i}`,
                  to_agent: `agent_${i + 1}`,
                })) : [],
              entry_point: 'agent_0',
              metadata: {},
            });
          }
        }
      }
    } catch (error) {
      console.error('Failed to create workflow:', error);
      alert('Failed to create workflow. Please try again.');
    } finally {
      setIsBuilding(false);
    }
  };

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      borderRadius: '12px',
      padding: '24px',
      border: '1px solid #333',
    }}>
      <h2 style={{ marginTop: 0, marginBottom: '16px' }}>Create Workflow from Description</h2>
      
      <div style={{ marginBottom: '16px' }}>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe your workflow in natural language..."
          style={{
            width: '100%',
            minHeight: '100px',
            padding: '12px',
            backgroundColor: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: '8px',
            color: '#fff',
            fontSize: '14px',
            resize: 'vertical',
          }}
          disabled={isBuilding}
        />
      </div>
      
      <button
        onClick={handleCreateWithStreaming}
        disabled={isBuilding || !description.trim()}
        style={{
          ...styles.button,
          width: '100%',
          justifyContent: 'center',
          opacity: isBuilding || !description.trim() ? 0.5 : 1,
          cursor: isBuilding || !description.trim() ? 'not-allowed' : 'pointer',
        }}
      >
        {isBuilding ? (
          <>
            <Loader2 className="animate-spin" size={16} />
            Building Workflow...
          </>
        ) : (
          <>
            <Sparkles size={16} />
            Create Workflow
          </>
        )}
      </button>
      
      {isBuilding && (
        <div style={{ marginTop: '16px' }}>
          <div style={{
            width: '100%',
            height: '4px',
            backgroundColor: '#333',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${progress}%`,
              height: '100%',
              backgroundColor: '#0084ff',
              transition: 'width 0.3s ease',
            }} />
          </div>
          
          <div style={{
            marginTop: '12px',
            maxHeight: '200px',
            overflow: 'auto',
            backgroundColor: '#0a0a0a',
            borderRadius: '8px',
            padding: '12px',
            fontSize: '12px',
          }}>
            {streamEvents.map((event, idx) => (
              <div key={idx} style={{ marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ color: '#666' }}>
                    {new Date(event.timestamp || 0).toLocaleTimeString()}
                  </span>
                  {event.type === 'agent' && (
                    <span style={{ color: '#3b82f6' }}>
                      <Brain size={12} className="inline mr-1" />
                      Created agent: {event.data.name}
                    </span>
                  )}
                  {event.type === 'tool' && (
                    <span style={{ color: '#10b981' }}>
                      <Code size={12} className="inline mr-1" />
                      Added tool: {event.data.tool}
                    </span>
                  )}
                  {event.type === 'edge' && (
                    <span style={{ color: '#f59e0b' }}>
                      <Network size={12} className="inline mr-1" />
                      Connected: {event.data.from_agent} → {event.data.to_agent}
                    </span>
                  )}
                  {event.type === 'progress' && (
                    <span style={{ color: '#666' }}>
                      {event.data.message}
                    </span>
                  )}
                  {event.type === 'error' && (
                    <span style={{ color: '#ef4444' }}>
                      Error: {event.data.error}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {createdWorkflow && (
        <div style={{
          marginTop: '16px',
          padding: '16px',
          backgroundColor: '#064e3b',
          borderRadius: '8px',
          border: '1px solid #10b981',
        }}>
          <h3 style={{ marginTop: 0, marginBottom: '8px', color: '#10b981' }}>
            ✓ Created Workflow: {createdWorkflow.name}
          </h3>
          <p style={{ marginBottom: '16px', opacity: 0.9 }}>{createdWorkflow.description}</p>
          <div>
            <h4 style={{ marginBottom: '8px' }}>Agents ({createdWorkflow.agents.length}):</h4>
            {createdWorkflow.agents.map((agent, index) => (
              <div key={index} style={{ 
                paddingLeft: '16px', 
                fontSize: '14px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '4px'
              }}>
                <Brain size={14} style={{ color: '#3b82f6' }} />
                <span style={{ fontWeight: 'bold' }}>{agent.name}</span> - {agent.role}
                {agent.tools && agent.tools.length > 0 && (
                  <span style={{ opacity: 0.7 }}>({agent.tools.length} tools)</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ===== Main Enhanced Dashboard Component =====
const EnhancedROWBOATDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'workflow' | 'monitor'>('workflow');
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  const tabs = [
    { id: 'workflow', label: 'Workflow Designer', icon: <GitBranch size={16} /> },
    { id: 'monitor', label: 'Monitoring', icon: <Activity size={16} /> },
  ];

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Network size={24} style={{ color: '#0084ff' }} />
          <h1 style={{ margin: 0, fontSize: '20px' }}>ROWBOAT Dashboard</h1>
          <MCPStatusIndicator />
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => setShowHelp(!showHelp)}
            style={{
              ...styles.secondaryButton,
              padding: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <HelpCircle size={18} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ padding: '0 24px', backgroundColor: '#1a1a1a', borderBottom: '1px solid #333' }}>
        <div style={{ display: 'flex', gap: '4px' }}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              style={{
                padding: '12px 24px',
                backgroundColor: activeTab === tab.id ? '#0a0a0a' : 'transparent',
                color: activeTab === tab.id ? '#0084ff' : '#888',
                border: 'none',
                borderBottom: activeTab === tab.id ? '2px solid #0084ff' : '2px solid transparent',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
              onClick={() => setActiveTab(tab.id as any)}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'workflow' && (
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Natural Language Input */}
            <div style={styles.toolbar}>
              <StreamingWorkflowBuilder />
            </div>
            
            {/* Visual Editor */}
            <ReactFlowProvider>
              <EnhancedWorkflowVisualEditor workflowId={selectedWorkflowId || undefined} />
            </ReactFlowProvider>
          </div>
        )}
        
        {activeTab === 'monitor' && (
          <div style={{ padding: '24px' }}>
            <div style={{
              backgroundColor: '#1a1a1a',
              borderRadius: '12px',
              padding: '24px',
              border: '1px solid #333',
              textAlign: 'center',
            }}>
              <Activity size={48} style={{ color: '#666', marginBottom: '16px' }} />
              <h2 style={{ marginTop: 0, marginBottom: '8px' }}>System Monitoring</h2>
              <p style={{ color: '#888', margin: 0 }}>
                Real-time monitoring and analytics coming soon...
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Help Modal */}
      {showHelp && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '16px',
          zIndex: 50,
        }}>
          <div style={{
            backgroundColor: '#1a1a1a',
            borderRadius: '12px',
            padding: '24px',
            maxWidth: '600px',
            maxHeight: '80vh',
            overflow: 'auto',
            border: '1px solid #333',
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '16px'
            }}>
              <h3 style={{ margin: 0 }}>ROWBOAT Help</h3>
              <button
                onClick={() => setShowHelp(false)}
                style={{ 
                  background: 'none',
                  border: 'none',
                  color: '#999',
                  cursor: 'pointer'
                }}
              >
                <X size={20} />
              </button>
            </div>
            
            <div style={{ lineHeight: 1.6 }}>
              <section style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '8px' }}>Getting Started</h4>
                <p style={{ color: '#ccc', margin: 0 }}>
                  ROWBOAT is a multi-agent workflow system that allows you to create complex AI workflows
                  using natural language descriptions. Agents can communicate via @mentions and pass
                  control between each other.
                </p>
              </section>
              
              <section style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '8px' }}>Building Workflows</h4>
                <ul style={{ margin: 0, paddingLeft: '20px', color: '#ccc' }}>
                  <li>Describe your workflow in plain English</li>
                  <li>The AI will create agents and connections automatically</li>
                  <li>Use the visual editor to fine-tune your workflow</li>
                  <li>Add MCP servers for external tool integration</li>
                </ul>
              </section>
              
              <section style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '8px' }}>Features</h4>
                <ul style={{ margin: 0, paddingLeft: '20px', color: '#ccc' }}>
                  <li><strong>Collapsible Canvas:</strong> Expand/collapse the workflow view</li>
                  <li><strong>Agent Avatars:</strong> Visual identification of agents</li>
                  <li><strong>Test Panel:</strong> Test workflows with custom inputs</li>
                  <li><strong>AI Enhancement:</strong> Get suggestions to improve your workflow</li>
                  <li><strong>Real-time Streaming:</strong> Watch workflows build in real-time</li>
                </ul>
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Export all components
export {
  EnhancedROWBOATDashboard,
  EnhancedROWBOATDashboard as ROWBOATDashboard,
  StreamingWorkflowBuilder,
  StreamingWorkflowBuilder as WorkflowBuilder,
  EnhancedWorkflowVisualEditor,
  EnhancedWorkflowVisualEditor as WorkflowVisualEditor,
  MCPStatusIndicator,
  EnhancedROWBOATApiService,
  EnhancedROWBOATWebSocket,
};

// Default export
export default EnhancedROWBOATDashboard;