// types/rowboat/workflow.types.ts

export interface WorkflowDefinition {
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

export interface AgentConfig {
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
}

export interface WorkflowEdge {
  from_agent: string;
  to_agent: string;
  condition?: string;
  isMentionBased?: boolean;
}

export interface WorkflowVisualization {
  nodes: VisualizationNode[];
  edges: VisualizationEdge[];
  metadata: Record<string, any>;
}

export interface VisualizationNode {
  id: string;
  type: string;
  data: NodeData;
  position: { x: number; y: number };
}

export interface NodeData {
  label: string;
  role: string;
  hasInstructions: boolean;
  connectedAgents: string[];
  tools?: string[];
  model?: string;
  status?: AgentStatus;
  metrics?: AgentMetrics;
  outputVisibility?: 'user_facing' | 'internal';
  name?: string;
  instructions?: string;
}

export interface VisualizationEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  animated: boolean;
  label?: string;
  data?: WorkflowEdge;
}

export type AgentStatus = 'idle' | 'thinking' | 'executing' | 'error' | 'complete';

export interface AgentMetrics {
  totalTokens: number;
  totalDuration: number;
  toolCallCount: number;
  thoughtSteps: number;
  errorCount: number;
  successRate: number;
}

// Enhanced ROWBOAT Node Data for ReactFlow
export interface ROWBOATNodeData extends NodeData {
  mcpServers?: string[];
  examples?: string[];
  isParallel?: boolean;
  isSubworkflow?: boolean;
  hasApproval?: boolean;
  suggestedTools?: string[];
  mentionedAgents?: string[];
}

export interface ROWBOATEdgeData {
  condition?: string | null;
  isMentionBased?: boolean;
}