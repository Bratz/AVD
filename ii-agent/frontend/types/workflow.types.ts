// types/workflow.types.ts
import type { Node as ReactFlowNode, Edge as ReactFlowEdge } from 'reactflow';

export interface WorkflowNodeData {
  label: string;
  role?: string;
  instructions?: string;
  tools?: string[];
  model?: string;
  temperature?: number;
  outputVisibility?: 'user_facing' | 'internal';
  controlType?: 'retain' | 'relinquish_to_parent' | 'start_agent';
  status?: 'idle' | 'thinking' | 'executing' | 'error' | 'complete';
}

export type WorkflowNode = ReactFlowNode<WorkflowNodeData>;

export interface WorkflowEdgeData {
  condition?: string;
  isMentionBased?: boolean;
}

export type WorkflowEdge = ReactFlowEdge<WorkflowEdgeData>;

export interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  entryPoint: string;
  createdAt: string;
  updatedAt: string;
  version: number;
  metadata?: Record<string, any>;
}