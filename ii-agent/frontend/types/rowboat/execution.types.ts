// types/rowboat/execution.types.ts

export interface WorkflowExecution {
  workflow_id: string;
  execution_id: string;
  status: string;
  current_agent?: string;
  messages: ExecutionMessage[];
  metrics?: Record<string, AgentMetrics>;
}

export interface ExecutionMessage {
  id?: string;
  agent: string;
  content: string;
  timestamp: string;
  type: MessageType;
  metadata?: Record<string, any>;
  visibility?: 'internal' | 'external';
}

export type MessageType = 
  | 'thinking' 
  | 'response' 
  | 'handoff' 
  | 'tool_call' 
  | 'tool_result' 
  | 'error';

export interface StreamEvent {
  type: StreamEventType;
  data: any;
  timestamp?: number;
}

export type StreamEventType = 
  | 'agent' 
  | 'tool' 
  | 'prompt' 
  | 'edge' 
  | 'progress' 
  | 'complete' 
  | 'error' 
  | 'message' 
  | 'done';

export interface AgentMetrics {
  totalTokens: number;
  totalDuration: number;
  toolCallCount: number;
  thoughtSteps: number;
  errorCount: number;
  successRate: number;
}

export interface TestResult {
  success: boolean;
  trace?: TraceStep[];
  error?: string;
}

export interface TraceStep {
  agent: string;
  action: string;
  output: string;
}