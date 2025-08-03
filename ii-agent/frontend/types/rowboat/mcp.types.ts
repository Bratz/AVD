// types/rowboat/mcp.types.ts

export interface MCPServer {
  name: string;
  url: string;
  tools: string[];
  status: MCPServerStatus;
  error?: string;
}

export type MCPServerStatus = 
  | 'connected' 
  | 'disconnected' 
  | 'error' 
  | 'available' 
  | 'unknown';

export interface MCPStatus {
  mcp_enabled: boolean;
  connection_status: string;
  available_tools: string[];
  error?: string;
}