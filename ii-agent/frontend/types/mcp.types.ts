export interface MCPServer {
  id: string;
  name: string;
  description: string;
  serverUrl?: string;
  tools: MCPTool[];
  availableTools?: MCPTool[];
  isActive: boolean;
  isReady: boolean;
  serverType: 'hosted' | 'custom';
  authNeeded: boolean;
  isAuthenticated: boolean;
  instanceId?: string;
  status?: 'connected' | 'disconnected' | 'error' | 'checking';
  error?: string;
}

export interface MCPTool {
  id: string;
  name: string;
  description: string;
  parameters?: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
}

export interface MCPAuthResponse {
  authUrl: string;
  instanceId: string;
}

export interface EnableServerResponse {
  instanceId?: string;
  billingError?: string;
  error?: string;
}