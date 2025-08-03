// services/rowboat/ROWBOATApiService.ts

import type { 
  WorkflowDefinition, 
  WorkflowVisualization,
  StreamEvent,
  TestResult
} from '@/types/rowboat/workflow.types';
import type { MCPServer, MCPStatus } from '@/types/rowboat/mcp.types';

export class ROWBOATApiService {
  private _baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl?: string, apiKey?: string) {
    this._baseUrl = baseUrl || process.env.NEXT_PUBLIC_ROWBOAT_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9000';
    this.apiKey = apiKey || process.env.NEXT_PUBLIC_ROWBOAT_API_KEY;
  }

  // Make baseUrl accessible for components
  get baseUrl(): string {
    return this._baseUrl;
  }

  getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  // Workflow Management
  async listWorkflows(category?: string, limit: number = 100, offset: number = 0): Promise<any[]> {
    try {
      const params = new URLSearchParams();
      if (category) params.append('category', category);
      params.append('limit', limit.toString());
      params.append('offset', offset.toString());
      
      const response = await fetch(`${this._baseUrl}/rowboat/workflows?${params}`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        console.warn('Failed to list workflows');
        return [];
      }
      
      const data = await response.json();
      // API returns { workflows: [], total: number, limit: number, offset: number }
      return data.workflows || [];
    } catch (error) {
      console.error('Failed to list workflows:', error);
      return [];
    }
  }

  async getWorkflow(workflowId: string): Promise<WorkflowDefinition> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get workflow');
      }
      
      const data = await response.json();
      
      // Transform API response to match WorkflowDefinition interface
      return {
        name: data.name,
        description: data.description,
        agents: data.agents,
        edges: data.edges,
        entry_point: data.entry_point || data.agents[0]?.name || 'entry',
        metadata: {
          ...data.metadata,
          workflow_id: data.id || workflowId,
          created_at: data.created_at,
          updated_at: data.updated_at
        }
      };
    } catch (error) {
      console.error('Failed to get workflow:', error);
      throw error;
    }
  }

  async createWorkflow(workflow: WorkflowDefinition): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          name: workflow.name,
          description: workflow.description,
          agents: workflow.agents,
          edges: workflow.edges,
          metadata: workflow.metadata
        }),
      });
      
      if (!response.ok) throw new Error('Failed to create workflow');
      return response.json();
    } catch (error) {
      console.error('Failed to create workflow:', error);
      throw error;
    }
  }

  async updateWorkflow(workflowId: string, workflow: Partial<WorkflowDefinition>): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}`, {
        method: 'PUT',
        headers: this.getHeaders(),
        body: JSON.stringify(workflow),
      });
      
      if (!response.ok) throw new Error('Failed to update workflow');
      return response.json();
    } catch (error) {
      console.error('Failed to update workflow:', error);
      throw error;
    }
  }

  async deleteWorkflow(workflowId: string): Promise<void> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}`, {
        method: 'DELETE',
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to delete workflow');
    } catch (error) {
      console.error('Failed to delete workflow:', error);
      throw error;
    }
  }

  // Natural Language Workflow Creation
  async createWorkflowFromDescription(
    description: string, 
    context?: {
      examples?: Array<{ input: string; output: string }>;
      documents?: string[];
      model_preferences?: Record<string, string>;
    }
  ): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/from-description`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          description,
          examples: context?.examples || [],
          documents: context?.documents || [],
          model_preferences: context?.model_preferences || {}
        }),
      });
      
      if (!response.ok) throw new Error('Failed to create workflow from description');
      const data = await response.json();
      
      // Get the full workflow details
      if (data.workflow_id) {
        return await this.getWorkflow(data.workflow_id);
      }
      
      return data;
    } catch (error) {
      console.error('Failed to create workflow from description:', error);
      throw error;
    }
  }

  // Streaming workflow creation from natural language
  async *createWorkflowFromDescriptionStream(
    description: string,
    context?: Record<string, any>
  ): AsyncGenerator<StreamEvent> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/from-description/stream`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          description,
          examples: context?.examples || [],
          documents: context?.documents || [],
          model_preferences: context?.model_preferences || {}
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error('Streaming not available');
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
              console.error('Failed to parse SSE event:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream creation error:', error);
      throw error;
    }
  }

  // Workflow Execution
  async executeWorkflow(workflowId: string, input: string | any): Promise<any> {
    try {
      const payload = typeof input === 'string' ? { input } : input;
      
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          ...payload,
          stream: false,
          mode: payload.mode || 'execute'
        }),
      });
      
      if (!response.ok) throw new Error('Failed to execute workflow');
      return response.json();
    } catch (error) {
      console.error('Failed to execute workflow:', error);
      throw error;
    }
  }

  // Streaming execution
  async *executeWorkflowStream(workflowId: string, input: string | any): AsyncGenerator<any> {
    try {
      const payload = typeof input === 'string' ? { input } : input;
      
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          ...payload,
          stream: true,
          mode: payload.mode || 'execute'
        }),
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

  // Chat API
  async chat(workflowId: string, messages: any[], state?: any, sessionId?: string): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/chat`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          workflow_id: workflowId,
          messages,
          state,
          session_id: sessionId
        }),
      });
      
      if (!response.ok) throw new Error('Failed to chat');
      return response.json();
    } catch (error) {
      console.error('Chat failed:', error);
      throw error;
    }
  }

  // Testing
  async testWorkflow(workflowId: string, testInput: string): Promise<TestResult> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/test`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ 
          custom_input: testInput,
          mock_responses: {}
        }),
      });
      
      if (!response.ok) throw new Error('Failed to test workflow');
      
      const data = await response.json();
      
      // Transform API response to TestResult format
      return {
        success: data.success || false,
        trace: data.test_result?.result?.agent_outputs 
          ? Object.entries(data.test_result.result.agent_outputs).map(([agent, output]: [string, any]) => ({
              agent,
              action: 'Executed',
              output: output.output || output.message || 'No output'
            }))
          : [],
        error: data.error
      };
    } catch (error) {
      console.error('Failed to test workflow:', error);
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  // Visualization
  async getVisualization(workflowId: string): Promise<WorkflowVisualization> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/visualization`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get visualization');
      return response.json();
    } catch (error) {
      console.error('Failed to get visualization:', error);
      // Return empty visualization on error
      return { nodes: [], edges: [], metadata: {} };
    }
  }

  // Metrics
  async getWorkflowMetrics(workflowId: string, startDate?: Date, endDate?: Date): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (startDate) params.append('start_date', startDate.toISOString());
      if (endDate) params.append('end_date', endDate.toISOString());
      
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/metrics?${params}`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get metrics');
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

  // Templates
  async getTemplates(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/templates`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get templates');
      const data = await response.json();
      return data.templates || {};
    } catch (error) {
      console.error('Failed to get templates:', error);
      return {};
    }
  }

  // Enhancement
  async enhanceWorkflow(workflowId: string, enhancementType: string = 'all'): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/enhance`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ enhancement_type: enhancementType }),
      });
      
      if (!response.ok) throw new Error('Failed to enhance workflow');
      return response.json();
    } catch (error) {
      console.error('Failed to enhance workflow:', error);
      throw error;
    }
  }

  // MCP Integration
  async getMCPServers(): Promise<MCPServer[]> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/mcp/servers`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP servers');
      }
      
      const data = await response.json();
      return data.servers || [];
    } catch (error) {
      console.error('Failed to get MCP servers:', error);
      return [];
    }
  }

  async getMCPStatus(): Promise<MCPStatus> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/mcp/status`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP status');
      }
      
      const data = await response.json();
      return {
        mcp_enabled: data.mcp_enabled || false,
        connection_status: data.connection_status || 'disconnected',
        available_tools: data.available_tools || [],
        error: data.last_error
      };
    } catch (error) {
      return {
        mcp_enabled: false,
        connection_status: 'disconnected',
        available_tools: [],
        error: (error as Error).message
      };
    }
  }

  async getMCPHealth(): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/mcp/health`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP health');
      }
      
      return response.json();
    } catch (error) {
      console.error('Failed to get MCP health:', error);
      return {
        mcp_wrapper: false,
        services: {},
        error: (error as Error).message
      };
    }
  }

  // Visual Builder Config
  async getVisualBuilderConfig(): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/visual-builder/config`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get visual builder config');
      return response.json();
    } catch (error) {
      console.error('Failed to get visual builder config:', error);
      return { nodes: [], edges: [] };
    }
  }
  async getWorkflowMetrics(workflowId: string, startDate?: Date, endDate?: Date): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (startDate) params.append('start_date', startDate.toISOString());
      if (endDate) params.append('end_date', endDate.toISOString());
      
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/metrics?${params}`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get metrics');
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

  // Templates
  async getTemplates(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/templates`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get templates');
      const data = await response.json();
      return data.templates || {};
    } catch (error) {
      console.error('Failed to get templates:', error);
      return {};
    }
  }

  // Enhancement
  async enhanceWorkflow(workflowId: string, enhancementType: string = 'all'): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/workflows/${workflowId}/enhance`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ enhancement_type: enhancementType }),
      });
      
      if (!response.ok) throw new Error('Failed to enhance workflow');
      return response.json();
    } catch (error) {
      console.error('Failed to enhance workflow:', error);
      throw error;
    }
  }

  // MCP Integration
  async getMCPServers(): Promise<MCPServer[]> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/mcp/servers`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP servers');
      }
      
      const data = await response.json();
      return data.servers || [];
    } catch (error) {
      console.error('Failed to get MCP servers:', error);
      return [];
    }
  }

  async getMCPStatus(): Promise<MCPStatus> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/mcp/status`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP status');
      }
      
      const data = await response.json();
      return {
        mcp_enabled: data.mcp_enabled || false,
        connection_status: data.connection_status || 'disconnected',
        available_tools: data.available_tools || [],
        error: data.last_error
      };
    } catch (error) {
      return {
        mcp_enabled: false,
        connection_status: 'disconnected',
        available_tools: [],
        error: (error as Error).message
      };
    }
  }

  async getMCPHealth(): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/mcp/health`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP health');
      }
      
      return response.json();
    } catch (error) {
      console.error('Failed to get MCP health:', error);
      return {
        mcp_wrapper: false,
        services: {},
        error: (error as Error).message
      };
    }
  }

  // Visual Builder Config
  async getVisualBuilderConfig(): Promise<any> {
    try {
      const response = await fetch(`${this._baseUrl}/rowboat/visual-builder/config`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get visual builder config');
      return response.json();
    } catch (error) {
      console.error('Failed to get visual builder config:', error);
      return { nodes: [], edges: [] };
    }
  }

  // Tool Suggestions (client-side implementation)
  async suggestTools(agentRole: string, instructions: string): Promise<string[]> {
    // This is kept client-side as the API doesn't have this endpoint
    const suggestions: Record<string, string[]> = {
      researcher: ['web_search', 'document_parser', 'mcp:research_server:deep_search'],
      analyzer: ['data_analyzer', 'chart_generator', 'mcp:analytics_server:analyze'],
      writer: ['markdown_formatter', 'grammar_checker', 'mcp:writing_server:enhance'],
      coordinator: ['task_router', 'status_tracker', 'mcp:workflow_server:coordinate'],
      specialist: ['data_processor', 'api_caller', 'database_query'],
      reviewer: ['quality_checker', 'approval_tool', 'feedback_generator'],
      customer_support: ['ticket_system', 'knowledge_base', 'mcp:support_server:assist'],
      intent_classifier: ['text_classifier', 'intent_detector', 'routing_engine'],
      billing_agent: ['payment_processor', 'account_lookup', 'subscription_manager'],
      tech_agent: ['knowledge_base_search', 'debug_analyzer', 'solution_generator']
    };
    
    // Get base suggestions for the role
    let toolSuggestions = suggestions[agentRole] || ['generic_tool'];
    
    // Add context-based suggestions based on instructions
    if (instructions.toLowerCase().includes('database') || instructions.toLowerCase().includes('sql')) {
      toolSuggestions.push('database_query');
    }
    if (instructions.toLowerCase().includes('api') || instructions.toLowerCase().includes('external')) {
      toolSuggestions.push('api_caller');
    }
    if (instructions.toLowerCase().includes('search') || instructions.toLowerCase().includes('find')) {
      toolSuggestions.push('web_search');
    }
    if (instructions.toLowerCase().includes('analyze') || instructions.toLowerCase().includes('analysis')) {
      toolSuggestions.push('data_analyzer');
    }
    if (instructions.toLowerCase().includes('mcp') || instructions.toLowerCase().includes('banking')) {
      toolSuggestions.push('mcp:banking_server:get_account', 'mcp:banking_server:list_transactions');
    }
    
    // Remove duplicates
    return [...new Set(toolSuggestions)];
  }
}
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get metrics');
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

  // Templates
  async getTemplates(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/templates`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get templates');
      const data = await response.json();
      return data.templates || {};
    } catch (error) {
      console.error('Failed to get templates:', error);
      return {};
    }
  }

  // Enhancement
  async enhanceWorkflow(workflowId: string, enhancementType: string = 'all'): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/workflows/${workflowId}/enhance`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ enhancement_type: enhancementType }),
      });
      
      if (!response.ok) throw new Error('Failed to enhance workflow');
      return response.json();
    } catch (error) {
      console.error('Failed to enhance workflow:', error);
      throw error;
    }
  }

  // MCP Integration
  async getMCPServers(): Promise<MCPServer[]> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/mcp/servers`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP servers');
      }
      
      const data = await response.json();
      return data.servers || [];
    } catch (error) {
      console.error('Failed to get MCP servers:', error);
      return [];
    }
  }

  async getMCPStatus(): Promise<MCPStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/mcp/status`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP status');
      }
      
      const data = await response.json();
      return {
        mcp_enabled: data.mcp_enabled || false,
        connection_status: data.connection_status || 'disconnected',
        available_tools: data.available_tools || [],
        error: data.last_error
      };
    } catch (error) {
      return {
        mcp_enabled: false,
        connection_status: 'disconnected',
        available_tools: [],
        error: (error as Error).message
      };
    }
  }

  async getMCPHealth(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/mcp/health`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get MCP health');
      }
      
      return response.json();
    } catch (error) {
      console.error('Failed to get MCP health:', error);
      return {
        mcp_wrapper: false,
        services: {},
        error: (error as Error).message
      };
    }
  }

  // Visual Builder Config
  async getVisualBuilderConfig(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/rowboat/visual-builder/config`, {
        headers: this.getHeaders(),
      });
      
      if (!response.ok) throw new Error('Failed to get visual builder config');
      return response.json();
    } catch (error) {
      console.error('Failed to get visual builder config:', error);
      return { nodes: [], edges: [] };
    }
  }

  // Tool Suggestions (client-side implementation)
  async suggestTools(agentRole: string, instructions: string): Promise<string[]> {
    // This is kept client-side as the API doesn't have this endpoint
    const suggestions: Record<string, string[]> = {
      researcher: ['web_search', 'document_parser', 'mcp:research_server:deep_search'],
      analyzer: ['data_analyzer', 'chart_generator', 'mcp:analytics_server:analyze'],
      writer: ['markdown_formatter', 'grammar_checker', 'mcp:writing_server:enhance'],
      coordinator: ['task_router', 'status_tracker', 'mcp:workflow_server:coordinate'],
      specialist: ['data_processor', 'api_caller', 'database_query'],
      reviewer: ['quality_checker', 'approval_tool', 'feedback_generator'],
      customer_support: ['ticket_system', 'knowledge_base', 'mcp:support_server:assist'],
      intent_classifier: ['text_classifier', 'intent_detector', 'routing_engine'],
      billing_agent: ['payment_processor', 'account_lookup', 'subscription_manager'],
      tech_agent: ['knowledge_base_search', 'debug_analyzer', 'solution_generator']
    };
    
    // Get base suggestions for the role
    let toolSuggestions = suggestions[agentRole] || ['generic_tool'];
    
    // Add context-based suggestions based on instructions
    if (instructions.toLowerCase().includes('database') || instructions.toLowerCase().includes('sql')) {
      toolSuggestions.push('database_query');
    }
    if (instructions.toLowerCase().includes('api') || instructions.toLowerCase().includes('external')) {
      toolSuggestions.push('api_caller');
    }
    if (instructions.toLowerCase().includes('search') || instructions.toLowerCase().includes('find')) {
      toolSuggestions.push('web_search');
    }
    if (instructions.toLowerCase().includes('analyze') || instructions.toLowerCase().includes('analysis')) {
      toolSuggestions.push('data_analyzer');
    }
    if (instructions.toLowerCase().includes('mcp') || instructions.toLowerCase().includes('banking')) {
      toolSuggestions.push('mcp:banking_server:get_account', 'mcp:banking_server:list_transactions');
    }
    
    // Remove duplicates
    return [...new Set(toolSuggestions)];
  }
}