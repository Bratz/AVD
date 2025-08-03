// frontend/src/services/langGraphAPI.ts
import axios, { AxiosInstance } from 'axios';

// Types matching backend
interface WorkflowDefinition {
  name: string;
  description: string;
  agents: Array<{
    name: string;
    role: string;
    instructions: string;
    tools: string[];
    temperature?: number;
    max_tokens?: number;
    model?: string;
  }>;
  edges: Array<{
    from_agent: string;
    to_agent: string;
    condition?: string;
  }>;
  metadata?: Record<string, any>;
}

interface WorkflowResponse {
  workflow_id: string;
  status: string;
  message: string;
}

interface ExecutionResponse {
  success: boolean;
  execution_id: string;
  result: any;
  agents_used: string[];
  metadata: Record<string, any>;
}

interface TestResult {
  scenario?: any;
  execution_result: any;
  assertions?: Array<{ assertion: any; passed: boolean }>;
  passed?: boolean;
}

interface EnhancementResponse {
  workflow_id: string;
  enhancements: {
    auto_tool_suggestion: Record<string, string[]>;
    workflow_optimization: any;
    error_recovery: any;
    performance_tuning: any;
    security_audit: any;
  };
  status: string;
}

class LangGraphAPI {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000', apiKey?: string) {
    this.api = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { Authorization: `Bearer ${apiKey}` }),
      },
    });

    // Add request/response interceptors for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Workflow Management
  async listWorkflows(category?: string, limit: number = 100, offset: number = 0) {
    const response = await this.api.get('/rowboat/workflows', {
      params: { category, limit, offset },
    });
    return response.data;
  }

  async createWorkflow(definition: WorkflowDefinition): Promise<WorkflowResponse> {
    const response = await this.api.post('/rowboat/workflows', definition);
    return response.data;
  }

  async createWorkflowFromDescription(
    description: string,
    options?: {
      examples?: Array<{ input: string; output: string }>;
      documents?: string[];
      model_preferences?: Record<string, string>;
    }
  ): Promise<WorkflowResponse> {
    const response = await this.api.post('/rowboat/workflows/from-description', {
      description,
      ...options,
    });
    return response.data;
  }

  async getWorkflow(workflowId: string) {
    const response = await this.api.get(`/rowboat/workflows/${workflowId}`);
    return response.data;
  }

  async updateWorkflow(workflowId: string, updates: Partial<WorkflowDefinition>) {
    const response = await this.api.put(`/rowboat/workflows/${workflowId}`, updates);
    return response.data;
  }

  async deleteWorkflow(workflowId: string) {
    const response = await this.api.delete(`/rowboat/workflows/${workflowId}`);
    return response.data;
  }

  // Workflow Execution
  async executeWorkflow(
    workflowId: string,
    input: string | { messages?: any[]; context?: any; state?: any },
    options?: {
      stream?: boolean;
      mode?: 'execute' | 'chat' | 'playground';
    }
  ): Promise<ExecutionResponse> {
    const payload = typeof input === 'string' ? { input } : input;
    const response = await this.api.post(
      `/rowboat/workflows/${workflowId}/execute`,
      {
        ...payload,
        ...options,
      }
    );
    return response.data;
  }

  // Streaming execution
  async executeWorkflowStream(
    workflowId: string,
    input: string,
    onEvent: (event: any) => void
  ) {
    const response = await fetch(
      `${this.api.defaults.baseURL}/rowboat/workflows/${workflowId}/execute`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.api.defaults.headers.Authorization && {
            Authorization: this.api.defaults.headers.Authorization as string,
          }),
        },
        body: JSON.stringify({ input, stream: true }),
      }
    );

    if (!response.ok) throw new Error('Stream request failed');

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No reader available');

    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.slice(6));
            onEvent(event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  }

  // Chat
  async chat(workflowId: string, messages: any[], state?: any, sessionId?: string) {
    const response = await this.api.post('/rowboat/chat', {
      workflow_id: workflowId,
      messages,
      state,
      session_id: sessionId,
    });
    return response.data;
  }

  // Testing
  async testWorkflow(
    workflowId: string,
    options: {
      scenario_id?: string;
      custom_input?: string;
      mock_responses?: Record<string, any>;
    }
  ): Promise<TestResult> {
    const response = await this.api.post(
      `/rowboat/workflows/${workflowId}/test`,
      options
    );
    return response.data;
  }

  async getTestScenarios(workflowId: string) {
    const response = await this.api.get(
      `/rowboat/workflows/${workflowId}/test-scenarios`
    );
    return response.data;
  }

  // Templates
  async getTemplates() {
    const response = await this.api.get('/rowboat/templates');
    return response.data.templates;
  }

  async createFromTemplate(
    templateName: string,
    customizations?: Record<string, any>
  ): Promise<WorkflowResponse> {
    const response = await this.api.post('/rowboat/workflows/from-template', {
      template_name: templateName,
      customizations,
    });
    return response.data;
  }

  // Enhancements
  async enhanceWorkflow(
    workflowId: string,
    enhancementType: 'all' | 'tools' | 'routing' | 'error_handling' | 'performance' | 'security' = 'all'
  ): Promise<EnhancementResponse> {
    const response = await this.api.post(
      `/rowboat/workflows/${workflowId}/enhance`,
      { enhancement_type: enhancementType }
    );
    return response.data;
  }

  // Visual Builder
  async getVisualBuilderConfig() {
    const response = await this.api.get('/rowboat/visual-builder/config');
    return response.data;
  }

  async saveVisualWorkflow(visualData: any, workflowId?: string) {
    const endpoint = workflowId
      ? `/rowboat/workflows/${workflowId}/visual`
      : '/rowboat/workflows/visual';
    const method = workflowId ? 'put' : 'post';
    
    const response = await this.api[method](endpoint, visualData);
    return response.data;
  }

  // Metrics
  async getWorkflowMetrics(
    workflowId: string,
    startDate?: Date,
    endDate?: Date
  ) {
    const response = await this.api.get(
      `/rowboat/workflows/${workflowId}/metrics`,
      {
        params: {
          start_date: startDate?.toISOString(),
          end_date: endDate?.toISOString(),
        },
      }
    );
    return response.data;
  }

  // WebSocket connection for real-time execution
  connectWebSocket(workflowId: string): WebSocket {
    const wsUrl = this.api.defaults.baseURL?.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/rowboat/ws/${workflowId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected for workflow:', workflowId);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    return ws;
  }
}

// Export singleton instance
export const langGraphAPI = new LangGraphAPI();

// Export class for custom instances
export { LangGraphAPI };

// Export types
export type {
  WorkflowDefinition,
  WorkflowResponse,
  ExecutionResponse,
  TestResult,
  EnhancementResponse,
};