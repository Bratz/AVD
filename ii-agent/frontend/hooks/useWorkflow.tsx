// frontend/src/hooks/useWorkflow.ts
import { useState, useCallback, useEffect } from 'react';
import { Node, Edge } from 'reactflow';
import { langGraphAPI, WorkflowDefinition, WorkflowResponse } from '../app/api/langGraph/langGraphAPI';

interface UseWorkflowOptions {
  autoSave?: boolean;
  autoSaveInterval?: number;
}

interface WorkflowState {
  id: string | null;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
  isDirty: boolean;
  isLoading: boolean;
  error: string | null;
  lastSaved: Date | null;
}

export const useWorkflow = (options: UseWorkflowOptions = {}) => {
  const { autoSave = false, autoSaveInterval = 30000 } = options;

  const [state, setState] = useState<WorkflowState>({
    id: null,
    name: 'Untitled Workflow',
    description: '',
    nodes: [],
    edges: [],
    isDirty: false,
    isLoading: false,
    error: null,
    lastSaved: null,
  });

  // Auto-save functionality
  useEffect(() => {
    if (!autoSave || !state.isDirty || !state.id) return;

    const timer = setTimeout(async () => {
      await saveWorkflow();
    }, autoSaveInterval);

    return () => clearTimeout(timer);
  }, [state.isDirty, state.id, autoSave, autoSaveInterval]);

  const setNodes = useCallback((nodes: Node[] | ((prev: Node[]) => Node[])) => {
    setState(prev => ({
      ...prev,
      nodes: typeof nodes === 'function' ? nodes(prev.nodes) : nodes,
      isDirty: true,
    }));
  }, []);

  const setEdges = useCallback((edges: Edge[] | ((prev: Edge[]) => Edge[])) => {
    setState(prev => ({
      ...prev,
      edges: typeof edges === 'function' ? edges(prev.edges) : edges,
      isDirty: true,
    }));
  }, []);

  const createWorkflow = useCallback(async (): Promise<WorkflowResponse | null> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const definition = nodesToWorkflowDefinition(state);
      const response = await langGraphAPI.createWorkflow(definition);
      
      setState(prev => ({
        ...prev,
        id: response.workflow_id,
        isDirty: false,
        isLoading: false,
        lastSaved: new Date(),
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to create workflow',
      }));
      return null;
    }
  }, [state]);

  const saveWorkflow = useCallback(async (): Promise<boolean> => {
    if (!state.id) {
      const response = await createWorkflow();
      return response !== null;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const definition = nodesToWorkflowDefinition(state);
      await langGraphAPI.updateWorkflow(state.id, definition);
      
      setState(prev => ({
        ...prev,
        isDirty: false,
        isLoading: false,
        lastSaved: new Date(),
      }));

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to save workflow',
      }));
      return false;
    }
  }, [state, createWorkflow]);

  const loadWorkflow = useCallback(async (workflowId: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const workflow = await langGraphAPI.getWorkflow(workflowId);
      const { nodes, edges } = workflowDefinitionToNodes(workflow);
      
      setState(prev => ({
        ...prev,
        id: workflow.id,
        name: workflow.name,
        description: workflow.description,
        nodes,
        edges,
        isDirty: false,
        isLoading: false,
        lastSaved: new Date(workflow.updated_at),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load workflow',
      }));
    }
  }, []);

  const deployWorkflow = useCallback(async () => {
    if (!state.id) {
      await createWorkflow();
    }
    // Deployment is handled by creation/update in this context
    return state.id;
  }, [state.id, createWorkflow]);

  const deleteWorkflow = useCallback(async (): Promise<boolean> => {
    if (!state.id) return false;

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await langGraphAPI.deleteWorkflow(state.id);
      
      setState({
        id: null,
        name: 'Untitled Workflow',
        description: '',
        nodes: [],
        edges: [],
        isDirty: false,
        isLoading: false,
        error: null,
        lastSaved: null,
      });

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to delete workflow',
      }));
      return false;
    }
  }, [state.id]);

  return {
    // State
    ...state,
    
    // Actions
    setNodes,
    setEdges,
    setName: (name: string) => setState(prev => ({ ...prev, name, isDirty: true })),
    setDescription: (description: string) => setState(prev => ({ ...prev, description, isDirty: true })),
    
    // API operations
    createWorkflow,
    saveWorkflow,
    loadWorkflow,
    deployWorkflow,
    deleteWorkflow,
    
    // Utility
    clearError: () => setState(prev => ({ ...prev, error: null })),
  };
};

// Helper functions
function nodesToWorkflowDefinition(state: WorkflowState): WorkflowDefinition {
  return {
    name: state.name,
    description: state.description,
    agents: state.nodes.map(node => ({
      name: node.data.name || node.id,
      role: node.data.role || 'generic',
      instructions: node.data.instructions || '',
      tools: node.data.tools || [],
      model: node.data.model,
      temperature: node.data.temperature,
      max_tokens: node.data.max_tokens,
    })),
    edges: state.edges.map(edge => {
      const sourceNode = state.nodes.find(n => n.id === edge.source);
      const targetNode = state.nodes.find(n => n.id === edge.target);
      return {
        from_agent: sourceNode?.data.name || edge.source,
        to_agent: targetNode?.data.name || edge.target,
        condition: edge.data?.condition,
      };
    }),
  };
}

function workflowDefinitionToNodes(workflow: any): { nodes: Node[], edges: Edge[] } {
  const nodes: Node[] = workflow.agents.map((agent: any, index: number) => ({
    id: agent.name,
    type: 'custom',
    position: { 
      x: 250 + (index % 3) * 200, 
      y: 100 + Math.floor(index / 3) * 150 
    },
    data: {
      name: agent.name,
      role: agent.role,
      model: agent.model || 'gpt-4-turbo-preview',
      tools: agent.tools || [],
      instructions: agent.instructions || '',
    },
  }));

  const edges: Edge[] = workflow.edges.map((edge: any, index: number) => ({
    id: `edge-${index}`,
    source: edge.from_agent,
    target: edge.to_agent,
    type: 'smoothstep',
    animated: true,
    data: {
      condition: edge.condition,
    },
  }));

  return { nodes, edges };
}

// Hook for workflow execution
export const useWorkflowExecution = (workflowId: string | null) => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [executionLogs, setExecutionLogs] = useState<string[]>([]);

  const execute = useCallback(async (input: string | any) => {
    if (!workflowId) {
      setExecutionError('No workflow ID provided');
      return null;
    }

    setIsExecuting(true);
    setExecutionError(null);
    setExecutionLogs([]);

    try {
      setExecutionLogs(prev => [...prev, `Starting execution at ${new Date().toISOString()}`]);
      
      const result = await langGraphAPI.executeWorkflow(workflowId, input);
      
      setExecutionResult(result);
      setExecutionLogs(prev => [...prev, `Execution completed successfully`]);
      
      return result;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Execution failed';
      setExecutionError(errorMsg);
      setExecutionLogs(prev => [...prev, `Error: ${errorMsg}`]);
      return null;
    } finally {
      setIsExecuting(false);
    }
  }, [workflowId]);

  const executeStream = useCallback(async (input: string, onEvent: (event: any) => void) => {
    if (!workflowId) {
      setExecutionError('No workflow ID provided');
      return;
    }

    setIsExecuting(true);
    setExecutionError(null);
    setExecutionLogs([]);

    try {
      setExecutionLogs(prev => [...prev, `Starting streaming execution at ${new Date().toISOString()}`]);
      
      await langGraphAPI.executeWorkflowStream(workflowId, input, (event) => {
        setExecutionLogs(prev => [...prev, `Event: ${JSON.stringify(event)}`]);
        onEvent(event);
      });
      
      setExecutionLogs(prev => [...prev, `Streaming completed`]);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Streaming failed';
      setExecutionError(errorMsg);
      setExecutionLogs(prev => [...prev, `Error: ${errorMsg}`]);
    } finally {
      setIsExecuting(false);
    }
  }, [workflowId]);

  return {
    isExecuting,
    executionResult,
    executionError,
    executionLogs,
    execute,
    executeStream,
    clearExecutionState: () => {
      setExecutionResult(null);
      setExecutionError(null);
      setExecutionLogs([]);
    },
  };
};

// Hook for workflow templates
export const useWorkflowTemplates = () => {
  const [templates, setTemplates] = useState<Record<string, any>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTemplates = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const templatesData = await langGraphAPI.getTemplates();
      setTemplates(templatesData);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to load templates');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createFromTemplate = useCallback(async (templateName: string, customizations?: any) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await langGraphAPI.createFromTemplate(templateName, customizations);
      return response;
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to create from template');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTemplates();
  }, [loadTemplates]);

  return {
    templates,
    isLoading,
    error,
    loadTemplates,
    createFromTemplate,
  };
};