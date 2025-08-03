// hooks/useWorkflowWebSocket.ts
import { useEffect, useState, useRef } from 'react';
import io, { Socket } from 'socket.io-client';

interface NodeStatus {
  nodeId: string;
  status: 'idle' | 'thinking' | 'executing' | 'error' | 'complete';
  timestamp: string;
}

interface ExecutionStatus {
  executionId: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  currentNodeId?: string;
  error?: string;
}

export function useWorkflowWebSocket(projectId: string, workflowId?: string) {
  const [connected, setConnected] = useState(false);
  const [nodeStatus, setNodeStatus] = useState<NodeStatus | null>(null);
  const [executionStatus, setExecutionStatus] = useState<ExecutionStatus | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!workflowId) return;

    // Initialize WebSocket connection
    const socket = io(process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:9001', {
      path: '/ws',
      query: {
        projectId,
        workflowId,
      },
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('WebSocket connected');
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    });

    // Listen for node status updates
    socket.on('node:status', (data: NodeStatus) => {
      setNodeStatus(data);
    });

    // Listen for execution status updates
    socket.on('execution:status', (data: ExecutionStatus) => {
      setExecutionStatus(data);
    });

    // Join workflow room
    socket.emit('workflow:join', { workflowId });

    return () => {
      socket.emit('workflow:leave', { workflowId });
      socket.disconnect();
    };
  }, [projectId, workflowId]);

  const sendMessage = (event: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    }
  };

  return {
    connected,
    nodeStatus,
    executionStatus,
    sendMessage,
  };
}