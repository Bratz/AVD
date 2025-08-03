// components/workflow/WorkflowExecutionMonitor.tsx
import React from 'react';
import { useSSE } from '@/hooks/useSSE';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Activity, AlertCircle, CheckCircle, Clock } from 'lucide-react';

interface ExecutionLog {
  timestamp: string;
  nodeId: string;
  nodeName: string;
  event: string;
  details?: any;
}

interface WorkflowExecutionMonitorProps {
  workflowId: string;
  executionId: string;
}

export function WorkflowExecutionMonitor({ 
  workflowId, 
  executionId 
}: WorkflowExecutionMonitorProps) {
  const [logs, setLogs] = React.useState<ExecutionLog[]>([]);
  const [status, setStatus] = React.useState<string>('running');

  const { isConnected } = useSSE(
    `${process.env.NEXT_PUBLIC_API_URL}/api/workflows/${workflowId}/executions/${executionId}/stream`,
    {
      onMessage: (data) => {
        if (data.type === 'log') {
          setLogs((prev) => [...prev, data.log]);
        } else if (data.type === 'status') {
          setStatus(data.status);
        }
      },
      reconnectInterval: 5000,
    }
  );

  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <Activity className="w-4 h-4 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  const getBadgeVariant = () => {
    switch (status) {
      case 'running':
        return 'default';
      case 'completed':
        return 'outline'; // Changed from 'success' to 'outline'
      case 'failed':
        return 'destructive';
      default:
        return 'secondary';
    }
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Execution Monitor</span>
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <Badge variant={getBadgeVariant()}>
              {status}
            </Badge>
            {isConnected && (
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px]">
          <div className="space-y-2">
            {logs.map((log, index) => (
              <div
                key={index}
                className="text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">{log.nodeName}</span>
                  <span className="text-xs text-gray-500">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-gray-600 dark:text-gray-400">{log.event}</p>
                {log.details && (
                  <pre className="text-xs mt-1 text-gray-500">
                    {JSON.stringify(log.details, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}