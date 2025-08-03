// components/workflow/nodes/AgentNode.tsx
import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Brain, Zap, AlertCircle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

type NodeStatus = 'idle' | 'thinking' | 'executing' | 'error' | 'complete';

const statusIcons: Record<NodeStatus, React.ReactElement | null> = {
  idle: null,
  thinking: <Brain className="w-4 h-4 animate-pulse" />,
  executing: <Zap className="w-4 h-4 animate-spin" />,
  error: <AlertCircle className="w-4 h-4" />,
  complete: <CheckCircle className="w-4 h-4" />,
};

const statusColors: Record<NodeStatus, string> = {
  idle: 'border-gray-300 bg-white',
  thinking: 'border-yellow-500 bg-yellow-50',
  executing: 'border-blue-500 bg-blue-50',
  error: 'border-red-500 bg-red-50',
  complete: 'border-green-500 bg-green-50',
};

export const AgentNode = memo(({ data, selected }: NodeProps) => {
  const status = (data.status as NodeStatus) || 'idle';

  return (
    <div
      className={cn(
        'px-4 py-3 rounded-lg border-2 min-w-[200px] transition-all',
        statusColors[status],
        selected && 'ring-2 ring-blue-500 ring-offset-2'
      )}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-gray-400"
      />
      
      <div className="flex items-center justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-sm">{data.label}</h3>
            {statusIcons[status]}
          </div>
          {data.role && (
            <p className="text-xs text-gray-600 mt-1">Role: {data.role}</p>
          )}
          {data.tools && data.tools.length > 0 && (
            <div className="flex gap-1 mt-2">
              {data.tools.slice(0, 3).map((tool: string, idx: number) => (
                <span
                  key={idx}
                  className="text-xs bg-gray-200 px-1.5 py-0.5 rounded"
                >
                  {tool}
                </span>
              ))}
              {data.tools.length > 3 && (
                <span className="text-xs text-gray-500">
                  +{data.tools.length - 3}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-gray-400"
      />
    </div>
  );
});

AgentNode.displayName = 'AgentNode';