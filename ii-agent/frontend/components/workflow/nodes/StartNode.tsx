// components/workflow/nodes/StartNode.tsx
import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Play } from 'lucide-react';

export const StartNode = memo(({ data, selected }: NodeProps) => {
  return (
    <div
      className={`
        px-4 py-2 rounded-full border-2 border-green-500 bg-green-50
        ${selected ? 'ring-2 ring-green-500 ring-offset-2' : ''}
      `}
    >
      <div className="flex items-center gap-2">
        <Play className="w-4 h-4 text-green-600" />
        <span className="font-medium text-sm">{data?.label || 'Start'}</span>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-green-500"
      />
    </div>
  );
});

StartNode.displayName = 'StartNode';