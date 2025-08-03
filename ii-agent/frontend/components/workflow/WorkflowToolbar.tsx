// components/workflow/WorkflowToolbar.tsx
import React from 'react';
import { Button } from '@/components/ui/button';
import { Plus, Save, Play, Trash2, Pause, Download, Upload } from 'lucide-react';

interface WorkflowToolbarProps {
  onAddAgent: () => void;
  onSave: () => void;
  onExecute: () => void;
  onDelete: () => void;
  isExecuting: boolean;
  hasSelection: boolean;
}

export function WorkflowToolbar({
  onAddAgent,
  onSave,
  onExecute,
  onDelete,
  isExecuting,
  hasSelection,
}: WorkflowToolbarProps) {
  return (
    <div className="flex items-center gap-2 bg-white p-2 rounded-lg shadow-lg">
      <Button onClick={onAddAgent} size="sm" variant="outline">
        <Plus className="w-4 h-4 mr-2" />
        Add Agent
      </Button>
      
      <Button onClick={onSave} size="sm" variant="outline">
        <Save className="w-4 h-4 mr-2" />
        Save
      </Button>
      
      <Button
        onClick={onExecute}
        size="sm"
        variant={isExecuting ? "destructive" : "default"}
        disabled={isExecuting}
      >
        {isExecuting ? (
          <>
            <Pause className="w-4 h-4 mr-2" />
            Running...
          </>
        ) : (
          <>
            <Play className="w-4 h-4 mr-2" />
            Execute
          </>
        )}
      </Button>
      
      {hasSelection && (
        <Button
          onClick={onDelete}
          size="sm"
          variant="destructive"
        >
          <Trash2 className="w-4 h-4 mr-2" />
          Delete
        </Button>
      )}
    </div>
  );
}