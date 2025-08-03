// components/workflow/AgentConfigPanel.tsx
import React, { useState } from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import type { WorkflowNode, WorkflowNodeData } from '@/types/workflow.types';

interface AgentConfigPanelProps {
  node: WorkflowNode;
  onUpdate: (data: Partial<WorkflowNodeData>) => void;
  onClose: () => void;
}

export function AgentConfigPanel({ node, onUpdate, onClose }: AgentConfigPanelProps) {
  const [data, setData] = useState<WorkflowNodeData>(node.data);
  const [newTool, setNewTool] = useState('');

  const handleChange = <K extends keyof WorkflowNodeData>(
    field: K,
    value: WorkflowNodeData[K]
  ) => {
    const updatedData = { ...data, [field]: value };
    setData(updatedData);
    onUpdate(updatedData);
  };

  const addTool = () => {
    if (newTool && !data.tools?.includes(newTool)) {
      const tools = [...(data.tools || []), newTool];
      handleChange('tools', tools);
      setNewTool('');
    }
  };

  const removeTool = (tool: string) => {
    const tools = data.tools?.filter((t) => t !== tool) || [];
    handleChange('tools', tools);
  };

  return (
    <div className="w-96 bg-white border-l shadow-lg h-full overflow-y-auto">
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="text-lg font-semibold">Agent Configuration</h2>
        <Button onClick={onClose} variant="ghost" size="sm">
          <X className="w-4 h-4" />
        </Button>
      </div>

      <div className="p-4 space-y-4">
        <div>
          <Label htmlFor="label">Agent Name</Label>
          <Input
            id="label"
            value={data.label || ''}
            onChange={(e) => handleChange('label', e.target.value)}
            placeholder="Enter agent name"
          />
        </div>

        <div>
          <Label htmlFor="role">Role</Label>
          <Select
            value={data.role || 'generic'}
            onValueChange={(value) => handleChange('role', value)}
          >
            <SelectTrigger id="role">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="coordinator">Coordinator</SelectItem>
              <SelectItem value="researcher">Researcher</SelectItem>
              <SelectItem value="analyzer">Analyzer</SelectItem>
              <SelectItem value="writer">Writer</SelectItem>
              <SelectItem value="specialist">Specialist</SelectItem>
              <SelectItem value="reviewer">Reviewer</SelectItem>
              <SelectItem value="generic">Generic</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="instructions">Instructions</Label>
          <Textarea
            id="instructions"
            value={data.instructions || ''}
            onChange={(e) => handleChange('instructions', e.target.value)}
            placeholder="Enter agent instructions"
            rows={4}
          />
        </div>

        <div>
          <Label htmlFor="model">Model</Label>
          <Select
            value={data.model || 'gpt-4'}
            onValueChange={(value) => handleChange('model', value)}
          >
            <SelectTrigger id="model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4">GPT-4</SelectItem>
              <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
              <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
              <SelectItem value="claude-3">Claude 3</SelectItem>
              <SelectItem value="claude-2">Claude 2</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label>Tools</Label>
          <div className="flex gap-2 mb-2">
            <Input
              value={newTool}
              onChange={(e) => setNewTool(e.target.value)}
              placeholder="Add a tool"
              onKeyPress={(e) => e.key === 'Enter' && addTool()}
            />
            <Button onClick={addTool} size="sm">Add</Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {data.tools?.map((tool) => (
              <Badge key={tool} variant="secondary">
                {tool}
                <button
                  onClick={() => removeTool(tool)}
                  className="ml-2 text-xs hover:text-red-500"
                >
                  ×
                </button>
              </Badge>
            ))}
          </div>
        </div>

        <div>
          <Label htmlFor="outputVisibility">Output Visibility</Label>
          <Select
            value={data.outputVisibility || 'user_facing'}
            onValueChange={(value) => handleChange('outputVisibility', value as 'user_facing' | 'internal')}
          >
            <SelectTrigger id="outputVisibility">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="user_facing">User Facing</SelectItem>
              <SelectItem value="internal">Internal</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="controlType">Control Type</Label>
          <Select
            value={data.controlType || 'retain'}
            onValueChange={(value) => handleChange('controlType', value as 'retain' | 'relinquish_to_parent' | 'start_agent')}
          >
            <SelectTrigger id="controlType">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="retain">Retain Control</SelectItem>
              <SelectItem value="relinquish_to_parent">Relinquish to Parent</SelectItem>
              <SelectItem value="start_agent">Start Specific Agent</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
}