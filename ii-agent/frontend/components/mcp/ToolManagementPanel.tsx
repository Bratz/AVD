// components/mcp/ToolManagementPanel.tsx
import React, { useState } from 'react';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Search, RefreshCw, Save, X } from 'lucide-react';
import type { MCPServer, MCPTool } from '@/types/mcp.types';

interface ToolManagementPanelProps {
  server: MCPServer | null;
  onClose: () => void;
  selectedTools: Set<string>;
  onToolSelectionChange: (toolId: string, selected: boolean) => void;
  onSaveTools: () => void;
  onSyncTools?: () => void;
  hasChanges: boolean;
  isSaving: boolean;
  isSyncing: boolean;
}

export function ToolManagementPanel({
  server,
  onClose,
  selectedTools,
  onToolSelectionChange,
  onSaveTools,
  onSyncTools,
  hasChanges,
  isSaving,
  isSyncing,
}: ToolManagementPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');

  if (!server) return null;

  const availableTools = server.availableTools || [];
  const filteredTools = availableTools.filter(tool =>
    tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectAll = () => {
    filteredTools.forEach(tool => {
      onToolSelectionChange(tool.id, true);
    });
  };

  const handleDeselectAll = () => {
    filteredTools.forEach(tool => {
      onToolSelectionChange(tool.id, false);
    });
  };

  return (
    <Sheet open={!!server} onOpenChange={() => onClose()}>
      <SheetContent className="w-full sm:max-w-lg">
        <SheetHeader>
          <SheetTitle>Manage Tools - {server.name}</SheetTitle>
        </SheetHeader>

        <div className="mt-6 space-y-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              type="text"
              placeholder="Search tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Selection Controls */}
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              {selectedTools.size} of {availableTools.length} tools selected
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={handleSelectAll}
              >
                Select All
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={handleDeselectAll}
              >
                Deselect All
              </Button>
            </div>
          </div>

          {/* Tool List */}
          <ScrollArea className="h-[400px] border rounded-lg p-4">
            <div className="space-y-3">
              {filteredTools.length === 0 ? (
                <p className="text-center text-gray-500 py-8">No tools found</p>
              ) : (
                filteredTools.map((tool) => (
                  <ToolItem
                    key={tool.id}
                    tool={tool}
                    isSelected={selectedTools.has(tool.id)}
                    onToggle={(selected) => onToolSelectionChange(tool.id, selected)}
                  />
                ))
              )}
            </div>
          </ScrollArea>

          {/* Actions */}
          <div className="flex justify-between pt-4 border-t">
            <div className="flex gap-2">
              {onSyncTools && (
                <Button
                  variant="outline"
                  onClick={onSyncTools}
                  disabled={isSyncing}
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${isSyncing ? 'animate-spin' : ''}`} />
                  Sync Tools
                </Button>
              )}
            </div>
            
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={onClose}
              >
                Cancel
              </Button>
              <Button
                onClick={onSaveTools}
                disabled={!hasChanges || isSaving}
              >
                <Save className="w-4 h-4 mr-2" />
                {isSaving ? 'Saving...' : 'Save Changes'}
              </Button>
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

interface ToolItemProps {
  tool: MCPTool;
  isSelected: boolean;
  onToggle: (selected: boolean) => void;
}

function ToolItem({ tool, isSelected, onToggle }: ToolItemProps) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
      <Checkbox
        checked={isSelected}
        onCheckedChange={onToggle}
        className="mt-0.5"
      />
      <div className="flex-1">
        <h4 className="text-sm font-medium">{tool.name}</h4>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          {tool.description}
        </p>
        {tool.parameters && (
          <div className="mt-2 flex flex-wrap gap-1">
            {Object.keys(tool.parameters.properties).map((param) => (
              <Badge key={param} variant="secondary" className="text-xs">
                {param}
                {tool.parameters?.required?.includes(param) && '*'}
              </Badge>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}