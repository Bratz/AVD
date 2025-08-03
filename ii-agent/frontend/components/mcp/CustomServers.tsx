// components/mcp/CustomServers.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Info, Plus, Search, Trash2 } from 'lucide-react';
import { ServerCard } from './ServerCard';
import { ToolManagementPanel } from './ToolManagementPanel';
import { Modal } from '@/components/ui/modal';
import { useToast } from '@/hooks/use-toast';
import type { MCPServer } from '@/types/mcp.types';

interface CustomServersProps {
  projectId: string;
}

export function CustomServers({ projectId }: CustomServersProps) {
  const { toast } = useToast();
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [selectedServer, setSelectedServer] = useState<MCPServer | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [togglingServers, setTogglingServers] = useState<Set<string>>(new Set());
  const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
  const [hasToolChanges, setHasToolChanges] = useState(false);
  const [savingTools, setSavingTools] = useState(false);
  const [syncingServers, setSyncingServers] = useState<Set<string>>(new Set());
  const [showAddServer, setShowAddServer] = useState(false);
  const [newServerName, setNewServerName] = useState('');
  const [newServerUrl, setNewServerUrl] = useState('');
  const [addingServer, setAddingServer] = useState(false);

  const fetchServers = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/custom`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch custom servers');
      }
      
      const data = await response.json();
      setServers(data.servers || []);
    } catch (err) {
      console.error('Error fetching servers:', err);
      toast({
        title: 'Error',
        description: 'Failed to load custom servers.',
        variant: 'destructive',
      });
      setServers([]);
    } finally {
      setLoading(false);
    }
  }, [projectId, toast]);

  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

  useEffect(() => {
    if (selectedServer) {
      setSelectedTools(new Set(selectedServer.tools.map(tool => tool.id)));
      setHasToolChanges(false);
    }
  }, [selectedServer]);

  const handleToggleServer = async (server: MCPServer) => {
    const serverKey = server.name;
    setTogglingServers(prev => new Set(prev).add(serverKey));

    try {
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !server.isActive }),
      });

      if (!response.ok) {
        throw new Error('Failed to toggle server');
      }

      setServers(prev => prev.map(s => 
        s.id === server.id ? { ...s, isActive: !s.isActive } : s
      ));

      toast({
        title: server.isActive ? 'Server disabled' : 'Server enabled',
        description: `${server.name} has been ${server.isActive ? 'disabled' : 'enabled'}.`,
      });
    } catch (err) {
      console.error('Toggle failed:', err);
      toast({
        title: 'Error',
        description: 'Failed to toggle server. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setTogglingServers(prev => {
        const next = new Set(prev);
        next.delete(serverKey);
        return next;
      });
    }
  };

  const handleSyncServer = async (server: MCPServer) => {
    if (!server.isActive) return;

    setSyncingServers(prev => new Set(prev).add(server.name));

    try {
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}/sync`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to sync server');
      }

      await fetchServers();
      
      toast({
        title: 'Tools synced',
        description: `${server.name} tools have been updated.`,
      });
    } catch (error) {
      console.error('Sync error:', error);
      toast({
        title: 'Sync failed',
        description: 'Failed to sync server tools. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setSyncingServers(prev => {
        const next = new Set(prev);
        next.delete(server.name);
        return next;
      });
    }
  };

  const handleAddServer = async () => {
    if (!newServerName || !newServerUrl) return;

    setAddingServer(true);
    try {
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newServerName,
          serverUrl: newServerUrl,
          description: `Custom MCP server at ${newServerUrl}`,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to add server');
      }

      const newServer = await response.json();
      setServers(prev => [...prev, newServer]);
      setShowAddServer(false);
      setNewServerName('');
      setNewServerUrl('');

      toast({
        title: 'Server added',
        description: 'Custom server has been added successfully.',
      });

      // Sync tools for the new server
      await handleSyncServer(newServer);
    } catch (err) {
      console.error('Error adding server:', err);
      toast({
        title: 'Error',
        description: 'Failed to add server. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setAddingServer(false);
    }
  };

  const handleRemoveServer = async (server: MCPServer) => {
    const shouldRemove = window.confirm(
      `Are you sure you want to delete "${server.name}"? This action cannot be undone.`
    );

    if (!shouldRemove) return;

    try {
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to remove server');
      }

      setServers(prev => prev.filter(s => s.id !== server.id));
      
      if (selectedServer?.id === server.id) {
        setSelectedServer(null);
      }

      toast({
        title: 'Server removed',
        description: 'Custom server has been removed.',
      });
    } catch (err) {
      console.error('Error removing server:', err);
      toast({
        title: 'Error',
        description: 'Failed to remove server. Please try again.',
        variant: 'destructive',
      });
    }
  };

  const handleSaveToolSelection = async () => {
    if (!selectedServer) return;
    
    setSavingTools(true);
    try {
      const toolIds = Array.from(selectedTools);
      
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${selectedServer.id}/tools`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabledTools: toolIds }),
      });

      if (!response.ok) {
        throw new Error('Failed to save tool selection');
      }

      setServers(prev => prev.map(s => 
        s.id === selectedServer.id 
          ? {
              ...s,
              tools: (s.availableTools || []).filter(tool => selectedTools.has(tool.id))
            }
          : s
      ));

      setSelectedServer(prev => {
        if (!prev) return null;
        return {
          ...prev,
          tools: (prev.availableTools || []).filter(tool => selectedTools.has(tool.id))
        };
      });
      
      setHasToolChanges(false);
      
      toast({
        title: 'Tools saved',
        description: 'Tool selection has been updated.',
      });
    } catch (error) {
      console.error('Error saving tool selection:', error);
      toast({
        title: 'Save failed',
        description: 'Failed to save tool selection. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setSavingTools(false);
    }
  };

  const filteredServers = servers.filter(server => {
    const searchLower = searchQuery.toLowerCase();
    return (
      server.name.toLowerCase().includes(searchLower) ||
      server.description.toLowerCase().includes(searchLower) ||
      server.tools.some(tool => 
        tool.name.toLowerCase().includes(searchLower) ||
        tool.description.toLowerCase().includes(searchLower)
      )
    );
  });

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg p-4">
        <div className="flex gap-3">
          <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
          <p className="text-sm text-blue-700 dark:text-blue-300">
            Add your own MCP servers here. These servers will be available to agents in the Build view once toggled ON.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <Button
          size="sm"
          onClick={() => setShowAddServer(true)}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Server
        </Button>
        
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Search servers or tools..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      {/* Add Server Modal */}
      <Modal
        isOpen={showAddServer}
        onClose={() => {
          setShowAddServer(false);
          setNewServerName('');
          setNewServerUrl('');
        }}
        title="Add Custom MCP Server"
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Server Name
            </label>
            <Input
              type="text"
              value={newServerName}
              onChange={(e) => setNewServerName(e.target.value)}
              placeholder="e.g., My Custom Server"
              disabled={addingServer}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">
              Server URL
            </label>
            <Input
              type="text"
              value={newServerUrl}
              onChange={(e) => setNewServerUrl(e.target.value)}
              placeholder="e.g., http://localhost:3000"
              disabled={addingServer}
            />
          </div>
          
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="outline"
              onClick={() => {
                setShowAddServer(false);
                setNewServerName('');
                setNewServerUrl('');
              }}
              disabled={addingServer}
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddServer}
              disabled={!newServerName || !newServerUrl || addingServer}
            >
              {addingServer ? 'Adding...' : 'Add Server'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Server Grid */}
      {loading ? (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-800 dark:border-gray-200 mx-auto" />
          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">Loading servers...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredServers.map((server) => (
            <ServerCard
              key={server.id}
              server={server}
              onToggle={() => handleToggleServer(server)}
              onManageTools={() => setSelectedServer(server)}
              onSync={() => handleSyncServer(server)}
              onRemove={() => handleRemoveServer(server)}
              isToggling={togglingServers.has(server.name)}
              isSyncing={syncingServers.has(server.name)}
              showAuth={false}
            />
          ))}
        </div>
      )}

      {/* Tool Management Panel */}
      <ToolManagementPanel
        server={selectedServer}
        onClose={() => {
          setSelectedServer(null);
          setSelectedTools(new Set());
          setHasToolChanges(false);
        }}
        selectedTools={selectedTools}
        onToolSelectionChange={(toolId, selected) => {
          setSelectedTools(prev => {
            const next = new Set(prev);
            if (selected) {
              next.add(toolId);
            } else {
              next.delete(toolId);
            }
            setHasToolChanges(true);
            return next;
          });
        }}
        onSaveTools={handleSaveToolSelection}
        onSyncTools={selectedServer ? () => handleSyncServer(selectedServer) : undefined}
        hasChanges={hasToolChanges}
        isSaving={savingTools}
        isSyncing={selectedServer ? syncingServers.has(selectedServer.name) : false}
      />
    </div>
  );
}