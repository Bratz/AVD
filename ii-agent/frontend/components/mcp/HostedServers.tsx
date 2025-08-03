// components/mcp/HostedServers.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Info, RefreshCw, Search, AlertTriangle } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';
import { ServerCard } from './ServerCard';
import { ToolManagementPanel } from './ToolManagementPanel';
import { useToast } from '@/hooks/use-toast';
import type { MCPServer, MCPTool, EnableServerResponse } from '@/types/mcp.types';

interface HostedServersProps {
  projectId: string;
  onSwitchTab?: (tab: string) => void;
}

export function HostedServers({ projectId, onSwitchTab }: HostedServersProps) {
  const { toast } = useToast();
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [selectedServer, setSelectedServer] = useState<MCPServer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showOnlyEnabled, setShowOnlyEnabled] = useState(false);
  const [showOnlyReady, setShowOnlyReady] = useState(false);
  const [toggleError, setToggleError] = useState<{serverId: string; message: string} | null>(null);
  const [togglingServers, setTogglingServers] = useState<Set<string>>(new Set());
  const [serverOperations, setServerOperations] = useState<Map<string, 'setup' | 'delete' | 'checking-auth'>>(new Map());
  const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
  const [hasToolChanges, setHasToolChanges] = useState(false);
  const [savingTools, setSavingTools] = useState(false);
  const [syncingServers, setSyncingServers] = useState<Set<string>>(new Set());

  const fetchServers = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/hosted`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch hosted servers');
      }
      
      const data = await response.json();
      setServers(data.servers || []);
      setError(null);
    } catch (err) {
      setError('Unable to load hosted tools. Please check your connection and try again.');
      console.error('Error fetching servers:', err);
      setServers([]);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

  useEffect(() => {
    if (selectedServer) {
      setSelectedTools(new Set(selectedServer.tools.map(t => t.id)));
      setHasToolChanges(false);
    }
  }, [selectedServer]);

  const handleToggleServer = async (server: MCPServer) => {
    const serverKey = server.name;
    const newState = !server.isActive;

    setTogglingServers(prev => new Set(prev).add(serverKey));
    setToggleError(null);
    
    setServerOperations(prev => {
      const next = new Map(prev);
      next.set(serverKey, newState ? 'setup' : 'delete');
      return next;
    });

    try {
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: newState }),
      });

      const result: EnableServerResponse = await response.json();
      
      // If billing error, just show a toast message instead
      if (result.billingError) {
        toast({
          title: 'Upgrade Required',
          description: result.billingError,
          variant: 'destructive',
        });
        return;
      }

      if (!response.ok) {
        throw new Error(result.error || 'Failed to toggle server');
      }

      // Update local state
      setServers(prev => prev.map(s => 
        s.name === serverKey 
          ? { ...s, isActive: newState, instanceId: result.instanceId }
          : s
      ));

      toast({
        title: newState ? 'Server enabled' : 'Server disabled',
        description: `${server.name} has been ${newState ? 'enabled' : 'disabled'}.`,
      });

      // Refresh servers to get updated state
      if (newState) {
        await fetchServers();
      }
    } catch (err) {
      console.error('Toggle failed:', err);
      setToggleError({
        serverId: serverKey,
        message: "We're having trouble setting up this server. Please try again.",
      });
    } finally {
      setTogglingServers(prev => {
        const next = new Set(prev);
        next.delete(serverKey);
        return next;
      });
      setServerOperations(prev => {
        const next = new Map(prev);
        next.delete(serverKey);
        return next;
      });
    }
  };

  const handleAuthenticate = async (server: MCPServer) => {
    try {
      if (!server.instanceId) {
        throw new Error('Server instance ID not found');
      }

      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}/auth-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instanceId: server.instanceId }),
      });

      if (!response.ok) {
        throw new Error('Failed to get authentication URL');
      }

      const { authUrl } = await response.json();
      
      const authWindow = window.open(
        authUrl,
        '_blank',
        'width=600,height=700'
      );

      if (authWindow) {
        // Poll for auth completion
        const checkInterval = setInterval(async () => {
          if (authWindow.closed) {
            clearInterval(checkInterval);
            
            setServerOperations(prev => {
              const next = new Map(prev);
              next.set(server.name, 'checking-auth');
              return next;
            });
            
            try {
              // Update server auth status
              await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}/auth-callback`, {
                method: 'POST',
              });
              
              // Refresh servers
              await fetchServers();
              
              toast({
                title: 'Authentication successful',
                description: `${server.name} has been authenticated.`,
              });
            } finally {
              setServerOperations(prev => {
                const next = new Map(prev);
                next.delete(server.name);
                return next;
              });
            }
          }
        }, 500);
      } else {
        toast({
          title: 'Authentication failed',
          description: 'Please check your popup blocker settings.',
          variant: 'destructive',
        });
      }
    } catch (error) {
      console.error('Auth error:', error);
      toast({
        title: 'Authentication failed',
        description: 'Failed to setup authentication. Please try again.',
        variant: 'destructive',
      });
    }
  };

  const handleSyncServer = async (server: MCPServer) => {
    if (!server.isActive || (server.authNeeded && !server.isAuthenticated)) return;

    setSyncingServers(prev => new Set(prev).add(server.name));

    try {
      const response = await fetch(`/api/projects/${projectId}/mcp/servers/${server.id}/sync`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to sync server tools');
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

  const handleSaveToolSelection = async () => {
    if (!selectedServer || !projectId) return;
    
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

      // Update local state
      setServers(prevServers => 
        prevServers.map(s => 
          s.id === selectedServer.id 
            ? {
                ...s,
                tools: (s.availableTools || []).filter(tool => selectedTools.has(tool.id))
              }
            : s
        )
      );

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
    const matchesSearch = 
      server.name.toLowerCase().includes(searchLower) ||
      server.description.toLowerCase().includes(searchLower) ||
      server.tools.some(tool => 
        tool.name.toLowerCase().includes(searchLower) ||
        tool.description.toLowerCase().includes(searchLower)
      );

    const matchesEnabled = !showOnlyEnabled || server.isActive;
    const isReady = server.isActive && (!server.authNeeded || server.isAuthenticated);
    const matchesReady = !showOnlyReady || isReady;

    return matchesSearch && matchesEnabled && matchesReady;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
        <span className="ml-2 text-sm text-gray-600">Loading tools...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <AlertTriangle className="w-8 h-8 text-red-500 mx-auto mb-4" />
        <p className="text-red-600 mb-4">{error}</p>
        <div className="flex gap-4 justify-center">
          <Button variant="outline" onClick={fetchServers}>
            Try Again
          </Button>
          <Button variant="outline" onClick={() => onSwitchTab?.('custom')}>
            Use Custom Servers
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg p-4">
        <div className="flex gap-3">
          <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
          <p className="text-sm text-blue-700 dark:text-blue-300">
            To make hosted MCP tools available to agents in the Build view, first toggle the servers ON here. 
            Some tools may require authentication after enabling.
          </p>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex items-center gap-4">
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
        
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={showOnlyEnabled}
              onCheckedChange={(checked) => {
                // Handle indeterminate state properly
                if (checked === 'indeterminate') {
                  setShowOnlyEnabled(false);
                } else {
                  setShowOnlyEnabled(checked);
                }
              }}
            />
            Enabled Only
          </label>
          
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={showOnlyReady}
              onCheckedChange={(checked) => {
                // Handle indeterminate state properly
                if (checked === 'indeterminate') {
                  setShowOnlyReady(false);
                } else {
                  setShowOnlyReady(checked);
                }
              }}
            />
            Ready to Use
          </label>
        </div>
        
        <Button
          size="sm"
          variant="outline"
          onClick={fetchServers}
          disabled={loading}
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span className="ml-2">Refresh</span>
        </Button>
      </div>

      {/* Server Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredServers.map((server) => (
          <ServerCard
            key={server.id}
            server={server}
            onToggle={() => handleToggleServer(server)}
            onManageTools={() => setSelectedServer(server)}
            onSync={() => handleSyncServer(server)}
            onAuth={() => handleAuthenticate(server)}
            isToggling={togglingServers.has(server.name)}
            isSyncing={syncingServers.has(server.name)}
            operation={serverOperations.get(server.name)}
            error={toggleError?.serverId === server.name ? toggleError : undefined}
          />
        ))}
      </div>

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