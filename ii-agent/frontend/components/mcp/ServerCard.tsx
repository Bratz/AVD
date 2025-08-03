// components/mcp/ServerCard.tsx
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { 
  Server, 
  Lock, 
  Unlock, 
  RefreshCw, 
  Settings, 
  Trash2,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';
import type { MCPServer } from '@/types/mcp.types';

interface ServerCardProps {
  server: MCPServer;
  onToggle: () => void;
  onManageTools?: () => void;
  onSync?: () => void;
  onAuth?: () => void;
  onRemove?: () => void;
  isToggling?: boolean;
  isSyncing?: boolean;
  operation?: 'setup' | 'delete' | 'checking-auth';
  error?: { message: string };
  showAuth?: boolean;
}

export function ServerCard({
  server,
  onToggle,
  onManageTools,
  onSync,
  onAuth,
  onRemove,
  isToggling = false,
  isSyncing = false,
  operation,
  error,
  showAuth = true,
}: ServerCardProps) {
  const isReady = server.isActive && (!server.authNeeded || server.isAuthenticated);
  const canManageTools = isReady && server.availableTools && server.availableTools.length > 0;

  return (
    <Card className="relative overflow-hidden">
      {operation && (
        <div className="absolute inset-0 bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm z-10 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
            <p className="text-sm">
              {operation === 'setup' ? 'Setting up...' :
               operation === 'delete' ? 'Removing...' :
               'Checking authentication...'}
            </p>
          </div>
        </div>
      )}

      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="flex items-center gap-2">
              <Server className="w-4 h-4" />
              {server.name}
            </CardTitle>
            <CardDescription className="mt-1">
              {server.description}
            </CardDescription>
          </div>
          
          <Switch
            checked={server.isActive}
            onCheckedChange={onToggle}
            disabled={isToggling}
          />
        </div>

        {error && (
          <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm text-red-600 dark:text-red-400">
            {error.message}
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Status Badges */}
        <div className="flex flex-wrap gap-2">
          {server.isActive && (
            <Badge variant={isReady ? 'default' : 'secondary'}>
              {isReady ? (
                <>
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Ready
                </>
              ) : (
                <>
                  <AlertCircle className="w-3 h-3 mr-1" />
                  Setup Required
                </>
              )}
            </Badge>
          )}
          
          {server.authNeeded && (
            <Badge variant={server.isAuthenticated ? 'default' : 'outline'}>
              {server.isAuthenticated ? (
                <>
                  <Unlock className="w-3 h-3 mr-1" />
                  Authenticated
                </>
              ) : (
                <>
                  <Lock className="w-3 h-3 mr-1" />
                  Auth Required
                </>
              )}
            </Badge>
          )}
          
          {server.serverType === 'custom' && (
            <Badge variant="outline">Custom</Badge>
          )}
        </div>

        {/* Tool Count */}
        {server.isActive && (
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {server.tools.length} of {server.availableTools?.length || 0} tools enabled
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2">
          {server.isActive && server.authNeeded && !server.isAuthenticated && showAuth && (
            <Button
              size="sm"
              variant="outline"
              onClick={onAuth}
              disabled={isToggling}
            >
              <Lock className="w-4 h-4 mr-2" />
              Authenticate
            </Button>
          )}
          
          {canManageTools && onManageTools && (
            <Button
              size="sm"
              variant="outline"
              onClick={onManageTools}
            >
              <Settings className="w-4 h-4 mr-2" />
              Manage Tools
            </Button>
          )}
          
          {isReady && onSync && (
            <Button
              size="sm"
              variant="outline"
              onClick={onSync}
              disabled={isSyncing}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isSyncing ? 'animate-spin' : ''}`} />
              Sync
            </Button>
          )}
          
          {server.serverType === 'custom' && onRemove && (
            <Button
              size="sm"
              variant="outline"
              onClick={onRemove}
              className="text-red-600 hover:text-red-700"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}