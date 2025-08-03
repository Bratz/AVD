// components/mcp/MCPToolsPage.tsx
import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PageHeader } from '@/components/ui/page-header';
import { HostedServers } from './HostedServers';
import { CustomServers } from './CustomServers';
import { WebhookConfig } from './WebhookConfig';
import { Server, Globe, Webhook } from 'lucide-react';

interface MCPToolsPageProps {
  projectId: string;
}

export function MCPToolsPage({ projectId }: MCPToolsPageProps) {
  const [activeTab, setActiveTab] = useState<'hosted' | 'custom' | 'webhook'>('hosted');

  // Create a wrapper function that accepts string
  const handleSwitchTab = (tab: string) => {
    setActiveTab(tab as 'hosted' | 'custom' | 'webhook');
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Tools & Integrations"
        description="Configure MCP servers and external tool integrations for your agents"
      />
      
      <div className="flex-1 p-6">
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)}>
          <TabsList className="grid w-full max-w-[600px] grid-cols-3">
            <TabsTrigger value="hosted" className="flex items-center gap-2">
              <Server className="w-4 h-4" />
              Hosted MCP Servers
            </TabsTrigger>
            <TabsTrigger value="custom" className="flex items-center gap-2">
              <Globe className="w-4 h-4" />
              Custom Servers
            </TabsTrigger>
            <TabsTrigger value="webhook" className="flex items-center gap-2">
              <Webhook className="w-4 h-4" />
              Webhooks
            </TabsTrigger>
          </TabsList>

          <TabsContent value="hosted" className="mt-6">
            <HostedServers 
              projectId={projectId}
              onSwitchTab={handleSwitchTab}
            />
          </TabsContent>

          <TabsContent value="custom" className="mt-6">
            <CustomServers projectId={projectId} />
          </TabsContent>

          <TabsContent value="webhook" className="mt-6">
            <WebhookConfig projectId={projectId} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}