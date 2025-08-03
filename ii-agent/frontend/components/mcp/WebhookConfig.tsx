// components/mcp/WebhookConfig.tsx
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { Copy, RefreshCw, Save, TestTube } from 'lucide-react';

interface WebhookConfigProps {
  projectId: string;
}

export function WebhookConfig({ projectId }: WebhookConfigProps) {
  const { toast } = useToast();
  const [webhookUrl, setWebhookUrl] = useState('');
  const [originalUrl, setOriginalUrl] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  useEffect(() => {
    fetchWebhookConfig();
  }, [projectId]);

  const fetchWebhookConfig = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/projects/${projectId}/webhook`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch webhook configuration');
      }
      
      const data = await response.json();
      setWebhookUrl(data.webhookUrl || '');
      setOriginalUrl(data.webhookUrl || '');
    } catch (error) {
      console.error('Error fetching webhook config:', error);
      toast({
        title: 'Error',
        description: 'Failed to load webhook configuration.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const response = await fetch(`/api/projects/${projectId}/webhook`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ webhookUrl }),
      });

      if (!response.ok) {
        throw new Error('Failed to save webhook configuration');
      }

      setOriginalUrl(webhookUrl);
      toast({
        title: 'Webhook saved',
        description: 'Webhook configuration has been updated.',
      });
    } catch (error) {
      console.error('Error saving webhook:', error);
      toast({
        title: 'Error',
        description: 'Failed to save webhook configuration.',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleTest = async () => {
    if (!webhookUrl) {
      toast({
        title: 'No webhook URL',
        description: 'Please enter a webhook URL to test.',
        variant: 'destructive',
      });
      return;
    }

    setIsTesting(true);
    try {
      const response = await fetch(`/api/projects/${projectId}/webhook/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ webhookUrl }),
      });

      if (!response.ok) {
        throw new Error('Webhook test failed');
      }

      const result = await response.json();
      
      toast({
        title: 'Test successful',
        description: `Webhook responded with status ${result.status}.`,
      });
    } catch (error) {
      console.error('Error testing webhook:', error);
      toast({
        title: 'Test failed',
        description: 'Failed to reach webhook endpoint.',
        variant: 'destructive',
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleCopyExample = () => {
    const example = `{
  "tool": "custom_tool",
  "parameters": {
    "action": "example",
    "data": {}
  },
  "context": {
    "projectId": "${projectId}",
    "agentId": "agent_123",
    "sessionId": "session_456"
  }
}`;
    navigator.clipboard.writeText(example);
    toast({
      title: 'Copied',
      description: 'Example payload copied to clipboard.',
    });
  };

  const hasChanges = webhookUrl !== originalUrl;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
        <span className="ml-2 text-sm text-gray-600">Loading configuration...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Webhook Configuration</CardTitle>
          <CardDescription>
            Configure a webhook URL to handle custom tool executions from your agents.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Webhook URL
            </label>
            <Input
              type="url"
              value={webhookUrl}
              onChange={(e) => setWebhookUrl(e.target.value)}
              placeholder="https://your-server.com/webhook"
              className="font-mono"
            />
            <p className="text-xs text-gray-500 mt-1">
              This URL will receive POST requests when agents execute custom tools.
            </p>
          </div>

          <div className="flex gap-2">
            <Button
              onClick={handleSave}
              disabled={!hasChanges || isSaving}
            >
              <Save className="w-4 h-4 mr-2" />
              {isSaving ? 'Saving...' : 'Save'}
            </Button>
            
            <Button
              variant="outline"
              onClick={handleTest}
              disabled={!webhookUrl || isTesting}
            >
              <TestTube className="w-4 h-4 mr-2" />
              {isTesting ? 'Testing...' : 'Test Connection'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Integration Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">Request Format</h4>
            <p className="text-sm text-gray-600 mb-2">
              Your webhook will receive POST requests with the following structure:
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded-lg font-mono text-sm">
              <pre className="whitespace-pre-wrap">
{`{
  "tool": "tool_name",
  "parameters": { ... },
  "context": {
    "projectId": "...",
    "agentId": "...",
    "sessionId": "..."
  }
}`}
              </pre>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopyExample}
              className="mt-2"
            >
              <Copy className="w-4 h-4 mr-2" />
              Copy Example
            </Button>
          </div>

          <div>
            <h4 className="font-medium mb-2">Response Format</h4>
            <p className="text-sm text-gray-600 mb-2">
              Your webhook should respond with:
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded-lg font-mono text-sm">
              <pre className="whitespace-pre-wrap">
{`{
  "success": true,
  "result": { ... },
  "error": null
}`}
              </pre>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Security</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline">HTTPS Required</Badge>
                <span className="text-sm text-gray-600">
                  Webhook URL must use HTTPS in production
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline">Authentication</Badge>
                <span className="text-sm text-gray-600">
                  Requests include a signature header for verification
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}