// services/rowboat/ROWBOATWebSocket.ts

export class ROWBOATWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<Function>> = new Map();

  constructor(private baseUrl: string = '') {
    this.baseUrl = baseUrl || process.env.NEXT_PUBLIC_API_URL || 'ws://localhost:9000';
  }

  connect(workflowId: string): void {
    const wsUrl = `${this.baseUrl.replace('http', 'ws')}/rowboat/ws/${workflowId}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('ROWBOAT WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected', {});
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected', {});
        this.attemptReconnect(workflowId);
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.emit('error', error);
    }
  }

  private handleMessage(data: any): void {
    const { type, payload } = data;
    
    switch (type) {
      case 'agent_message':
        this.emit('agent_message', payload);
        break;
      case 'agent_status':
        this.emit('agent_status', payload);
        break;
      case 'execution_complete':
        this.emit('execution_complete', payload);
        break;
      case 'error':
        this.emit('error', payload);
        break;
      default:
        this.emit(type, payload);
    }
  }

  private attemptReconnect(workflowId: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect(workflowId);
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }

  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)?.add(callback);
  }

  off(event: string, callback: Function): void {
    this.listeners.get(event)?.delete(callback);
  }

  removeAllListeners(): void {
    this.listeners.clear();
  }

  private emit(event: string, data: any): void {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}