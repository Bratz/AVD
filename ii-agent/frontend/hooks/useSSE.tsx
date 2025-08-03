// hooks/useSSE.ts
import { useEffect, useState, useRef } from 'react';

interface SSEOptions {
  onMessage?: (data: any) => void;
  onError?: (error: Error) => void;
  onOpen?: () => void;
  reconnectInterval?: number;
}

export function useSSE(url: string, options: SSEOptions = {}) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = () => {
    try {
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        console.log('SSE connected');
        setIsConnected(true);
        options.onOpen?.();
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          options.onMessage?.(data);
        } catch (error) {
          console.error('Failed to parse SSE message:', error);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        setIsConnected(false);
        options.onError?.(error as any);
        
        // Reconnect after interval
        if (options.reconnectInterval) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, options.reconnectInterval);
        }
      };
    } catch (error) {
      console.error('Failed to create EventSource:', error);
      options.onError?.(error as Error);
    }
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [url]);

  return {
    isConnected,
    lastMessage,
    eventSource: eventSourceRef.current,
  };
}