// use-sse.tsx
import { useEffect, useRef, useState } from 'react';

interface UseSSEOptions {
  onMessage?: (data: any) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
}

export function useSSE(url: string | null, options: UseSSEOptions = {}) {
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = () => {
    if (!url) return;

    // Clean up existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setConnectionState('connecting');
    console.log('Connecting to SSE:', url);
    
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setConnectionState('connected');
      reconnectAttemptsRef.current = 0;
      options.onOpen?.();
      console.log('SSE connected');
    };

    eventSource.onmessage = (event: MessageEvent) => {
      try {
        // Parse the data from event.data, not event.content
        const data = JSON.parse(event.data);
        console.log('SSE message received:', data.type);
        // Pass the parsed data object to onMessage
        options.onMessage?.(data);
      } catch (error) {
        console.error('Error parsing SSE message:', error);
      }
    };

    eventSource.onerror = (error: Event) => {
      console.error('SSE error:', error);
      setConnectionState('disconnected');
      eventSourceRef.current?.close();
      options.onError?.(error);
      
      // Reconnect with exponential backoff
      const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
      reconnectAttemptsRef.current++;
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`Attempting SSE reconnect (attempt ${reconnectAttemptsRef.current})...`);
        connect();
      }, delay);
    };
  };

  useEffect(() => {
    connect();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [url, options.onMessage, options.onError, options.onOpen]); // Add dependencies

  const close = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    setConnectionState('disconnected');
  };

  return {
    connectionState,
    close,
    reconnect: connect
  };
}