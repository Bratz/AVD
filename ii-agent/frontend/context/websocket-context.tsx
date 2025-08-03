// context/websocket-context.tsx
"use client";

import React, { createContext, useContext, useCallback, useRef } from "react";
import { useAppContext } from "./app-context";
import { generateUniqueId } from "@/utils/id-generator";

interface WebSocketContextValue {
  socket: WebSocket | null;
  sendMessage: (payload: { type: string; content: any }) => boolean;
  isConnected: boolean;
  registerCopilotHandler: (handler: CopilotEventHandler) => () => void;
}

type CopilotEventHandler = (event: {
  id: string;
  type: string;
  content: Record<string, unknown>;
}) => void;

const WebSocketContext = createContext<WebSocketContextValue | undefined>(undefined);

export function useWebSocketContext() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocketContext must be used within WebSocketProvider");
  }
  return context;
}

interface WebSocketProviderProps {
  children: React.ReactNode;
  socket: WebSocket | null;
  sendMessage: (payload: { type: string; content: any }) => boolean;
  handleEvent: (data: any, workspacePath?: string) => void;
}

export function WebSocketProvider({ 
  children, 
  socket, 
  sendMessage,
  handleEvent 
}: WebSocketProviderProps) {
  const { state } = useAppContext();
  const copilotHandlers = useRef<Set<CopilotEventHandler>>(new Set());
  const originalHandleEvent = useRef(handleEvent);

  const isConnected = state.wsConnectionState === "connected" && socket?.readyState === WebSocket.OPEN;

  // Register a handler for CopilotKit events
  const registerCopilotHandler = useCallback((handler: CopilotEventHandler) => {
    copilotHandlers.current.add(handler);
    return () => {
      copilotHandlers.current.delete(handler);
    };
  }, []);

  // Wrapped sendMessage that can handle CopilotKit-specific messages
  const wrappedSendMessage = useCallback((payload: { type: string; content: any }) => {
    // Add ID if not present
    if (payload.content && !payload.content.id) {
      payload.content.id = generateUniqueId(payload.type);
    }

    // For CopilotKit messages, notify handlers when we get responses
    if (payload.type === "copilot_query" || payload.content?.source === "copilot") {
      // Store the request ID so we can match responses
      const requestId = payload.content.id;
      
      // We'll need to intercept responses in the handleEvent
      // This is handled by the enhanced handleEvent wrapper below
    }

    return sendMessage(payload);
  }, [sendMessage]);

  // Create an enhanced handleEvent that also notifies CopilotKit handlers
  const enhancedHandleEvent = useCallback((data: any, workspacePath?: string) => {
    // Call the original handler first
    originalHandleEvent.current(data, workspacePath);

    // If this is a response to a CopilotKit query, notify handlers
    if (data.content?.source === "copilot" || data.type === "copilot_response") {
      const eventData = {
        ...data,
        id: data.id || generateUniqueId('copilot-event')
      };

      copilotHandlers.current.forEach(handler => {
        try {
          handler(eventData);
        } catch (error) {
          console.error("[CopilotKit] Error in event handler:", error);
        }
      });
    }
  }, []);

  // Update the original handleEvent reference if it changes
  React.useEffect(() => {
    originalHandleEvent.current = handleEvent;
  }, [handleEvent]);

  const value: WebSocketContextValue = {
    socket,
    sendMessage: wrappedSendMessage,
    isConnected,
    registerCopilotHandler,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}