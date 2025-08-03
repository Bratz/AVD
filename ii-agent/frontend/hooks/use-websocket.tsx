// // hooks/use-websocket.tsx
// import { AgentEvent, WebSocketConnectionState,Message } from "@/typings/agent";
// import { useState, useEffect } from "react";
// import { toast } from "sonner";
// import { useAppContext } from "@/context/app-context";
// import { generateUniqueId } from '@/utils/id-generator';
// import { handleAGUIWebSocketMessage } from '../utils/ag-ui-websocket-handler';
// import { sendResponse } from '@/agentic-ui/ag-ui-client';

// interface WebSocketMessageContent {
//   [key: string]: unknown;
// }

// export function useWebSocket(
//   deviceId: string,
//   isReplayMode: boolean,
//   handleEvent: (
//     data: {
//       id: string;
//       type: AgentEvent;
//       content: Record<string, unknown>;
//     },
//     workspacePath?: string
//   ) => void
// ) {
//   const [socket, setSocket] = useState<WebSocket | null>(null);
//   const { dispatch } = useAppContext();

//   const connectWebSocket = () => {
//     dispatch({
//       type: "SET_WS_CONNECTION_STATE",
//       payload: WebSocketConnectionState.CONNECTING,
//     });

//     const token = document.cookie
//       .split('; ')
//       .find(row => row.startsWith('auth-token='))
//       ?.split('=')[1];
      
//     const params = new URLSearchParams({ 
//       device_id: deviceId,
//       ...(token && { auth_token: token })
//     });
    
//     const ws = new WebSocket(
//       `${process.env.NEXT_PUBLIC_API_URL}/ws?${params.toString()}`
//     );

//     ws.onopen = () => {
//       console.log("WebSocket connection established");
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.CONNECTED,
//       });
//       ws.send(
//         JSON.stringify({
//           type: "workspace_info",
//           content: {},
//         })
//       );
//     };

//     ws.onmessage = (event) => {
//       try {
//         const data = JSON.parse(event.data);
        
//         // Handle CopilotKit responses
//         if (data.type === "copilot_response") {
//           // Create a Message object matching your interface
//           const message: Message = {
//             id: data.id || generateUniqueId('copilot'),
//             role: "assistant",
//             content: data.content.text || data.content.message,
//             timestamp: Date.now(),
//             metadata: {
//               model_used: data.content.model || "gpt-4",
//               routing_info: { source: "copilot" }
//             }
//           };
          
//           dispatch({
//             type: "ADD_MESSAGE",
//             payload: message
//           });
//         } else if (data.type === "copilot_error") {
//           toast.error(`CopilotKit error: ${data.content.error}`);
//         } else {
//           // Handle regular ii-agent messages
//           handleEvent({ 
//             ...data, 
//             id: data.id || generateUniqueId(data.type) 
//           });
//         }
//       } catch (error) {
//         console.error("Error parsing WebSocket data:", error);
//       }
//     };

//     ws.onerror = (error) => {
//       console.log("WebSocket error:", error);
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.DISCONNECTED,
//       });
//       toast.error("WebSocket connection error");
//     };

//     ws.onclose = () => {
//       console.log("WebSocket connection closed");
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.DISCONNECTED,
//       });
//       setSocket(null);
//     };

//     setSocket(ws);
//   };

//   const sendMessage = (payload: {
//     type: string;
//     content: WebSocketMessageContent;
//   }) => {
//     if (!socket || socket.readyState !== WebSocket.OPEN) {
//       toast.error("WebSocket connection is not open. Please try again.");
//       return;
//     }
    
//     // Add unique ID to all messages
//     const messageWithId = {
//       ...payload,
//       id: generateUniqueId(payload.type),
//     };
    
//     socket.send(JSON.stringify(messageWithId));
//   };

//   useEffect(() => {
//     if (!isReplayMode) {
//       connectWebSocket();
//     }

//     return () => {
//       if (socket && socket.readyState === WebSocket.OPEN) {
//         socket.close();
//       }
//     };
//   }, [deviceId, isReplayMode]);

//   return { socket, sendMessage };
// }

// // hooks/use-websocket.tsx
// import { AgentEvent, WebSocketConnectionState, Message } from "@/typings/agent";
// import { useState, useEffect } from "react";
// import { toast } from "sonner";
// import { useAppContext } from "@/context/app-context";
// import { generateUniqueId } from '@/utils/id-generator';
// import { handleAGUIWebSocketMessage } from '../utils/ag-ui-websocket-handler';
// import { subscribe } from '@/agentic-ui/ag-ui-client';

// interface WebSocketMessageContent {
//   [key: string]: unknown;
// }

// export function useWebSocket(
//   deviceId: string,
//   isReplayMode: boolean,
//   handleEvent: (
//     data: {
//       id: string;
//       type: AgentEvent;
//       content: Record<string, unknown>;
//     },
//     workspacePath?: string
//   ) => void
// ) {
//   const [socket, setSocket] = useState<WebSocket | null>(null);
//   const { dispatch } = useAppContext();

//   const connectWebSocket = () => {
//     dispatch({
//       type: "SET_WS_CONNECTION_STATE",
//       payload: WebSocketConnectionState.CONNECTING,
//     });

//     const token = document.cookie
//       .split('; ')
//       .find(row => row.startsWith('auth-token='))
//       ?.split('=')[1];
      
//     const params = new URLSearchParams({ 
//       device_id: deviceId,
//       ...(token && { auth_token: token })
//     });
    
//     const ws = new WebSocket(
//       `${process.env.NEXT_PUBLIC_API_URL}/ws?${params.toString()}`
//     );

//     ws.onopen = () => {
//       console.log("WebSocket connection established");
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.CONNECTED,
//       });
//       ws.send(
//         JSON.stringify({
//           type: "workspace_info",
//           content: {},
//         })
//       );
//     };

//     ws.onmessage = (event) => {
//       try {
//         const data = JSON.parse(event.data);
        
//         // First, check if this is an AG-UI protocol message
//         const aguiEvent = handleAGUIWebSocketMessage(data);
//         if (aguiEvent) {
//           // Emit AG-UI event through the AG-UI client
//           // The AG-UI client will notify all subscribed components
//           window.dispatchEvent(new CustomEvent('ag-ui-event', { detail: aguiEvent }));
//           return;
//         }
        
//         // Handle regular agent messages
//         handleEvent({ 
//           ...data, 
//           id: data.id || generateUniqueId(data.type) 
//         });
//       } catch (error) {
//         console.error("Error parsing WebSocket data:", error);
//       }
//     };

//     ws.onerror = (error) => {
//       console.log("WebSocket error:", error);
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.DISCONNECTED,
//       });
//       toast.error("WebSocket connection error");
//     };

//     ws.onclose = () => {
//       console.log("WebSocket connection closed");
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.DISCONNECTED,
//       });
//       setSocket(null);
//     };

//     setSocket(ws);
//   };

//   const sendMessage = (payload: {
//     type: string;
//     content: WebSocketMessageContent;
//   }) => {
//     if (!socket || socket.readyState !== WebSocket.OPEN) {
//       toast.error("WebSocket connection is not open. Please try again.");
//       return;
//     }
    
//     // Add unique ID to all messages
//     const messageWithId = {
//       ...payload,
//       id: generateUniqueId(payload.type),
//     };
    
//     socket.send(JSON.stringify(messageWithId));
//   };

//   // Set up AG-UI event listener for outgoing messages
//   useEffect(() => {
//     const unsubscribe = subscribe((event) => {
//       if (event.type === 'agentResponse' || event.type === 'toolCall') {
//         // Send AG-UI events through WebSocket
//         sendMessage({
//           type: 'ag_ui_event',
//           content: event
//         });
//       }
//     });

//     return unsubscribe;
//   }, [socket]);

//   useEffect(() => {
//     if (!isReplayMode) {
//       connectWebSocket();
//     }

//     return () => {
//       if (socket && socket.readyState === WebSocket.OPEN) {
//         socket.close();
//       }
//     };
//   }, [deviceId, isReplayMode]);

//   return { socket, sendMessage };
// }

// hooks/use-websocket.tsx
// import { AgentEvent, WebSocketConnectionState, Message } from "@/typings/agent";
// import { useState, useEffect } from "react";
// import { toast } from "sonner";
// import { useAppContext } from "@/context/app-context";
// import { generateUniqueId } from '@/utils/id-generator';
// import { subscribe, getTool } from '@/agentic-ui/ag-ui-client';

// interface WebSocketMessageContent {
//   [key: string]: unknown;
// }

// export function useWebSocket(
//   deviceId: string,
//   isReplayMode: boolean,
//   handleEvent: (
//     data: {
//       id: string;
//       type: AgentEvent;
//       content: Record<string, unknown>;
//     },
//     workspacePath?: string
//   ) => void
// ) {
//   const [socket, setSocket] = useState<WebSocket | null>(null);
//   const { dispatch } = useAppContext();

//   const connectWebSocket = () => {
//     dispatch({
//       type: "SET_WS_CONNECTION_STATE",
//       payload: WebSocketConnectionState.CONNECTING,
//     });

//     const token = document.cookie
//       .split('; ')
//       .find(row => row.startsWith('auth-token='))
//       ?.split('=')[1];
      
//     const params = new URLSearchParams({ 
//       device_id: deviceId,
//       ...(token && { auth_token: token })
//     });
    
//     const ws = new WebSocket(
//       `${process.env.NEXT_PUBLIC_API_URL}/ws?${params.toString()}`
//     );

//     ws.onopen = () => {
//       console.log("WebSocket connection established");
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.CONNECTED,
//       });
//       ws.send(
//         JSON.stringify({
//           type: "workspace_info",
//           content: {},
//         })
//       );
//     };

//     ws.onmessage = (event) => {
//       try {
//         const data = JSON.parse(event.data);
        
//         // Check if this is an assistant message that might contain AG-UI component
//         if (data.type === AgentEvent.ASSISTANT_MESSAGE && data.content?.text) {
//           try {
//             const parsed = JSON.parse(data.content.text);
//             if (parsed.type && parsed.props && getTool(parsed.type.toLowerCase())) {
//               // Emit AG-UI event for component rendering
//               window.dispatchEvent(new CustomEvent('ag-ui-event', { 
//                 detail: {
//                   type: 'renderComponent',
//                   payload: {
//                     componentName: parsed.type.toLowerCase(),
//                     props: parsed.props
//                   }
//                 }
//               }));
//             }
//           } catch {
//             // Not a component spec, continue normal processing
//           }
//         }
        
//         // Always handle the event normally (for message display, etc.)
//         handleEvent({ 
//           ...data, 
//           id: data.id || generateUniqueId(data.type) 
//         });
//       } catch (error) {
//         console.error("Error parsing WebSocket data:", error);
//       }
//     };

//     ws.onerror = (error) => {
//       console.log("WebSocket error:", error);
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.DISCONNECTED,
//       });
//       toast.error("WebSocket connection error");
//     };

//     ws.onclose = () => {
//       console.log("WebSocket connection closed");
//       dispatch({
//         type: "SET_WS_CONNECTION_STATE",
//         payload: WebSocketConnectionState.DISCONNECTED,
//       });
//       setSocket(null);
//     };

//     setSocket(ws);
//   };

//   const sendMessage = (payload: {
//     type: string;
//     content: WebSocketMessageContent;
//   }) => {
//     if (!socket || socket.readyState !== WebSocket.OPEN) {
//       toast.error("WebSocket connection is not open. Please try again.");
//       return;
//     }
    
//     // Add unique ID to all messages
//     const messageWithId = {
//       ...payload,
//       id: generateUniqueId(payload.type),
//     };
    
//     socket.send(JSON.stringify(messageWithId));
//   };

//   // Set up AG-UI event listener for component interactions
//   useEffect(() => {
//     const unsubscribe = subscribe((event) => {
//       if (!socket || socket.readyState !== WebSocket.OPEN) return;
      
//       if (event.type === 'agentResponse') {
//         // Component is sending a response back - send as user message
//         sendMessage({
//           type: 'user_message',
//           content: {
//             text: event.payload
//           }
//         });
//       } else if (event.type === 'toolCall') {
//         // Component is requesting a tool call - send as user message with structured format
//         sendMessage({
//           type: 'user_message',
//           content: {
//             text: `Execute tool: ${event.payload.tool} with parameters: ${JSON.stringify(event.payload.args)}`
//           }
//         });
//       }
//     });

//     return unsubscribe;
//   }, [socket]);

//   useEffect(() => {
//     if (!isReplayMode) {
//       connectWebSocket();
//     }

//     return () => {
//       if (socket && socket.readyState === WebSocket.OPEN) {
//         socket.close();
//       }
//     };
//   }, [deviceId, isReplayMode]);

//   return { socket, sendMessage };
// }

// hooks/use-websocket.tsx
import { AgentEvent, WebSocketConnectionState, Message } from "@/typings/agent";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import { useAppContext } from "@/context/app-context";
import { generateUniqueId } from '@/utils/id-generator';
import { subscribe, getTool } from '@/agentic-ui/ag-ui-client';

interface WebSocketMessageContent {
  [key: string]: unknown;
}

export function useWebSocket(
  deviceId: string,
  isReplayMode: boolean,
  handleEvent: (
    data: {
      id: string;
      type: AgentEvent;
      content: Record<string, unknown>;
    },
    workspacePath?: string
  ) => void
) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const { dispatch } = useAppContext();

  const connectWebSocket = () => {
    dispatch({
      type: "SET_WS_CONNECTION_STATE",
      payload: WebSocketConnectionState.CONNECTING,
    });

    const token = document.cookie
      .split('; ')
      .find(row => row.startsWith('auth-token='))
      ?.split('=')[1];
      
    const params = new URLSearchParams({ 
      device_id: deviceId,
      ...(token && { auth_token: token })
    });
    
    const ws = new WebSocket(
      `${process.env.NEXT_PUBLIC_API_URL}/ws?${params.toString()}`
    );

    ws.onopen = () => {
      console.log("WebSocket connection established");
      dispatch({
        type: "SET_WS_CONNECTION_STATE",
        payload: WebSocketConnectionState.CONNECTED,
      });
      ws.send(
        JSON.stringify({
          type: "workspace_info",
          content: {},
        })
      );
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Check if this is an AGENT_RESPONSE that contains AG-UI component JSON
        if (data.type === AgentEvent.AGENT_RESPONSE && data.content?.text) {
          try {
            const parsed = JSON.parse(data.content.text);
            if (parsed.type && parsed.props && getTool(parsed.type.toLowerCase())) {
              // Emit AG-UI event for component rendering
              window.dispatchEvent(new CustomEvent('ag-ui-event', { 
                detail: {
                  type: 'renderComponent',
                  payload: {
                    componentName: parsed.type.toLowerCase(),
                    props: parsed.props
                  }
                }
              }));
            }
          } catch {
            // Not a component spec, continue normal processing
          }
        }
        
        // Always handle the event normally (for message display, etc.)
        handleEvent({ 
          ...data, 
          id: data.id || generateUniqueId(data.type) 
        });
      } catch (error) {
        console.error("Error parsing WebSocket data:", error);
      }
    };

    ws.onerror = (error) => {
      console.log("WebSocket error:", error);
      dispatch({
        type: "SET_WS_CONNECTION_STATE",
        payload: WebSocketConnectionState.DISCONNECTED,
      });
      toast.error("WebSocket connection error");
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
      dispatch({
        type: "SET_WS_CONNECTION_STATE",
        payload: WebSocketConnectionState.DISCONNECTED,
      });
      setSocket(null);
    };

    setSocket(ws);
  };

  const sendMessage = (payload: {
    type: string;
    content: WebSocketMessageContent;
  }) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      toast.error("WebSocket connection is not open. Please try again.");
      return;
    }
    
    // Add unique ID to all messages
    const messageWithId = {
      ...payload,
      id: generateUniqueId(payload.type),
    };
    
    socket.send(JSON.stringify(messageWithId));
  };

  // Set up AG-UI event listener for component interactions
  useEffect(() => {
    const unsubscribe = subscribe((event) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      
      if (event.type === 'agentResponse') {
        // Component is sending a response back - send as user message
        sendMessage({
          type: 'user_message',
          content: {
            text: event.payload
          }
        });
      } else if (event.type === 'toolCall') {
        // Component is requesting a tool call - send as user message
        sendMessage({
          type: 'user_message',
          content: {
            text: `Component action: ${event.payload.tool} with ${JSON.stringify(event.payload.args)}`
          }
        });
      }
    });

    return unsubscribe;
  }, [socket]);

  useEffect(() => {
    if (!isReplayMode) {
      connectWebSocket();
    }

    return () => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, [deviceId, isReplayMode]);

  return { socket, sendMessage };
}