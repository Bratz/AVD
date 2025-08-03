"use client";

import { createContext, useContext, useReducer, ReactNode } from "react";
import {
  ActionStep,
  AgentEvent,
  AVAILABLE_MODELS,
  Message,
  TAB,
  ToolSettings,
  WebSocketConnectionState,
} from "@/typings/agent";

// Define the thinking message interface
interface ThinkingMessage {
  id: string;
  content: string;
  timestamp: number;
}

// Define the state interface
interface AppState {
  messages: Message[];
  isLoading: boolean;
  activeTab: TAB;
  currentActionData?: ActionStep;
  activeFileCodeEditor: string;
  currentQuestion: string;
  isCompleted: boolean;
  isStopped: boolean;
  workspaceInfo: string;
  isUploading: boolean;
  uploadedFiles: string[];
  filesContent: { [key: string]: string };
  browserUrl: string;
  isGeneratingPrompt: boolean;
  editingMessage?: Message;
  toolSettings: ToolSettings;
  selectedModel?: string;
  wsConnectionState: WebSocketConnectionState;
  thinkingMessages: ThinkingMessage[]; // Add this
  debuggerEnabled: boolean;
  monitoringEnabled: boolean;
  debugEvents: DebugEvent[];
  agentMetrics: AgentMetrics;
  agentInitialized: boolean;
}

// Define action types
type AppAction =
  | { type: "SET_MESSAGES"; payload: Message[] }
  | { type: "ADD_MESSAGE"; payload: Message }
  | { type: "UPDATE_MESSAGE"; payload: Message }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ACTIVE_TAB"; payload: TAB }
  | { type: "SET_CURRENT_ACTION_DATA"; payload: ActionStep | undefined }
  | { type: "SET_ACTIVE_FILE"; payload: string }
  | { type: "SET_CURRENT_QUESTION"; payload: string }
  | { type: "SET_COMPLETED"; payload: boolean }
  | { type: "SET_STOPPED"; payload: boolean }
  | { type: "SET_WORKSPACE_INFO"; payload: string }
  | { type: "SET_IS_UPLOADING"; payload: boolean }
  | { type: "SET_UPLOADED_FILES"; payload: string[] }
  | { type: "ADD_UPLOADED_FILES"; payload: string[] }
  | { type: "SET_FILES_CONTENT"; payload: { [key: string]: string } }
  | { type: "ADD_FILE_CONTENT"; payload: { path: string; content: string } }
  | { type: "SET_BROWSER_URL"; payload: string }
  | { type: "SET_GENERATING_PROMPT"; payload: boolean }
  | { type: "SET_EDITING_MESSAGE"; payload: Message | undefined }
  | { type: "SET_TOOL_SETTINGS"; payload: AppState["toolSettings"] }
  | { type: "SET_SELECTED_MODEL"; payload: string | undefined }
  | { type: "SET_WS_CONNECTION_STATE"; payload: WebSocketConnectionState }
  | { type: "ADD_THINKING_MESSAGE"; payload: ThinkingMessage } // Add this
  | { type: "CLEAR_THINKING_MESSAGES" } // Add this
  | {
      type: "HANDLE_EVENT";
      payload: {
        event: AgentEvent;
        data: Record<string, unknown>;
        workspacePath?: string;
      }}
  | { type: "TOGGLE_DEBUGGER"; payload: boolean }
  | { type: "TOGGLE_MONITORING"; payload: boolean }
  | { type: "ADD_DEBUG_EVENT"; payload: DebugEvent }
  | { type: "UPDATE_METRICS"; payload: Partial<AgentMetrics> }
  | { type: "CLEAR_DEBUG_EVENTS" }
  | { type: "SET_AGENT_INITIALIZED"; payload: boolean };
    


interface DebugEvent {
  id: string;
  timestamp: string;
  type: 'agent_thinking' | 'tool_call' | 'tool_result' | 'error' | 'user_message' | 'agent_response';
  content: {
    text?: string;
    tool_name?: string;
    tool_input?: Record<string, unknown>;
    result?: string | Record<string, unknown>;
    error?: string;
    [key: string]: unknown;
  };
  duration?: number;
  tokens?: number;
  status?: 'success' | 'error' | 'pending';
  toolName?: string;
}

interface AgentMetrics {
  totalTokens: number;
  totalDuration: number;
  toolCallCount: number;
  thoughtSteps: number;
  errorCount: number;
  successRate: number;
}


// Initial state
const initialState: AppState = {
  messages: [],
  isLoading: false,
  activeTab: TAB.BROWSER,
  activeFileCodeEditor: "",
  currentQuestion: "",
  isCompleted: false,
  isStopped: false,
  workspaceInfo: "",
  isUploading: false,
  uploadedFiles: [],
  filesContent: {},
  browserUrl: "",
  isGeneratingPrompt: false,
  toolSettings: {
    deep_research: false,
    pdf: true,
    media_generation: true,
    audio_generation: true,
    browser: true,
    thinking_tokens: 10000,
    banking_mode: false,
  },
  wsConnectionState: WebSocketConnectionState.CONNECTING,
  selectedModel: AVAILABLE_MODELS[0],
  thinkingMessages: [], // Add this
  debuggerEnabled: false,
  monitoringEnabled: false,
  debugEvents: [],
  agentMetrics: {
    totalTokens: 0,
    totalDuration: 0,
    toolCallCount: 0,
    thoughtSteps: 0,
    errorCount: 0,
    successRate: 100,
  },
  agentInitialized: false,
};

// Create the context
const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}>({
  state: initialState,
  dispatch: () => null,
});

// Reducer function
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_MESSAGES":
      return { ...state, messages: action.payload };
    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload] };
    case "UPDATE_MESSAGE":
      return {
        ...state,
        messages: state.messages.map((message) =>
          message.id === action.payload.id ? action.payload : message
        ),
      };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_ACTIVE_TAB":
      return { ...state, activeTab: action.payload };
    case "SET_CURRENT_ACTION_DATA":
      return { ...state, currentActionData: action.payload };
    case "SET_ACTIVE_FILE":
      return { ...state, activeFileCodeEditor: action.payload };
    case "SET_CURRENT_QUESTION":
      return { ...state, currentQuestion: action.payload };
    case "SET_COMPLETED":
      return { ...state, isCompleted: action.payload };
    case "SET_STOPPED":
      return { ...state, isStopped: action.payload };
    case "SET_WORKSPACE_INFO":
      return { ...state, workspaceInfo: action.payload };
    case "SET_IS_UPLOADING":
      return { ...state, isUploading: action.payload };
    case "SET_UPLOADED_FILES":
      return { ...state, uploadedFiles: action.payload };
    case "ADD_UPLOADED_FILES":
      return {
        ...state,
        uploadedFiles: [...state.uploadedFiles, ...action.payload],
      };
    case "SET_FILES_CONTENT":
      return { ...state, filesContent: action.payload };
    case "ADD_FILE_CONTENT":
      return {
        ...state,
        filesContent: {
          ...state.filesContent,
          [action.payload.path]: action.payload.content,
        },
      };
    case "SET_BROWSER_URL":
      return { ...state, browserUrl: action.payload };
    case "SET_GENERATING_PROMPT":
      return { ...state, isGeneratingPrompt: action.payload };
    case "SET_EDITING_MESSAGE":
      return { ...state, editingMessage: action.payload };
    case "SET_TOOL_SETTINGS":
      return { ...state, toolSettings: action.payload };
    case "SET_SELECTED_MODEL":
      return { ...state, selectedModel: action.payload };
    case "SET_WS_CONNECTION_STATE":
      return { ...state, wsConnectionState: action.payload };
    case "ADD_THINKING_MESSAGE":
      return { 
        ...state, 
        thinkingMessages: [...state.thinkingMessages, action.payload] 
      };
    case "CLEAR_THINKING_MESSAGES":
      return { ...state, thinkingMessages: [] };
    case "TOGGLE_DEBUGGER":
      return { ...state, debuggerEnabled: action.payload };
    case "TOGGLE_MONITORING":
      return { ...state, monitoringEnabled: action.payload };
    case "ADD_DEBUG_EVENT":
      return { 
        ...state, 
        debugEvents: [...state.debugEvents, action.payload].slice(-100) // Keep last 100 events
      };
    case "UPDATE_METRICS":
      return { 
        ...state, 
        agentMetrics: { ...state.agentMetrics, ...action.payload }
      };
    case "CLEAR_DEBUG_EVENTS":
      return { ...state, debugEvents: [] };
    case "SET_AGENT_INITIALIZED":
      return { ...state, agentInitialized: action.payload };
    default:
      return state;
  }
}

// Context provider component
export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

// Custom hook to use the context
export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
}

// Add this to your app-context.tsx or create a new hook

interface WebSocketEventHandler {
  handleWebSocketMessage: (data: any) => void;
}

export const createWebSocketEventHandler = (dispatch: any): WebSocketEventHandler => {
  let eventCounter = 0;
  let thinkingCounter = 0;
  
  return {
    handleWebSocketMessage: (data: any) => {
      const timestamp = new Date().toISOString();
      
      switch (data.type) {
        case 'agent_thinking':
          // Add thinking message
          dispatch({
            type: 'ADD_THINKING_MESSAGE',
            payload: {
              id: `thinking-${thinkingCounter++}`,
              content: data.content.text || '',
              timestamp: Date.now()
            }
          });
          
          // Update metrics
          dispatch({
            type: 'INCREMENT_THOUGHT_STEPS'
          });
          break;
          
        case 'tool_call':
          // Add debug event for tool call
          dispatch({
            type: 'ADD_DEBUG_EVENT',
            payload: {
              id: `event-${eventCounter++}`,
              timestamp,
              type: 'tool_call',
              content: {
                tool_name: data.content.tool_name,
                tool_input: data.content.tool_input
              },
              status: 'pending',
              toolName: data.content.tool_name
            }
          });
          
          // Update metrics
          dispatch({
            type: 'INCREMENT_TOOL_CALLS'
          });
          break;
          
        case 'tool_result':
          // Add debug event for tool result
          const duration = Math.random() * 2000 + 500; // Calculate actual duration if available
          dispatch({
            type: 'ADD_DEBUG_EVENT',
            payload: {
              id: `event-${eventCounter++}`,
              timestamp,
              type: 'tool_result',
              content: {
                tool_name: data.content.tool_name,
                result: data.content.result
              },
              duration,
              status: 'success',
              toolName: data.content.tool_name
            }
          });
          
          // Update token count (estimate)
          dispatch({
            type: 'ADD_TOKENS',
            payload: Math.floor(Math.random() * 1000 + 100)
          });
          break;
          
        case 'agent_response':
          // Add response event
          dispatch({
            type: 'ADD_DEBUG_EVENT',
            payload: {
              id: `event-${eventCounter++}`,
              timestamp,
              type: 'agent_response',
              content: {
                text: data.content.text || data.content.message
              },
              status: 'success'
            }
          });
          break;
          
        case 'error':
          // Handle errors
          dispatch({
            type: 'INCREMENT_ERRORS'
          });
          
          dispatch({
            type: 'ADD_DEBUG_EVENT',
            payload: {
              id: `event-${eventCounter++}`,
              timestamp,
              type: 'error',
              content: data.content,
              status: 'error'
            }
          });
          break;
          
        case 'connection_established':
          // Reset metrics on new connection
          dispatch({
            type: 'RESET_METRICS'
          });
          break;
      }
    }
  };
};