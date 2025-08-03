"use client";

import { Terminal as XTerm } from "@xterm/xterm";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import {
  Code,
  Globe,
  Terminal as TerminalIcon,
  X,
  Loader2,
  Brain,
  Network,
  ChevronDown,
  ChevronUp,
  Zap
} from "lucide-react";
import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import dynamic from "next/dynamic";
import { Orbitron } from "next/font/google";
import { useSearchParams } from "next/navigation";

import { useDeviceId } from "@/hooks/use-device-id";
import { useWebSocket } from "@/hooks/use-websocket";
import { useSessionManager } from "@/hooks/use-session-manager";
import { useAppEvents } from "@/hooks/use-app-events";
import { useAppContext } from "@/context/app-context";

import SidebarButton from "@/components/sidebar-button";
import ConnectionStatus from "@/components/connection-status";
import Browser from "@/components/browser";
import CodeEditor from "@/components/code-editor";
import QuestionInput from "@/components/question-input";
import SearchBrowser from "@/components/search-browser";
import { Button } from "@/components/ui/button";
import ChatMessage from "@/components/chat-message";
import ImageBrowser from "@/components/image-browser";
import { Message, TAB, TOOL } from "@/typings/agent";
import Markdown from "@/components/markdown";
import { TypingMarkdown } from '@/components/typing-markdown';
import AgentDebuggerPanel from "@/components/agent-debugger-panel";
import AgentMonitoringDashboard from "@/components/agent-monitoring-dashboard";
import { DebugControls } from "@/components/debug-controls";
import { ThemeToggle } from "@/components/theme-toggle";
import { UIModeToggle } from "@/components/ui-mode-toggle";
import { getCurrentUIMode } from '@/utils/ui-mode';
import React from "react";

const Terminal = dynamic(() => import("@/components/terminal"), {
  ssr: false,
});

const orbitron = Orbitron({
  subsets: ["latin"],
});

export default function HomeContent() {
  const xtermRef = useRef<XTerm | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { state, dispatch } = useAppContext();
  const { handleEvent, handleClickAction } = useAppEvents({ xtermRef });
  const searchParams = useSearchParams();

  const { deviceId } = useDeviceId();

  // State for thinking messages
  const thinkingScrollRef = useRef<HTMLDivElement>(null);
  const [typingCompleted, setTypingCompleted] = useState<Set<string>>(new Set());
  const [hasNewThinking, setHasNewThinking] = useState(false);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

 


  const [showROWBOAT, setShowROWBOAT] = useState(false);

  // Use from global state:
  const thinkingMessages = state.thinkingMessages;

  const [showThinking, setShowThinking] = useState(false);
  const [showRightPanel, setShowRightPanel] = useState(false);

  // Use the Session Manager hook
  const { sessionId, isLoadingSession, isReplayMode, setSessionId } =
    useSessionManager({
      searchParams,
      handleEvent,
    });

  // Use the WebSocket hook with custom handler
  const { socket, sendMessage } = useWebSocket(
    deviceId,
    isReplayMode,
    handleEvent
  );

  const handleEnhancePrompt = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      toast.error("WebSocket connection is not open. Please try again.");
      return;
    }
    dispatch({ type: "SET_GENERATING_PROMPT", payload: true });
    sendMessage({
      type: "enhance_prompt",
      content: {
        model_name: state.selectedModel,
        text: state.currentQuestion,
        files: state.uploadedFiles?.map((file) => `.${file}`),
        tool_args: {
          thinking_tokens: 0,
        },
      },
    });
  };

  const handleQuestionSubmit = async (newQuestion: string) => {
    if (!newQuestion.trim() || state.isLoading) return;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      toast.error("WebSocket connection is not open. Please try again.");
      dispatch({ type: "SET_LOADING", payload: false });
      return;
    }

    // Clear thinking messages for new query
    dispatch({ type: "CLEAR_THINKING_MESSAGES" });
    setTypingCompleted(new Set());
    setHasNewThinking(false);
    setIsUserScrolling(false);

    // Ensure thinking box is visible for new query
    setShowThinking(true);
    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_CURRENT_QUESTION", payload: "" });
    dispatch({ type: "SET_COMPLETED", payload: false });
    dispatch({ type: "SET_STOPPED", payload: false });

    if (!sessionId) {
      const id = `${state.workspaceInfo}`.split("/").pop();
      if (id) {
        setSessionId(id);
      }
    }

    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: newQuestion,
      timestamp: Date.now(),
    };

    dispatch({
      type: "ADD_MESSAGE",
      payload: newUserMessage,
    });
    
    // send init agent event when first query
    if (!sessionId) {
      interface InitAgentContent {
        model_name: string | undefined;
        tool_args: typeof state.toolSettings;
        enable_mcp?: boolean;
        mcp_config?: {
          base_url: string;
          sse_endpoint: string;
          api_key: string;
        };
        use_routing?: boolean;
        simple_model?: string;
        complex_model?: string;
        complexity_threshold?: number;
        [key: string]: unknown; // Add index signature to match WebSocketMessageContent
      }

    const initContent: InitAgentContent = {
      model_name: state.selectedModel,
      tool_args: {
        ...state.toolSettings,
        // Ensure routing settings are included
        use_routing: state.toolSettings.use_routing || false,
        simple_model: state.toolSettings.simple_model,
        complex_model: state.toolSettings.complex_model,
        complexity_threshold: state.toolSettings.complexity_threshold || 0.5,
        // ui_mode: getCurrentUIMode(false),
      },
      // Pass routing settings at the top level too for the LLM client
      use_routing: state.toolSettings.use_routing || false,
      simple_model: state.toolSettings.simple_model,
      complex_model: state.toolSettings.complex_model,
      complexity_threshold: state.toolSettings.complexity_threshold || 0.5,
    };

      // If banking mode is enabled, also enable MCP
      if (state.toolSettings.banking_mode) {
        initContent.enable_mcp = true;
        initContent.mcp_config = {
          base_url: process.env.NEXT_PUBLIC_MCP_BASE_URL || "http://localhost:8082",
          sse_endpoint: process.env.NEXT_PUBLIC_MCP_SSE_ENDPOINT || "http://localhost:8084/mcp",
          api_key: process.env.NEXT_PUBLIC_MCP_API_KEY || "test-api-key-123"
        };
      }

      sendMessage({
        type: "init_agent",
        content: initContent,
      });
    }

    // Send the query using the existing socket connection
    sendMessage({
      type: "query",
      content: {
        text: newQuestion,
        resume: state.messages.length > 0,
        files: state.uploadedFiles?.map((file) => `.${file}`),
      },
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuestionSubmit((e.target as HTMLTextAreaElement).value);
    }
  };

  const handleResetChat = () => {
    window.location.href = "/";
  };

  const handleOpenVSCode = () => {
    let url = process.env.NEXT_PUBLIC_VSCODE_URL || "http://127.0.0.1:8080";
    url += `/?folder=${state.workspaceInfo}`;
    window.open(url, "_blank");
  };

  const parseJson = (jsonString: string) => {
    try {
      return JSON.parse(jsonString);
    } catch {
      return null;
    }
  };

  const handleEditMessage = (newQuestion: string) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      toast.error("WebSocket connection is not open. Please try again.");
      dispatch({ type: "SET_LOADING", payload: false });
      return;
    }

    socket.send(
      JSON.stringify({
        type: "edit_query",
        content: {
          text: newQuestion,
          files: state.uploadedFiles?.map((file) => `.${file}`),
        },
      })
    );

    // Update the edited message and remove all subsequent messages
    const editIndex = state.messages.findIndex(
      (m) => m.id === state.editingMessage?.id
    );

    if (editIndex >= 0) {
      const updatedMessages = [...state.messages.slice(0, editIndex + 1)];
      updatedMessages[editIndex] = {
        ...updatedMessages[editIndex],
        content: newQuestion,
      };

      dispatch({
        type: "SET_MESSAGES",
        payload: updatedMessages,
      });
    }

    dispatch({ type: "SET_COMPLETED", payload: false });
    dispatch({ type: "SET_STOPPED", payload: false });
    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_EDITING_MESSAGE", payload: undefined });
  };

  const getRemoteURL = (path: string | undefined) => {
    if (!path || !state.workspaceInfo) return "";
    const workspaceId = state.workspaceInfo.split("/").pop();
    return `${process.env.NEXT_PUBLIC_API_URL}/workspace/${workspaceId}/${path}`;
  };

  const isInChatView = useMemo(
    () => !!sessionId && !isLoadingSession,
    [isLoadingSession, sessionId]
  );

  const handleShare = () => {
    if (!sessionId) return;
    const url = `${window.location.origin}/?id=${sessionId}`;
    navigator.clipboard.writeText(url);
    toast.success("Copied to clipboard");
  };

  const handleCancelQuery = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      toast.error("WebSocket connection is not open.");
      return;
    }

    // Send cancel message to the server
    sendMessage({
      type: "cancel",
      content: {}
    });
    dispatch({ type: "SET_LOADING", payload: false });
    dispatch({ type: "SET_STOPPED", payload: true });
  };

  const isBrowserTool = useMemo(
    () =>
      [
        TOOL.BROWSER_VIEW,
        TOOL.BROWSER_CLICK,
        TOOL.BROWSER_ENTER_TEXT,
        TOOL.BROWSER_PRESS_KEY,
        TOOL.BROWSER_GET_SELECT_OPTIONS,
        TOOL.BROWSER_SELECT_DROPDOWN_OPTION,
        TOOL.BROWSER_SWITCH_TAB,
        TOOL.BROWSER_OPEN_NEW_TAB,
        TOOL.BROWSER_WAIT,
        TOOL.BROWSER_SCROLL_DOWN,
        TOOL.BROWSER_SCROLL_UP,
        TOOL.BROWSER_NAVIGATION,
        TOOL.BROWSER_RESTART,
      ].includes(state.currentActionData?.type as TOOL),
    [state.currentActionData]
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.messages?.length]);

  // Enhanced auto-scroll with user scroll detection
  useEffect(() => {
    if (thinkingScrollRef.current && showThinking && !isUserScrolling) {
      const element = thinkingScrollRef.current;
      // Smooth scroll to bottom
      element.scrollTo({
        top: element.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [thinkingMessages.length, showThinking, isUserScrolling, typingCompleted.size]);

  // Add scroll event handler to detect user scrolling
  useEffect(() => {
    const handleScroll = () => {
      if (!thinkingScrollRef.current) return;
      
      const element = thinkingScrollRef.current;
      const isAtBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 10;
      
      // User is scrolling up
      if (!isAtBottom) {
        setIsUserScrolling(true);
        
        // Clear existing timeout
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }
        
        // Reset after 3 seconds of no scrolling
        scrollTimeoutRef.current = setTimeout(() => {
          setIsUserScrolling(false);
        }, 3000);
      } else {
        setIsUserScrolling(false);
      }
    };
    
    const element = thinkingScrollRef.current;
    if (element) {
      element.addEventListener('scroll', handleScroll);
      return () => element.removeEventListener('scroll', handleScroll);
    }
  }, [showThinking]);

  // Clear typing completed state when messages are cleared
  useEffect(() => {
    if (thinkingMessages.length === 0) {
      setTypingCompleted(new Set());
    }
  }, [thinkingMessages.length]);

  // New thinking indicator
  useEffect(() => {
    if (thinkingMessages.length > 0 && !showThinking) {
      setHasNewThinking(true);
    }
  }, [thinkingMessages.length, showThinking]);

  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Fixed Header for Chat View */}
      {isInChatView && (
        <motion.header 
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          className="fixed top-0 left-0 right-0 h-16 bg-background/80 backdrop-blur-sm border-b border-border z-40"
        >
          <div className="h-full px-4 flex items-center justify-between">
            {/* Left section with proper spacing */}
            <div className="flex items-center gap-3 flex-1">
              {/* Spacer for sidebar button */}
              <div className="w-16" />
              
              {/* Logo and title */}
              <motion.div 
                className="flex items-center gap-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Image
                  src="/logo-only.png"
                  alt="chatGBP Logo"
                  width={24}
                  height={24}
                  className="rounded-sm"
                  style={{ width: '24px', height: 'auto' }}
                />
                <h1 className={`font-semibold text-xl sm:text-xl ${orbitron.className}`}>
                  chatGBP
                </h1>
              </motion.div>
            </div>
            
            {/* Center section - Routing indicator */}
            {state.toolSettings.use_routing && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 px-3 py-1 rounded-full bg-muted/50 text-xs"
              >
                <div className="flex items-center gap-1">
                  <Zap className="h-3 w-3 text-yellow-400" />
                  <span>{state.toolSettings.simple_model?.split('/').pop()}</span>
                </div>
                <span className="text-muted-foreground">↔</span>
                <div className="flex items-center gap-1">
                  <Brain className="h-3 w-3 text-purple-400" />
                  <span>{state.toolSettings.complex_model?.split('/').pop()}</span>
                </div>
              </motion.div>
            )}
            
            {/* Right section */}
            <motion.div 
              className="flex items-center gap-2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowROWBOAT(true)}
                title="Open Multi-Agent Workflows"
              >
                <Network className="h-4 w-4 mr-2" />
                Multi-Agent
              </Button>
              <Button 
                className="h-9" 
                variant="outline"
                onClick={handleResetChat}
              >
                <X className="h-4 w-4" />
              </Button>
              <div className="ml-2 h-6 w-px bg-border" />
              <UIModeToggle isCopilot={false} compact={true} />
              <div className="ml-2 h-6 w-px bg-border" />
              <ThemeToggle />
            </motion.div>
          </div>
        </motion.header>
      )}

      {/* Fixed Controls Layer */}
      <div className="fixed top-0 left-0 right-0 z-50 pointer-events-none">
        <div className="flex items-start justify-between p-4">
          {/* Left side controls */}
          <div className="flex flex-col gap-2 pointer-events-auto">
            <SidebarButton />
            {isInChatView && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <DebugControls />
              </motion.div>
            )}
          </div>

          {/* Right side controls */}
          <motion.div 
            className="pointer-events-auto flex items-center gap-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            {!isInChatView && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowROWBOAT(true)}
                  title="Open Multi-Agent Workflows"
                >
                  <Network className="h-4 w-4 mr-2" />
                  Multi-Agent
                </Button>
                <UIModeToggle isCopilot={false} compact={true} />
                <div className="ml-2 h-6 w-px bg-border" />
              </>
            )}
            <ThemeToggle />
          </motion.div>
        </div>
      </div>

      {/* Thinking Box */}
      <AnimatePresence>
        {thinkingMessages.length > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            className="fixed top-20 right-4 w-[360px] max-h-[400px] bg-card/95 backdrop-blur-md border border-border rounded-xl shadow-2xl z-40 overflow-hidden"
          >
            {/* Enhanced header with gradient */}
            <div className="relative bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-b border-border">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 animate-pulse opacity-50" />
              <div className="relative flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Brain className="h-5 w-5 text-blue-400" />
                    <motion.div
                      className="absolute -top-1 -right-1 h-2 w-2 bg-blue-400 rounded-full"
                      animate={{
                        scale: [1, 1.5, 1],
                        opacity: [1, 0.5, 1],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                      }}
                    />
                  </div>
                  <span className="text-sm font-medium text-foreground">
                    Agent Thinking
                  </span>
                  <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                    {thinkingMessages.length}
                  </span>
                  {hasNewThinking && !showThinking && (
                    <motion.span 
                      className="h-2 w-2 bg-blue-400 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                  )}
                </div>
                
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      dispatch({ type: "CLEAR_THINKING_MESSAGES" });
                      setTypingCompleted(new Set());
                    }}
                    className="h-7 w-7 hover:bg-muted/50 rounded-lg transition-all"
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setShowThinking(!showThinking);
                      setHasNewThinking(false);
                    }}
                    className="h-7 w-7 hover:bg-muted/50 rounded-lg transition-all"
                  >
                    <motion.div
                      animate={{ rotate: showThinking ? 180 : 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChevronDown className="h-3.5 w-3.5" />
                    </motion.div>
                  </Button>
                </div>
              </div>
            </div>
            
            <AnimatePresence>
              {showThinking && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                >
                  <div 
                    ref={thinkingScrollRef}
                    className="p-3 max-h-[350px] overflow-y-auto scrollbar-thin space-y-2"
                  >
                    {[...thinkingMessages]
                      .sort((a, b) => a.timestamp - b.timestamp)
                      .map((msg, index, sortedMessages) => {
                        const prevMsg = index > 0 ? sortedMessages[index - 1] : null;
                        const isNewSection = prevMsg && (msg.timestamp - prevMsg.timestamp) > 5000;
                        
                        return (
                          <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: Math.min(index * 0.05, 0.3) }}
                            className="text-xs text-muted-foreground"
                          >
                            {isNewSection && (
                              <div className="flex items-center gap-2 my-2 opacity-40">
                                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
                                <span className="text-[10px] text-muted-foreground px-2">new phase</span>
                                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
                              </div>
                            )}
                            
                            <div className="flex items-start gap-2">
                              <div className="mt-1 relative">
                                <div className="h-1.5 w-1.5 bg-blue-400 rounded-full flex-shrink-0" />
                                {index < sortedMessages.length - 1 && !isNewSection && (
                                  <div className="absolute top-2 left-[3px] w-px h-4 bg-blue-400/30" />
                                )}
                              </div>
                              <div className="flex-1 markdown-compact">
                                {typingCompleted.has(msg.id) ? (
                                  <div className="prose prose-xs dark:prose-invert max-w-none">
                                    <Markdown>{msg.content}</Markdown>
                                  </div>
                                ) : (
                                  <TypingMarkdown
                                    text={msg.content}
                                    speed={15}
                                    onComplete={() => {
                                      setTypingCompleted(prev => new Set(prev).add(msg.id));
                                    }}
                                  />
                                )}
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                    
                    {thinkingMessages.length > 0 && 
                    !typingCompleted.has(thinkingMessages[thinkingMessages.length - 1].id) && (
                      <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
                        <div className="flex space-x-1">
                          <motion.div 
                            className="w-1.5 h-1.5 bg-blue-400 rounded-full"
                            animate={{ y: [0, -8, 0] }}
                            transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                          />
                          <motion.div 
                            className="w-1.5 h-1.5 bg-blue-400 rounded-full"
                            animate={{ y: [0, -8, 0] }}
                            transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                          />
                          <motion.div 
                            className="w-1.5 h-1.5 bg-blue-400 rounded-full"
                            animate={{ y: [0, -8, 0] }}
                            transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                          />
                        </div>
                        <span>Processing...</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Debug Panel */}
      {state.debuggerEnabled && isInChatView && (
        <AgentDebuggerPanel 
          agentEvents={state.debugEvents}
          thinkingMessages={state.thinkingMessages}
          toolCalls={state.debugEvents.filter(e => e.type === 'tool_call')}
        />
      )}
      
      {/* Monitoring Dashboard Modal */}
      <AnimatePresence>
        {state.monitoringEnabled && isInChatView && (
          <AgentMonitoringDashboard 
            sessionId={sessionId}
            agentData={{
              metrics: state.agentMetrics,
              events: state.debugEvents,
              thinkingMessages: state.thinkingMessages
            }}
            onClose={() => dispatch({ type: "TOGGLE_MONITORING", payload: false })}
          />
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <main className={`flex-1 flex flex-col items-center justify-center min-h-screen ${isInChatView ? 'pt-16' : ''}`}>
        {/* Loading State */}
        {isLoadingSession ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center p-8"
          >
            <div className="relative">
              <Loader2 className="h-12 w-12 text-foreground animate-spin" />
              <motion.div
                className="absolute inset-0 h-12 w-12 rounded-full border-2 border-blue-500"
                animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
            <p className="text-foreground text-lg mt-4">Loading session history...</p>
          </motion.div>
        ) : (
          <LayoutGroup>
            <AnimatePresence mode="wait">
              {!isInChatView ? (
                /* Landing Page View */
                <motion.div
                  key="landing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center w-full max-w-4xl px-4 mx-auto"
                >
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Image
                      src="/logo-only.png"
                      alt="chatGBP Logo"
                      width={50}
                      height={50}
                      className="rounded-sm mb-6"
                      style={{ width: '50px', height: 'auto' }}
                    />
                  </motion.div>
                  
                  <motion.h1
                    className={`text-4xl sm:text-3xl font-semibold text-center mb-8 ${orbitron.className}`}
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    chatGBP
                  </motion.h1>
                  
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="w-full"
                  >
                    <QuestionInput
                      placeholder="Give chatGBP a task to work on..."
                      value={state.currentQuestion}
                      setValue={(value) =>
                        dispatch({ type: "SET_CURRENT_QUESTION", payload: value })
                      }
                      handleKeyDown={handleKeyDown}
                      handleSubmit={handleQuestionSubmit}
                      isDisabled={!socket || socket.readyState !== WebSocket.OPEN}
                      handleEnhancePrompt={handleEnhancePrompt}
                    />
                  </motion.div>
                  
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="mt-8"
                  >
                    <ConnectionStatus />
                  </motion.div>
                </motion.div>
              ) : (
                /* Chat View */
                <motion.div
                  key="chat"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={`w-full grid ${showRightPanel ? 'grid-cols-10' : 'grid-cols-1'} gap-4 p-4 max-w-[1920px] mx-auto`}
                >
                  {/* Chat Column */}
                  <div className={showRightPanel ? 'col-span-4' : 'col-span-1 max-w-4xl mx-auto w-full'}>
                    <ChatMessage
                      handleClickAction={handleClickAction}
                      isReplayMode={isReplayMode}
                      messagesEndRef={messagesEndRef}
                      setCurrentQuestion={(value) =>
                        dispatch({ type: "SET_CURRENT_QUESTION", payload: value })
                      }
                      handleKeyDown={handleKeyDown}
                      handleQuestionSubmit={handleQuestionSubmit}
                      handleEnhancePrompt={handleEnhancePrompt}
                      handleCancel={handleCancelQuery}
                      handleEditMessage={handleEditMessage}
                    />
                  </div>

                  {/* Toggle button for right panel */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 }}
                    className="fixed bottom-20 right-4 z-30"
                  >
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => setShowRightPanel(!showRightPanel)}
                      className="shadow-lg hover:shadow-xl transition-all group"
                      title={showRightPanel ? "Hide panels" : "Show panels"}
                    >
                      <motion.div
                        animate={{ rotate: showRightPanel ? 180 : 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <ChevronUp className="size-4 group-hover:text-blue-500 transition-colors" />
                      </motion.div>
                    </Button>
                  </motion.div>

                  {/* Right Panel */}
                  <AnimatePresence>
                    {showRightPanel && (
                      <motion.div 
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className="col-span-6"
                      >
                        <div className="bg-card border border-border rounded-2xl overflow-hidden h-full flex flex-col">
                          {/* Tab Header */}
                          <div className="p-4 border-b border-border bg-muted/30">
                            <div className="flex items-center justify-between">
                              <div className="flex gap-2">
                                {[
                                  { tab: TAB.BROWSER, icon: Globe, label: "Browser" },
                                  { tab: TAB.CODE, icon: Code, label: "Code" },
                                  { tab: TAB.TERMINAL, icon: TerminalIcon, label: "Terminal" }
                                ].map(({ tab, icon: Icon, label }) => (
                                  <Button
                                    key={tab}
                                    className={`transition-all ${
                                      state.activeTab === tab
                                        ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg"
                                        : "hover:bg-muted"
                                    }`}
                                    variant={state.activeTab === tab ? "default" : "outline"}
                                    onClick={() =>
                                      dispatch({
                                        type: "SET_ACTIVE_TAB",
                                        payload: tab,
                                      })
                                    }
                                  >
                                    <Icon className="size-4" />
                                    <span className="ml-2">{label}</span>
                                  </Button>
                                ))}
                              </div>
                              
                              <Button
                                variant="outline"
                                onClick={handleOpenVSCode}
                                className="hover:bg-muted transition-colors"
                              >
                                <Image
                                  src="/vscode.png"
                                  alt="VS Code"
                                  width={20}
                                  height={20}
                                  style={{ width: '20px', height: 'auto' }}
                                />
                                <span className="ml-2 hidden lg:inline">Open with VS Code</span>
                              </Button>
                            </div>
                          </div>

                          {/* Tab Content */}
                          <div className="flex-1 overflow-hidden">
                            <AnimatePresence mode="wait">
                              {state.activeTab === TAB.BROWSER && (
                                <motion.div
                                  key="browser"
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  exit={{ opacity: 0 }}
                                  className="h-full"
                                >
                                  {state.currentActionData?.type === TOOL.WEB_SEARCH ? (
                                    <SearchBrowser
                                      keyword={state.currentActionData?.data.tool_input?.query}
                                      search_results={
                                        state.currentActionData?.data?.result
                                          ? parseJson(state.currentActionData?.data?.result as string)
                                          : undefined
                                      }
                                    />
                                  ) : (state.currentActionData?.type === TOOL.IMAGE_GENERATE ||
                                      state.currentActionData?.type === TOOL.IMAGE_SEARCH) ? (
                                    <ImageBrowser
                                      url={
                                        state.currentActionData?.data.tool_input?.output_filename ||
                                        state.currentActionData?.data.tool_input?.query
                                      }
                                      images={
                                        state.currentActionData?.type === TOOL.IMAGE_SEARCH
                                          ? parseJson(state.currentActionData?.data?.result as string)
                                              ?.map((item: { image_url: string }) => item?.image_url)
                                          : [getRemoteURL(state.currentActionData?.data.tool_input?.output_filename)]
                                      }
                                    />
                                  ) : (
                                    <Browser
                                      url={state.currentActionData?.data?.tool_input?.url || state.browserUrl}
                                      screenshot={
                                        isBrowserTool
                                          ? (state.currentActionData?.data.result as string)
                                          : undefined
                                      }
                                      raw={
                                        state.currentActionData?.type === TOOL.VISIT
                                          ? (state.currentActionData?.data?.result as string)
                                          : undefined
                                      }
                                    />
                                  )}
                                </motion.div>
                              )}
                              
                              {state.activeTab === TAB.CODE && (
                                <motion.div
                                  key="code"
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  exit={{ opacity: 0 }}
                                  className="h-full"
                                >
                                  <CodeEditor
                                    currentActionData={state.currentActionData}
                                    activeTab={state.activeTab}
                                    workspaceInfo={state.workspaceInfo}
                                    activeFile={state.activeFileCodeEditor}
                                    setActiveFile={(file) =>
                                      dispatch({
                                        type: "SET_ACTIVE_FILE",
                                        payload: file,
                                      })
                                    }
                                    filesContent={state.filesContent}
                                    isReplayMode={isReplayMode}
                                  />
                                </motion.div>
                              )}
                              
                              {state.activeTab === TAB.TERMINAL && (
                                <motion.div
                                  key="terminal"
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  exit={{ opacity: 0 }}
                                  className="h-full"
                                >
                                  <Terminal ref={xtermRef} />
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}
            </AnimatePresence>
          </LayoutGroup>
        )}
      </main>

      {/* ROWBOAT Modal - Now at the end, outside of all conditionals */}
      {/* ROWBOAT Modal */}
      {showROWBOAT && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50"
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="fixed inset-4 bg-background border rounded-lg shadow-lg overflow-hidden"
          >
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between p-4 border-b">
                <div className="flex items-center gap-2">
                  <Network className="h-5 w-5 text-primary" />
                  <h2 className="text-lg font-semibold">TCS BaNCS SWARM Studio</h2>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowROWBOAT(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex-1 overflow-hidden">
                <React.Suspense fallback={<div className="p-4">Loading...</div>}>
                  {(() => {
                    try {
                      // Fixed import path
                      const { registry } = require('@/agentic-ui/registry');
                      const Component = registry.ROWBOATDashboard;
                      
                      if (!Component) {
                        console.error('ROWBOATDashboard not found in registry');
                        return (
                          <div className="p-4 text-center">
                            <p className="text-red-500 mb-2">Error: ROWBOATDashboard component not found</p>
                            <p className="text-sm text-muted-foreground">Check console for details</p>
                          </div>
                        );
                      }
                      
                      return <Component />;
                    } catch (error) {
                      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
                      console.error('Error loading ROWBOAT:', error);
                      return (
                        <div className="p-4 text-center">
                          <p className="text-red-500 mb-2">Error loading ROWBOAT</p>
                          <p className="text-sm text-muted-foreground">{errorMessage}</p>
                        </div>
                      );
                    }
                  })()}
                </React.Suspense>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}