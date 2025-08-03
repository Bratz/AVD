"use client";

import { RefObject } from "react";
import { cloneDeep, debounce } from "lodash";
import { toast } from "sonner";

import { useAppContext } from "@/context/app-context";
import { AgentEvent, TOOL, ActionStep, Message, TAB, PlanStep } from "@/typings/agent";
import { Terminal as XTerm } from "@xterm/xterm";
import { cleanContent, cleanAIResponse,extractThinkingContent } from '@/utils/content-cleaner';

// Store tool start times for duration calculation
const toolStartTimes = new Map<string, number>();

interface ErrorContent {
  message: string;
  error_type?: string;
  suggestions?: string[];
  [key: string]: unknown;
}

// Helper function to estimate tokens (rough approximation)
function estimateTokens(text: string | undefined): number {
  if (!text) return 0;
  // Rough estimate: 1 token ≈ 4 characters
  return Math.ceil(text.length / 4);
}

// Helper function to calculate success rate
function calculateSuccessRate(totalCalls: number, errorCount: number): number {
  if (totalCalls === 0) return 100;
  return Math.round(((totalCalls - errorCount) / totalCalls) * 100);
}

export function useAppEvents({
  xtermRef,
}: {
  xtermRef: RefObject<XTerm | null>;
}) {
  const { state, dispatch } = useAppContext();

  // Helper function to extract steps from text
  function extractStepsFromText(text: string): PlanStep[] {
    const steps: PlanStep[] = [];
    
    // Look for numbered lists or bullet points
    const lines = text.split('\n');
    const stepPatterns = [
      /^\d+\.\s+(.+)/, // 1. Step
      /^[-*]\s+(.+)/,  // - Step or * Step
      /^Step\s+\d+:\s+(.+)/i, // Step 1: ...
    ];
    
    let stepNumber = 1;
    lines.forEach((line, index) => {
      for (const pattern of stepPatterns) {
        const match = line.match(pattern);
        if (match) {
          steps.push({
            id: `step-${index}`,
            step: stepNumber++,
            description: match[1].trim(),
            status: "pending",
            timestamp: Date.now()
          });
          break;
        }
      }
    });
    
    return steps;
  }

  const handleEvent = (
    data: {
      id: string;
      type: AgentEvent;
      content: Record<string, unknown>;
    },
    workspacePath?: string
  ) => {
    // ALWAYS track events if debugging/monitoring is enabled
    if (state.debuggerEnabled || state.monitoringEnabled) {
      const timestamp = new Date().toISOString();
      let debugEvent: any = null;

      switch (data.type) {
        case AgentEvent.USER_MESSAGE:
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'user_message',
            content: data.content,
            status: 'success'
          };
          break;

        case AgentEvent.AGENT_THINKING:
          const thinkingTokens = estimateTokens(data.content.text as string);
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'agent_thinking',
            content: data.content,
            tokens: thinkingTokens,
            status: 'success'
          };
          dispatch({ 
            type: "UPDATE_METRICS", 
            payload: { 
              thoughtSteps: state.agentMetrics.thoughtSteps + 1,
              totalTokens: state.agentMetrics.totalTokens + thinkingTokens
            }
          });
          break;

        case AgentEvent.TOOL_CALL:
          const toolName = data.content.tool_name as string;
          // Store start time for duration calculation
          toolStartTimes.set(toolName + data.id, Date.now());
          
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'tool_call',
            content: data.content,
            toolName,
            status: 'pending'
          };
          dispatch({ 
            type: "UPDATE_METRICS", 
            payload: { 
              toolCallCount: state.agentMetrics.toolCallCount + 1 
            }
          });
          break;

        case AgentEvent.TOOL_RESULT:
          const resultToolName = data.content.tool_name as string;
          const toolKey = resultToolName + data.id;
          const startTime = toolStartTimes.get(toolKey);
          const duration = startTime ? Date.now() - startTime : 1000;
          toolStartTimes.delete(toolKey); // Clean up
          
          const isError = !!data.content.error || 
                         (typeof data.content.result === 'string' && 
                          data.content.result.toLowerCase().includes('error'));
          
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'tool_result',
            content: data.content,
            toolName: resultToolName,
            duration,
            status: isError ? 'error' : 'success'
          };
          
          // Update metrics
          const newErrorCount = isError ? state.agentMetrics.errorCount + 1 : state.agentMetrics.errorCount;
          const totalCalls = state.agentMetrics.toolCallCount || 1;
          
          dispatch({ 
            type: "UPDATE_METRICS", 
            payload: { 
              errorCount: newErrorCount,
              successRate: calculateSuccessRate(totalCalls, newErrorCount),
              totalDuration: state.agentMetrics.totalDuration + duration
            }
          });
          break;

        case AgentEvent.ERROR:
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'error',
            content: data.content,
            status: 'error'
          };
          dispatch({ 
            type: "UPDATE_METRICS", 
            payload: { 
              errorCount: state.agentMetrics.errorCount + 1,
              successRate: calculateSuccessRate(
                state.agentMetrics.toolCallCount,
                state.agentMetrics.errorCount + 1
              )
            }
          });
          break;

        case AgentEvent.AGENT_RESPONSE:
          const responseTokens = estimateTokens(data.content.text as string);
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'agent_response',
            content: data.content,
            tokens: responseTokens,
            status: 'success'
          };
          dispatch({ 
            type: "UPDATE_METRICS", 
            payload: { 
              totalTokens: state.agentMetrics.totalTokens + responseTokens
            }
          });
          break;

        case AgentEvent.PROCESSING:
        case 'COMPLETED' as any:
        case 'STOPPED' as any:
        case AgentEvent.PROMPT_GENERATED:
        case AgentEvent.WORKSPACE_INFO:
        case AgentEvent.FILE_EDIT:
        case AgentEvent.BROWSER_USE:
        case AgentEvent.AGENT_INITIALIZED:
        case AgentEvent.SYSTEM:
        case AgentEvent.UPLOAD_SUCCESS:
          // Track these as agent_response events
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'agent_response',
            content: data.content,
            status: 'success'
          };
          break;

        default:
          // Track any unknown events as agent_response
          debugEvent = {
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp,
            type: 'agent_response',
            content: data.content,
            status: 'success'
          };
          break;
      }

      // Dispatch the debug event if one was created
      if (debugEvent) {
        dispatch({ type: "ADD_DEBUG_EVENT", payload: debugEvent });
      }
    }

    // Handle the actual event logic
    switch (data.type) {
      case AgentEvent.USER_MESSAGE:
        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: data.id,
            role: "user",
            content: data.content.text as string,
            timestamp: Date.now(),
          },
        });
        break;

      case AgentEvent.PROMPT_GENERATED:
        dispatch({ type: "SET_GENERATING_PROMPT", payload: false });
        dispatch({
          type: "SET_CURRENT_QUESTION",
          payload: data.content.result as string,
        });
        break;

      case 'COMPLETED' as any:
        dispatch({ type: "SET_COMPLETED", payload: true });
        dispatch({ type: "SET_LOADING", payload: false });
        
        // Update any active plan to show completion
        const completedPlanMessage = state.messages
          .filter(m => m.plan)
          .reverse()[0];
          
        if (completedPlanMessage?.plan) {
          const updatedPlan = { ...completedPlanMessage.plan };
          // Mark all remaining pending steps as completed
          updatedPlan.steps = updatedPlan.steps.map(step => 
            step.status === 'pending' ? { ...step, status: 'completed' as const } : step
          );
          
          dispatch({
            type: "UPDATE_MESSAGE", 
            payload: {
              ...completedPlanMessage,
              plan: updatedPlan
            }
          });
        }
        break;

      case 'STOPPED' as any:
        dispatch({ type: "SET_STOPPED", payload: true });
        dispatch({ type: "SET_LOADING", payload: false });
        
        // Show a message that the agent was stopped
        toast.info("Agent execution stopped by user");
        
        // Update any active plan to show it was stopped
        const stoppedPlanMessage = state.messages
          .filter(m => m.plan)
          .reverse()[0];
          
        if (stoppedPlanMessage?.plan) {
          const updatedPlan = { ...stoppedPlanMessage.plan };
          // Mark current step as failed and remaining as pending
          if (updatedPlan.currentStep !== undefined && updatedPlan.currentStep < updatedPlan.steps.length) {
            updatedPlan.steps[updatedPlan.currentStep].status = 'failed';
          }
          
          dispatch({
            type: "UPDATE_MESSAGE",
            payload: {
              ...stoppedPlanMessage,
              plan: updatedPlan
            }
          });
        }
        break;

      case AgentEvent.PROCESSING:
        dispatch({ type: "SET_LOADING", payload: true });
        break;

      case AgentEvent.WORKSPACE_INFO:
        dispatch({
          type: "SET_WORKSPACE_INFO",
          payload: data.content.path as string,
        });
        break;

      case AgentEvent.AGENT_THINKING:
        // Handle the type properly - content could be string or Record<string, unknown>
        let thinkingText: string;
        
        if (typeof data.content === 'string') {
          thinkingText = data.content;
        } else if (data.content && typeof data.content === 'object' && 'text' in data.content) {
          thinkingText = data.content.text as string;
        } else {
          // Skip if we can't extract text
          break;
        }
        
        // Extract thinking content (content within <think> tags)
        const thinkingContent = extractThinkingContent(thinkingText);
        
        // Extract cleaned content (content without <think> tags)
        const cleanedContent = cleanContent(thinkingText);
        
        // Check if this is a plan
        const planMatch = thinkingText.match(/(?:plan|steps|approach):/i);
        if (planMatch) {
          // Extract steps from the thinking text
          const steps = extractStepsFromText(thinkingText);
          if (steps.length > 0) {
            const planMessage: Message = {
              id: data.id || Date.now().toString(),
              role: "assistant",
              timestamp: Date.now(),
              plan: {
                steps: steps,
                currentStep: 0
              }
            };
            dispatch({ type: "ADD_MESSAGE", payload: planMessage });
            
            // Optionally, don't add to thinking messages if it's a plan
            // to avoid duplication
            break;
          }
        }
        
        // Add cleaned content to main chat if it exists
        if (cleanedContent && cleanedContent.trim().length > 0) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: data.id || Date.now().toString(),
              role: "assistant",
              content: cleanedContent,
              timestamp: Date.now()
            }
          });
        }
        
        // Add the full thinking content to thinking messages
        if (thinkingContent) {
          dispatch({
            type: "ADD_THINKING_MESSAGE",
            payload: {
              id: data.id || Date.now().toString(),
              content: thinkingContent,
              timestamp: Date.now()
            }
          });
        }
        break;

      case AgentEvent.TOOL_CALL:
        // Update plan step status if there's an active plan
        const lastPlanMessage = state.messages
          .filter(m => m.plan)
          .reverse()[0];
          
        if (lastPlanMessage?.plan) {
          const toolName = data.content.tool_name as string;
          const updatedPlan = { ...lastPlanMessage.plan };
          
          // Find step related to this tool
          let stepIndex = updatedPlan.steps.findIndex(
            step => step.tool === toolName || step.description.toLowerCase().includes(toolName.toLowerCase())
          );
          if (stepIndex === -1) {
            stepIndex = updatedPlan.steps.findIndex(step => step.status === "pending");
          }
          if (stepIndex >= 0) {
            updatedPlan.steps[stepIndex].status = "in-progress";
            updatedPlan.steps[stepIndex].tool = toolName;
            updatedPlan.currentStep = stepIndex;
            
            dispatch({
              type: "UPDATE_MESSAGE",
              payload: {
                ...lastPlanMessage,
                plan: updatedPlan
              }
            });
          }
        }
        
        if (data.content.tool_name === TOOL.SEQUENTIAL_THINKING) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: data.id,
              role: "assistant",
              content: (data.content.tool_input as { thought: string })
                .thought as string,
              timestamp: Date.now(),
            },
          });
        } else if (data.content.tool_name === TOOL.MESSAGE_USER) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: data.id,
              role: "assistant",
              content: (data.content.tool_input as { text: string })
                .text as string,
              timestamp: Date.now(),
            },
          });
        } else {
          const message: Message = {
            id: data.id,
            role: "assistant",
            action: {
              type: data.content.tool_name as TOOL,
              data: data.content,
            },
            timestamp: Date.now(),
          };
          const toolInput = data.content.tool_input as any;
          if (toolInput?.thought || toolInput?.text || toolInput?.message) {
            message.content = toolInput.thought || toolInput.text || toolInput.message;
          }

          const url = (data.content.tool_input as { url: string })
            ?.url as string;
          if (url) {
            dispatch({ type: "SET_BROWSER_URL", payload: url });
          }
          dispatch({ type: "ADD_MESSAGE", payload: message });
          handleClickAction(message.action);
        }
        break;
      
      case AgentEvent.AGENT_INITIALIZED:
        // Handle agent initialization success
        dispatch({ type: "SET_LOADING", payload: false });
        
        // Log the initialization details
        console.log("Agent initialized:", data.content);
        
        // You might want to show a success toast
        toast.success(`Agent initialized with ${data.content.model}`);
        
        // Store any relevant info like tool summary
        if (data.content.tool_summary) {
          // You could add a new state to track available tools
          console.log("Available tools:", data.content.tool_summary);
        }
        break;

      case AgentEvent.SYSTEM:
        // Handle various system messages based on flags
        const systemContent = data.content;
        
        if (systemContent.status_update) {
          // Handle status updates
          console.log("Agent status:", systemContent.status);
        } else if (systemContent.thought_trail) {
          // Handle thought trail updates
          console.log("Thought trail:", systemContent.thoughts);
        } else if (systemContent.tool_list) {
          // Handle tool list
          console.log("Available tools:", systemContent.tools);
        } else if (systemContent.mcp_prompt) {
          // Handle MCP prompt responses
          console.log("MCP prompt result:", systemContent.prompt_content);
        } else if (systemContent.warning) {
          // Handle warnings (like MCP initialization failures)
          const message = systemContent.message as string;
          toast.info(message, {duration: 10000});
        } else {
          // General system message
          console.log("System message:", systemContent.message);
        }
        break;
      
      case AgentEvent.ERROR:
        // Enhanced error handling
        const errorContent = data.content as ErrorContent;
        
        // Show the main error message
        toast.error(errorContent.message);
        
        // If there are suggestions, show them
        if (errorContent.suggestions && Array.isArray(errorContent.suggestions)) {
          errorContent.suggestions.forEach((suggestion: string) => {
            toast.info(suggestion, { duration: 10000 });
          });
        }
        
        // Handle specific error types
        switch (errorContent.error_type) {
          case "agent_not_initialized":
            // Could trigger agent initialization UI
            break;
          case "ollama_oom":
          case "context_exceeded":
            // Could show model switching options
            break;
          case "mcp_not_initialized":
            // Could show MCP setup options
            break;
        }
        
        dispatch({ type: "SET_LOADING", payload: false });
        break;

      case AgentEvent.FILE_EDIT:
        const messages = [...state.messages];
        const lastMessage = cloneDeep(messages[messages.length - 1]);

        if (
          lastMessage?.action &&
          lastMessage.action.type === TOOL.STR_REPLACE_EDITOR
        ) {
          lastMessage.action.data.content = data.content.content as string;
          lastMessage.action.data.path = data.content.path as string;
          const workspace = workspacePath || state.workspaceInfo;
          const filePath = (data.content.path as string)?.includes(workspace)
            ? (data.content.path as string)
            : `${workspace}/${data.content.path}`;

          dispatch({
            type: "ADD_FILE_CONTENT",
            payload: {
              path: filePath,
              content: data.content.content as string,
            },
          });

          setTimeout(() => {
            handleClickAction(lastMessage.action);
          }, 500);

          dispatch({
            type: "UPDATE_MESSAGE",
            payload: lastMessage,
          });
        }
        break;

      case AgentEvent.BROWSER_USE:
        // Handle browser automation events
        const browserContent = data.content;
        
        // If there's a result, add it as a message
        if (browserContent.result) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: data.id,
              role: "assistant",
              content: browserContent.result as string,
              timestamp: Date.now(),
            },
          });
        }
        
        // Update the browser tab
        dispatch({ type: "SET_ACTIVE_TAB", payload: TAB.BROWSER });
        
        // If there's a screenshot or specific action, handle it
        if (browserContent.screenshot || browserContent.action) {
          dispatch({ 
            type: "SET_CURRENT_ACTION_DATA", 
            payload: {
              type: TOOL.BROWSER_USE,
              data: browserContent
            }
          });
        }
        break;

      case AgentEvent.TOOL_RESULT:
        // FIRST: Update plan step status when tool completes
        const lastPlanMessageForResult = state.messages
          .filter(m => m.plan)
          .reverse()[0];
          
        if (lastPlanMessageForResult?.plan) {
          const toolNameResult = data.content.tool_name as string;
          const toolResult = data.content.result;
          const updatedPlanResult = { ...lastPlanMessageForResult.plan };
          
          // Find step that's currently in-progress or matches this tool
          const stepIndexResult = updatedPlanResult.steps.findIndex(
            step => step.status === "in-progress" || 
                    step.tool === toolNameResult || 
                    step.description.toLowerCase().includes(toolNameResult.toLowerCase())
          );
          
          if (stepIndexResult >= 0) {
            // Determine if the tool succeeded or failed
            const resultStr = String(toolResult || '');
            const isError = resultStr.toLowerCase().includes('error') || 
                          resultStr.toLowerCase().includes('failed') ||
                          resultStr.toLowerCase().includes('exception');
            const isSuccess = !isError && toolResult !== null && toolResult !== undefined;
            
            updatedPlanResult.steps[stepIndexResult].status = isSuccess ? "completed" : "failed";
            updatedPlanResult.steps[stepIndexResult].tool = toolNameResult;
            
            // Move to next step if current step completed successfully
            if (isSuccess && stepIndexResult < updatedPlanResult.steps.length - 1) {
              updatedPlanResult.currentStep = stepIndexResult + 1;
            }
            
            dispatch({
              type: "UPDATE_MESSAGE",
              payload: {
                ...lastPlanMessageForResult,
                plan: updatedPlanResult
              }
            });
          }
        }

        // SPECIAL HANDLING FOR CHART GENERATION
        /*
        if (data.content.tool_name === 'generate_chart' || data.content.tool_name === TOOL.GENERATE_CHART) {
          console.log('[TOOL_RESULT] Chart generation result:', data.content.result);
          
          // The chart tool result contains the formatted chart block
          if (data.content.result && typeof data.content.result === 'string') {
            // Add the chart as a message
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: data.id || Date.now().toString(),
                role: "assistant",
                content: data.content.result as string,
                timestamp: Date.now(),
              },
            });
          }
          break; // Exit early for chart results
        }*/

        // EXISTING TOOL RESULT HANDLING
        if (data.content.tool_name === TOOL.BROWSER_USE) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: data.id,
              role: "assistant",
              content: data.content.result as string,
              timestamp: Date.now(),
            },
          });
        } else {
          if (
            data.content.tool_name !== TOOL.SEQUENTIAL_THINKING &&
            data.content.tool_name !== TOOL.PRESENTATION &&
            data.content.tool_name !== TOOL.MESSAGE_USER &&
            data.content.tool_name !== TOOL.RETURN_CONTROL_TO_USER
          ) {
            const messages = [...state.messages];
            const lastMessage = cloneDeep(messages[messages.length - 1]);

            if (
              lastMessage?.action &&
              lastMessage.action?.type === data.content.tool_name
            ) {
              lastMessage.action.data.result = `${data.content.result}`;
              if (
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
                ].includes(data.content.tool_name as TOOL)
              ) {
                lastMessage.action.data.result =
                  data.content.result && Array.isArray(data.content.result)
                    ? data.content.result.find((item) => item.type === "image")
                        ?.source?.data
                    : undefined;
              }
              lastMessage.action.data.isResult = true;
              setTimeout(() => {
                handleClickAction(lastMessage.action);
              }, 500);

              dispatch({
                type: "UPDATE_MESSAGE",
                payload: lastMessage,
              });
            } else {
              dispatch({
                type: "ADD_MESSAGE",
                payload: { ...lastMessage, action: data.content as ActionStep },
              });
            }
          }
        }
        break;      
      case AgentEvent.AGENT_RESPONSE:
        const responseText = data.content.text as string;
        const cleanedResponse = cleanAIResponse(responseText);
        
        if (cleanedResponse && cleanedResponse.trim().length > 0) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: Date.now().toString(),
              role: "assistant",
              content: cleanedResponse,
              timestamp: Date.now(),
            },
          });
        }
        dispatch({ type: "SET_COMPLETED", payload: true });
        dispatch({ type: "SET_LOADING", payload: false });
        break;

      case AgentEvent.UPLOAD_SUCCESS:
        dispatch({ type: "SET_IS_UPLOADING", payload: false });

        // Update the uploaded files state
        const newFiles = data.content.files as {
          path: string;
          saved_path: string;
        }[];
        const paths = newFiles.map((f) => f.path);
        dispatch({ type: "ADD_UPLOADED_FILES", payload: paths });
        break;

      case "error" as any:
        toast.error(data.content.message as string);
        dispatch({ type: "SET_IS_UPLOADING", payload: false });
        dispatch({ type: "SET_LOADING", payload: false });
        dispatch({ type: "SET_GENERATING_PROMPT", payload: false });
        break;

      default:
        console.warn(`Unhandled event type: ${data.type}`, data);
        break;
    }
  };

  const handleClickAction = debounce(
    (data: ActionStep | undefined, showTabOnly = false) => {
      if (!data) return;

      switch (data.type) {
        case TOOL.WEB_SEARCH:
          dispatch({ type: "SET_ACTIVE_TAB", payload: TAB.BROWSER });
          dispatch({ type: "SET_CURRENT_ACTION_DATA", payload: data });
          break;

        case TOOL.IMAGE_GENERATE:
        case TOOL.IMAGE_SEARCH:
        case TOOL.BROWSER_USE:
        case TOOL.VISIT:
          dispatch({ type: "SET_ACTIVE_TAB", payload: TAB.BROWSER });
          dispatch({ type: "SET_CURRENT_ACTION_DATA", payload: data });
          break;

        case TOOL.BROWSER_CLICK:
        case TOOL.BROWSER_ENTER_TEXT:
        case TOOL.BROWSER_PRESS_KEY:
        case TOOL.BROWSER_GET_SELECT_OPTIONS:
        case TOOL.BROWSER_SELECT_DROPDOWN_OPTION:
        case TOOL.BROWSER_SWITCH_TAB:
        case TOOL.BROWSER_OPEN_NEW_TAB:
        case TOOL.BROWSER_VIEW:
        case TOOL.BROWSER_NAVIGATION:
        case TOOL.BROWSER_RESTART:
        case TOOL.BROWSER_WAIT:
        case TOOL.BROWSER_SCROLL_DOWN:
        case TOOL.BROWSER_SCROLL_UP:
          dispatch({ type: "SET_ACTIVE_TAB", payload: TAB.BROWSER });
          dispatch({ type: "SET_CURRENT_ACTION_DATA", payload: data });
          break;

        case TOOL.BASH:
          dispatch({ type: "SET_ACTIVE_TAB", payload: TAB.TERMINAL });
          if (!showTabOnly) {
            setTimeout(() => {
              if (!data.data?.isResult) {
                // query
                xtermRef?.current?.writeln(
                  `${data.data.tool_input?.command || ""}`
                );
              }
              // result
              if (data.data.result) {
                const lines = `${data.data.result || ""}`.split("\n");
                lines.forEach((line) => {
                  xtermRef?.current?.writeln(line);
                });
                xtermRef?.current?.write("$ ");
              }
            }, 500);
          }
          break;

        case TOOL.STR_REPLACE_EDITOR:
          dispatch({ type: "SET_ACTIVE_TAB", payload: TAB.CODE });
          dispatch({ type: "SET_CURRENT_ACTION_DATA", payload: data });
          const path = data.data.tool_input?.path || data.data.tool_input?.file;
          if (path) {
            dispatch({
              type: "SET_ACTIVE_FILE",
              payload: path.startsWith(state.workspaceInfo)
                ? path
                : `${state.workspaceInfo}/${path}`,
            });
          }
          break;

        default:
          break;
      }
    },
    50
  );

  return { handleEvent, handleClickAction };
}