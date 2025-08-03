// utils/ag-ui-websocket-handler.ts
import { getTool } from '../agentic-ui/ag-ui-client';

interface AGUIMessage {
  type: string;
  content: {
    component?: string;
    props?: any;
    action?: string;
    response?: any;
  };
}

export function handleAGUIWebSocketMessage(message: AGUIMessage) {
  switch (message.type) {
    case 'render_component':
      return {
        type: 'renderComponent',
        payload: {
          componentName: message.content.component,
          props: message.content.props
        }
      };
      
    case 'agent_response':
      return {
        type: 'agentResponse',
        payload: message.content.response
      };
      
    case 'tool_call':
      return {
        type: 'toolCall',
        payload: {
          tool: message.content.action,
          args: message.content.props
        }
      };
      
    default:
      return null;
  }
}