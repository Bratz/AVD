// types/ag-ui.d.ts
export interface AGUIEvent {
  type: 'agentResponse' | 'toolCall' | 'renderComponent';
  payload: any;
}

export interface AGUIToolCall {
  tool: string;
  args: any;
}

export interface AGUIRenderComponent {
  componentName: string;
  props: any;
}

export interface AGUIComponent {
  name: string;
  component: React.ComponentType<any>;
}