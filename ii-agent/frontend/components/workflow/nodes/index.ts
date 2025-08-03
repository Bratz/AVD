// components/workflow/nodes/index.ts
import { AgentNode } from './AgentNode';
import { StartNode } from './StartNode';

export const nodeTypes = {
  agent: AgentNode,
  start: StartNode,
};

export { AgentNode, StartNode };