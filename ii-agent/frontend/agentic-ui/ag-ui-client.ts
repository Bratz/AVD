// ag-ui-client.ts - Minimal fix version
/**
 * AG-UI client implementation with minimal standardization fix
 * This version solves the naming inconsistency with minimal changes
 */

// Add TypeScript interface for window
declare global {
  interface Window {
    registeredComponents?: Record<string, any>
  }
}

// Store for registered tools
const registeredTools: Record<string, any> = {}

// Make registered tools available globally for debugging
if (typeof window !== "undefined") {
  window.registeredComponents = registeredTools
}

// Event emitter for AG-UI events
type Listener = (event: any) => void
const listeners: Listener[] = []

/**
 * Normalize component names to ensure consistency
 * This is the KEY CHANGE that fixes the naming issue
 */
function normalizeName(name: string): string {
  // Convert to lowercase and replace common variations
  return name
    .toLowerCase()
    .replace(/[_\s]/g, '-')  // Replace underscores and spaces with hyphens
    .replace(/([a-z])([A-Z])/g, '$1-$2')  // Convert camelCase to kebab-case
    .toLowerCase();  // Ensure all lowercase after camelCase conversion
}

/**
 * Register a tool with the AG-UI protocol
 * @param name Tool name (will be normalized)
 * @param component React component to render
 */
export function registerTool(name: string, component: any) {
  const normalizedName = normalizeName(name);
  
  // Register with normalized name
  registeredTools[normalizedName] = component;
  
  // ALSO register with original name for backward compatibility
  if (name !== normalizedName) {
    registeredTools[name.toLowerCase()] = component;
  }
  
  console.log(`Registered tool: ${normalizedName} (original: ${name})`);

  // Log total registered tools
  if (typeof window !== "undefined") {
    console.log(`Total registered tools: ${Object.keys(registeredTools).length}`);
    window.registeredComponents = registeredTools;
  }
}

/**
 * Get a registered tool by name
 * @param name Tool name (will be normalized)
 * @returns The registered component or undefined
 */
export function getTool(name: string) {
  // Try normalized name first
  const normalizedName = normalizeName(name);
  let component = registeredTools[normalizedName];
  
  // Fallback to lowercase version of original name
  if (!component) {
    component = registeredTools[name.toLowerCase()];
  }
  
  // Fallback to exact name (for backward compatibility)
  if (!component) {
    component = registeredTools[name];
  }

  if (!component) {
    console.warn(`Tool not found: ${name} (normalized: ${normalizedName})`);
    console.warn(`Available tools: ${Object.keys(registeredTools).join(", ")}`);
  }

  return component;
}

/**
 * Send a response back to the agent
 * @param response Response data
 */
export function sendResponse(response: any) {
  listeners.forEach((listener) =>
    listener({
      type: "agentResponse",
      payload: response,
    }),
  )
}

/**
 * Call a tool with arguments
 * @param toolCall Tool call data
 */
export function callTool(toolCall: { tool: string; args: any }) {
  listeners.forEach((listener) =>
    listener({
      type: "toolCall",
      payload: toolCall,
    }),
  )
}

/**
 * Subscribe to AG-UI events
 * @param listener Event listener function
 * @returns Unsubscribe function
 */
export function subscribe(listener: Listener) {
  listeners.push(listener)
  return () => {
    const index = listeners.indexOf(listener)
    if (index !== -1) {
      listeners.splice(index, 1)
    }
  }
}