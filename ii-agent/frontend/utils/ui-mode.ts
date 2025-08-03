export type UIMode = 'agentic' | 'traditional';

export interface UIModeConfig {
  mode: UIMode;
  enableAgenticUI: boolean;
  copilotMode: UIMode;
  forceTraditionalCharts: boolean;
}

/**
 * Get UI mode configuration from environment variables
 */
export function getUIModeConfig(): UIModeConfig {
  return {
    mode: (process.env.NEXT_PUBLIC_UI_MODE as UIMode) || 'agentic',
    enableAgenticUI: process.env.NEXT_PUBLIC_ENABLE_AGENTIC_UI === 'true',
    copilotMode: (process.env.NEXT_PUBLIC_COPILOT_UI_MODE as UIMode) || 'agentic',
    forceTraditionalCharts: process.env.COPILOT_FORCE_TRADITIONAL_CHARTS === 'true',
  };
}

/**
 * Determine which UI mode to use based on context
 */
export function getCurrentUIMode(isCopilot: boolean = false): UIMode {
  const config = getUIModeConfig();
  
  // Force traditional if flag is set
  if (config.forceTraditionalCharts) {
    return 'traditional';
  }
  
  // Check if AG-UI is disabled globally
  if (!config.enableAgenticUI) {
    return 'traditional';
  }
  
  // Use copilot-specific mode if in copilot
  if (isCopilot && process.env.NEXT_PUBLIC_COPILOT_ENABLED === 'true') {
    return config.copilotMode;
  }
  
  // Use default mode
  return config.mode;
}

/**
 * Check if we should use AG-UI components
 */
export function shouldUseAgenticUI(isCopilot: boolean = false): boolean {
  return getCurrentUIMode(isCopilot) === 'agentic';
}

/**
 * Get UI mode from localStorage with fallback to env
 */
export function getStoredUIMode(isCopilot: boolean = false): UIMode {
  // Always return default on server
  if (typeof window === 'undefined') {
    return getCurrentUIMode(isCopilot);
  }
  
  try {
    const storageKey = isCopilot ? 'copilot-ui-mode' : 'ui-mode';
    const stored = localStorage.getItem(storageKey);
    
    if (stored === 'agentic' || stored === 'traditional') {
      return stored as UIMode;
    }
  } catch (e) {
    // localStorage might throw in some environments
    console.warn('Failed to access localStorage:', e);
  }
  
  return getCurrentUIMode(isCopilot);
}

/**
 * Save UI mode preference to localStorage
 */
export function saveUIMode(mode: UIMode, isCopilot: boolean = false): void {
  if (typeof window === 'undefined') return;
  
  try {
    const storageKey = isCopilot ? 'copilot-ui-mode' : 'ui-mode';
    localStorage.setItem(storageKey, mode);
  } catch (e) {
    console.warn('Failed to save to localStorage:', e);
  }
}