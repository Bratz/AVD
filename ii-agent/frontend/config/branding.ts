/**
 * Central branding configuration for the application
 * This file contains all branding-related constants and settings
 */

export interface BrandingConfig {
  // Main application name
  appName: string;
  
  // Short name for compact displays
  appShortName: string;
  
  // Agent system name
  agentName: string;
  
  // Description for metadata
  appDescription: string;
  
  // Logo paths
  logos: {
    main: string;
    icon: string;
    favicon: string;
  };
  
  // UI text templates
  messages: {
    welcome: string;
    taskCompleted: string;
    taskStopped: string;
    connectionEstablished: string;
  };
  
  // File system references
  filesystem: {
    rootFolderName: string;
    workspaceName: string;
  };
}

// Default branding configuration
export const defaultBranding: BrandingConfig = {
  appName: "chatGBP",
  appShortName: "chatGBP",
  agentName: "II-Agent",
  appDescription: "chatGBP is a tool for in-depth analysis and research powered by II-Agent.",
  
  logos: {
    main: "/logo-only.png",
    icon: "/logo-only.png",
    favicon: "/favicon",
  },
  
  messages: {
    welcome: "Welcome to II-Agent!",
    taskCompleted: "II-Agent has completed the current task.",
    taskStopped: "II-Agent has stopped, send a new message to continue.",
    connectionEstablished: "Connected to Agent WebSocket Server",
  },
  
  filesystem: {
    rootFolderName: "ii-agent",
    workspaceName: "II-Agent Workspace",
  },
};

// Allow runtime configuration through environment variables
export const brandingConfig: BrandingConfig = {
  appName: process.env.NEXT_PUBLIC_APP_NAME || defaultBranding.appName,
  appShortName: process.env.NEXT_PUBLIC_APP_SHORT_NAME || defaultBranding.appShortName,
  agentName: process.env.NEXT_PUBLIC_AGENT_NAME || defaultBranding.agentName,
  appDescription: process.env.NEXT_PUBLIC_APP_DESCRIPTION || defaultBranding.appDescription,
  
  logos: {
    main: process.env.NEXT_PUBLIC_LOGO_MAIN || defaultBranding.logos.main,
    icon: process.env.NEXT_PUBLIC_LOGO_ICON || defaultBranding.logos.icon,
    favicon: process.env.NEXT_PUBLIC_FAVICON || defaultBranding.logos.favicon,
  },
  
  messages: {
    welcome: process.env.NEXT_PUBLIC_WELCOME_MESSAGE || defaultBranding.messages.welcome,
    taskCompleted: process.env.NEXT_PUBLIC_TASK_COMPLETED_MESSAGE || defaultBranding.messages.taskCompleted,
    taskStopped: process.env.NEXT_PUBLIC_TASK_STOPPED_MESSAGE || defaultBranding.messages.taskStopped,
    connectionEstablished: process.env.NEXT_PUBLIC_CONNECTION_MESSAGE || defaultBranding.messages.connectionEstablished,
  },
  
  filesystem: {
    rootFolderName: process.env.NEXT_PUBLIC_ROOT_FOLDER_NAME || defaultBranding.filesystem.rootFolderName,
    workspaceName: process.env.NEXT_PUBLIC_WORKSPACE_NAME || defaultBranding.filesystem.workspaceName,
  },
};

// Export a hook for React components
export function useBranding() {
  return brandingConfig;
}

// Export utility functions
export function getAppTitle(includeAgent: boolean = false): string {
  if (includeAgent && brandingConfig.appName !== brandingConfig.agentName) {
    return `${brandingConfig.appName} - Powered by ${brandingConfig.agentName}`;
  }
  return brandingConfig.appName;
}

export function getLogoAlt(): string {
  return `${brandingConfig.agentName} Logo`;
}