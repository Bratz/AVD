// providers/index.tsx
"use client";

import { AppProgressBar as ProgressBar } from "next-nprogress-bar";
import { Toaster } from "@/components/ui/sonner";
import { useEffect } from "react";
import "../app/github-markdown.css";
import { ThemeProvider } from "@/components/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";
import { registerAllAgentComponents } from "@/agentic-ui/registry";
import { getCurrentUIMode } from "@/utils/ui-mode";

export default function Providers({ children }: { children: React.ReactNode }) {
  // Register components based on UI mode
  useEffect(() => {
    // Detect if we're in copilot mode from URL or other context
    const isCopilot = typeof window !== 'undefined' && 
      (window.location.pathname.includes('copilot') || 
       window.location.search.includes('copilot=true'));
    
    // Get the current UI mode
    const uiMode = getCurrentUIMode(isCopilot);
    
    console.log('=== UI Initialization ===');
    console.log('Current UI Mode:', uiMode);
    console.log('Is Copilot:', isCopilot);
    console.log('Agentic UI Enabled:', process.env.NEXT_PUBLIC_ENABLE_AGENTIC_UI);
    
    if (uiMode === 'agentic') {
      console.log("Initializing Agentic UI components...");
      registerAllAgentComponents();
      
      // Verify registration in development
      if (process.env.NODE_ENV === 'development') {
        setTimeout(() => {
          if (typeof window !== 'undefined' && window.registeredComponents) {
            const componentCount = Object.keys(window.registeredComponents).length;
            console.log(`✅ Registered ${componentCount} AG-UI components`);
            console.log('Available components:', Object.keys(window.registeredComponents));
          }
        }, 100);
      }
    } else {
      console.log("Running in traditional chart mode - AG-UI components not registered");
    }
    
    // Store current mode in window for debugging
    if (typeof window !== 'undefined') {
      (window as any).currentUIMode = uiMode;
      (window as any).isCopilotMode = isCopilot;
    }
  }, []);

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      themes={["light", "dark"]}
      disableTransitionOnChange
    >
      <TooltipProvider>
        <ProgressBar
          height="2px"
          color="#BAE9F4"
          options={{ showSpinner: false }}
          shallowRouting
        />
        {children}
      </TooltipProvider>
      <Toaster richColors />
    </ThemeProvider>
  );
}