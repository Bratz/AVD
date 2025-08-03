import { useState, useEffect } from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { getStoredUIMode, saveUIMode, getCurrentUIMode } from "@/utils/ui-mode";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

interface UIModeToggleProps {
  isCopilot?: boolean;
  compact?: boolean; // Add compact mode for header
}

export function UIModeToggle({ isCopilot = false, compact = false }: UIModeToggleProps) {
  const router = useRouter();
  const defaultMode = getCurrentUIMode(isCopilot);
  const [mode, setMode] = useState(defaultMode);
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
    const storedMode = getStoredUIMode(isCopilot);
    if (storedMode !== mode) {
      setMode(storedMode);
    }
  }, [isCopilot]);
  
  const handleToggle = (checked: boolean) => {
    const newMode = checked ? 'agentic' : 'traditional';
    setMode(newMode);
    saveUIMode(newMode, isCopilot);
    
    toast.success(
      `Switched to ${newMode === 'agentic' ? 'Agentic UI' : 'Traditional'} mode`,
      {
        description: 'Refreshing to apply changes...'
      }
    );
    
    setTimeout(() => {
      router.refresh();
      window.location.reload();
    }, 1000);
  };
  
  // Skeleton loader for better UX
  if (!mounted) {
    return (
      <div className={`
        flex items-center gap-2 
        ${compact ? 'p-1.5' : 'p-2'} 
        rounded-lg bg-muted/50 animate-pulse
      `}>
        <div className="h-4 w-16 bg-muted rounded" />
        <div className="h-5 w-9 bg-muted rounded-full" />
        <div className="h-4 w-16 bg-muted rounded" />
      </div>
    );
  }
  
  // Compact version for header
  if (compact) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/30 border border-border/50">
        <span className={`text-xs transition-all ${
          mode === 'traditional' 
            ? 'font-medium text-foreground' 
            : 'text-muted-foreground'
        }`}>
          Traditional
        </span>
        <Switch
          id="ui-mode-compact"
          checked={mode === 'agentic'}
          onCheckedChange={handleToggle}
          className="h-4 w-7 data-[state=checked]:bg-primary"
        />
        <span className={`text-xs transition-all ${
          mode === 'agentic' 
            ? 'font-medium text-foreground' 
            : 'text-muted-foreground'
        }`}>
          Agentic
        </span>
        {isCopilot && (
          <Badge variant="secondary" className="text-xs h-5 px-1.5">
            Copilot
          </Badge>
        )}
      </div>
    );
  }
  
  // Full version with enhanced styling
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative"
    >
      <div className="flex items-center gap-3 p-3 rounded-xl bg-card border border-border shadow-sm">
        <Label htmlFor="ui-mode" className="text-sm font-medium text-muted-foreground">
          UI Mode:
        </Label>
        
        <div className="flex items-center bg-muted/30 rounded-lg p-1">
          <motion.span
            animate={{
              opacity: mode === 'traditional' ? 1 : 0.5,
              scale: mode === 'traditional' ? 1 : 0.95
            }}
            className={`text-xs px-3 py-1.5 rounded-md transition-all cursor-pointer select-none ${
              mode === 'traditional' 
                ? 'bg-background shadow-sm font-semibold text-foreground' 
                : 'text-muted-foreground hover:text-foreground'
            }`}
            onClick={() => handleToggle(false)}
          >
            Traditional
          </motion.span>
          
          <Switch
            id="ui-mode"
            checked={mode === 'agentic'}
            onCheckedChange={handleToggle}
            className="mx-1.5 data-[state=checked]:bg-primary"
          />
          
          <motion.span
            animate={{
              opacity: mode === 'agentic' ? 1 : 0.5,
              scale: mode === 'agentic' ? 1 : 0.95
            }}
            className={`text-xs px-3 py-1.5 rounded-md transition-all cursor-pointer select-none ${
              mode === 'agentic' 
                ? 'bg-background shadow-sm font-semibold text-foreground' 
                : 'text-muted-foreground hover:text-foreground'
            }`}
            onClick={() => handleToggle(true)}
          >
            Agentic
          </motion.span>
        </div>
        
        {isCopilot && (
          <Badge variant="secondary" className="text-xs">
            Copilot
          </Badge>
        )}
        
        {/* Mode indicator dot */}
        <div className={`absolute -top-1 -right-1 h-2 w-2 rounded-full ${
          mode === 'agentic' ? 'bg-primary' : 'bg-muted-foreground'
        } animate-pulse`} />
      </div>
      
      {/* Tooltip on hover */}
      <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 opacity-0 hover:opacity-100 transition-opacity pointer-events-none">
        <div className="text-xs text-muted-foreground bg-popover px-2 py-1 rounded shadow-lg">
          {mode === 'agentic' 
            ? 'Using AI-powered components' 
            : 'Using traditional charts'}
        </div>
      </div>
    </motion.div>
  );
}