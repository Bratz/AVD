// components/plan-display.tsx
import { motion } from "framer-motion";
import { CheckCircle2, Circle, Loader2, XCircle, ListChecks, ChevronDown } from "lucide-react";
import { StepPlan, PlanStep } from "@/typings/agent";

interface PlanDisplayProps {
  plan: StepPlan;
  isCollapsed?: boolean;
  onToggle?: () => void;
}

export function PlanDisplay({ plan, isCollapsed = false, onToggle }: PlanDisplayProps) {
  const steps = plan.steps;
  
  const getStepIcon = (status: PlanStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'in-progress':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Circle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStepStatusColor = (status: PlanStep['status']) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-950 dark:border-green-800';
      case 'in-progress':
        return 'text-blue-600 bg-blue-50 border-blue-200 dark:text-blue-400 dark:bg-blue-950 dark:border-blue-800';
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-950 dark:border-red-800';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-800 dark:border-gray-600';
    }
  };

  const completedCount = steps.filter(s => s.status === 'completed').length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card text-card-foreground rounded-lg p-3 mb-4 border border-border shadow-sm"
    >
      <div 
        className="flex items-center justify-between cursor-pointer p-2 -m-2 rounded-md bg-background/90 border-b border-border"
        onClick={onToggle}
      >
        <div className="flex items-center gap-2">
          <ListChecks className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-foreground">
            Agent Thinking ({completedCount}/{steps.length} completed)
          </span>
        </div>
        <motion.div
          animate={{ rotate: isCollapsed ? 0 : 180 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </motion.div>
      </div>

      {!isCollapsed && (
        <motion.div
          initial={{ height: 0 }}
          animate={{ height: "auto" }}
          exit={{ height: 0 }}
          className="mt-3 space-y-2 overflow-hidden"
        >
          {steps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`flex items-start gap-2 text-sm p-2 rounded-md border transition-colors ${getStepStatusColor(step.status)}`}
            >
              <div className="mt-0.5 flex-shrink-0">{getStepIcon(step.status || 'pending')}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium">
                    {step.step}. {step.description}
                  </span>
                  
                  {/* Status indicator */}
                  <span className="text-xs px-2 py-0.5 rounded-full bg-white/50 dark:bg-black/50">
                    {step.status || 'pending'}
                  </span>
                </div>
                
                {/* Tool indicator */}
                {step.tool && (
                  <div className="mt-1">
                    <span className="text-xs px-1.5 py-0.5 rounded bg-black/10 dark:bg-white/10">
                      Tool: {step.tool}
                    </span>
                  </div>
                )}
                
                {/* Timestamp */}
                <div className="text-xs opacity-70 mt-1">
                  {new Date(step.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* Progress indicator */}
      <div className="mt-3 pt-2 border-t border-border">
        <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
          <span>Progress</span>
          <span>{Math.round((completedCount / steps.length) * 100)}%</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${(completedCount / steps.length) * 100}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="bg-green-500 h-2 rounded-full"
          />
        </div>
      </div>
    </motion.div>
  );
}