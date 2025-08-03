// Create components/plan-indicator.tsx
import { ListChecks } from "lucide-react";
import { StepPlan } from "@/typings/agent";

interface PlanIndicatorProps {
  plan: StepPlan;
}

export function PlanIndicator({ plan }: PlanIndicatorProps) {
  const completed = plan.steps.filter(s => s.status === "completed").length;
  const total = plan.steps.length;
  const percentage = (completed / total) * 100;
  
  return (
    <div className="inline-flex items-center gap-2 bg-secondary px-3 py-1 rounded-full text-xs">
      <ListChecks className="h-3 w-3" />
      <span>{completed}/{total} steps</span>
      <div className="w-16 h-1.5 bg-gray-600 rounded-full overflow-hidden">
        <div 
          className="h-full bg-blue-400 transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}