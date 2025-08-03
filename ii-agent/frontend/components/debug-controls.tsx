import { Bug, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAppContext } from "@/context/app-context";

export function DebugControls() {
  const { state, dispatch } = useAppContext();

  return (
    <div className="fixed top-20 left-4 flex flex-col gap-2 z-40">
      <Button
        variant="outline"
        size="icon"
        onClick={() => dispatch({ type: "TOGGLE_DEBUGGER", payload: !state.debuggerEnabled })}
        className={state.debuggerEnabled ? "bg-purple-500/20 border-purple-500" : ""}
        title="Toggle Debugger"
      >
        <Bug className="h-5 w-5" />
      </Button>
      
      <Button
        variant="outline"
        size="icon"
        onClick={() => dispatch({ type: "TOGGLE_MONITORING", payload: !state.monitoringEnabled })}
        className={state.monitoringEnabled ? "bg-blue-500/20 border-blue-500" : ""}
        title="Toggle Monitoring"
      >
        <BarChart3 className="h-5 w-5" />
      </Button>
    </div>
  );
}