"""
Metrics collection utility for ROWBOAT workflows
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import aiofiles
import asyncio
from pathlib import Path

class MetricsCollector:
    """Collect and persist workflow execution metrics"""
    
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._write_lock = asyncio.Lock()
    
    async def record_execution(
        self,
        workflow_id: str,
        execution_id: str,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        agent_metrics: Dict[str, Any],
        error: Optional[str] = None
    ):
        """Record a workflow execution"""
        
        metrics_file = self.metrics_dir / f"{workflow_id}_metrics.json"
        
        execution_record = {
            "execution_id": execution_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_ms": int((end_time - start_time).total_seconds() * 1000),
            "success": success,
            "error": error,
            "agent_metrics": agent_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with self._write_lock:
            # Load existing metrics
            if metrics_file.exists():
                async with aiofiles.open(metrics_file, 'r') as f:
                    metrics = json.loads(await f.read())
            else:
                metrics = {
                    "workflow_id": workflow_id,
                    "executions": [],
                    "summary": {
                        "total_executions": 0,
                        "successful_executions": 0,
                        "failed_executions": 0,
                        "average_duration_ms": 0
                    }
                }
            
            # Add execution record
            metrics["executions"].append(execution_record)
            
            # Update summary
            summary = metrics["summary"]
            summary["total_executions"] += 1
            if success:
                summary["successful_executions"] += 1
            else:
                summary["failed_executions"] += 1
            
            # Update average duration
            total_duration = sum(e["duration_ms"] for e in metrics["executions"])
            summary["average_duration_ms"] = total_duration / len(metrics["executions"])
            
            # Keep only last 1000 executions
            if len(metrics["executions"]) > 1000:
                metrics["executions"] = metrics["executions"][-1000:]
            
            # Write back
            async with aiofiles.open(metrics_file, 'w') as f:
                await f.write(json.dumps(metrics, indent=2))
            
            # Update cache
            self._metrics_cache[workflow_id] = metrics
    
    async def get_metrics(
        self,
        workflow_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get metrics for a workflow"""
        
        # Check cache first
        if workflow_id in self._metrics_cache:
            metrics = self._metrics_cache[workflow_id]
        else:
            metrics_file = self.metrics_dir / f"{workflow_id}_metrics.json"
            if metrics_file.exists():
                async with aiofiles.open(metrics_file, 'r') as f:
                    metrics = json.loads(await f.read())
                    self._metrics_cache[workflow_id] = metrics
            else:
                return {
                    "workflow_id": workflow_id,
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "average_duration_ms": 0,
                    "executions": []
                }
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_executions = []
            for execution in metrics["executions"]:
                exec_time = datetime.fromisoformat(execution["timestamp"])
                if start_date and exec_time < start_date:
                    continue
                if end_date and exec_time > end_date:
                    continue
                filtered_executions.append(execution)
            
            # Recalculate summary for filtered data
            summary = {
                "total_executions": len(filtered_executions),
                "successful_executions": sum(1 for e in filtered_executions if e["success"]),
                "failed_executions": sum(1 for e in filtered_executions if not e["success"]),
                "average_duration_ms": sum(e["duration_ms"] for e in filtered_executions) / len(filtered_executions) if filtered_executions else 0
            }
            
            return {
                "workflow_id": workflow_id,
                **summary,
                "executions": filtered_executions[-10:]  # Last 10 executions
            }
        
        return metrics