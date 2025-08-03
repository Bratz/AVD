# src/ii_agent/tools/chart_tool.py
import json
from typing import Dict, List, Any, Optional, Literal
from src.ii_agent.tools.base import LLMTool, ToolImplOutput
from src.ii_agent.llm.message_history import MessageHistory


class ChartTool(LLMTool):
    """Tool for generating chart visualizations with modern styling."""
    
    name = "generate_chart"
    description = """Generate interactive charts with modern, professional styling.
    Use this tool when you need to display data in a visual format like:
    - Bar charts for comparing categories
    - Line charts for showing trends over time
    - Pie charts for showing proportions/percentages
    
    Supports modern color themes and gradients for professional visualizations."""
    
    # Modern color palettes
    COLOR_PALETTES = {
        "default": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#6366f1", "#14b8a6", "#f97316"],
        "ocean": ["#0891b2", "#06b6d4", "#22d3ee", "#67e8f9", "#a5f3fc", "#cffafe", "#e0f2fe", "#f0f9ff"],
        "sunset": ["#dc2626", "#ea580c", "#f97316", "#fb923c", "#fdba74", "#fed7aa", "#ffedd5", "#fff7ed"],
        "forest": ["#166534", "#15803d", "#16a34a", "#22c55e", "#4ade80", "#86efac", "#bbf7d0", "#dcfce7"],
        "purple": ["#6b21a8", "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe", "#ede9fe", "#f5f3ff"],
        "modern": ["#6366f1", "#8b5cf6", "#a855f7", "#c026d3", "#e11d48", "#f43f5e", "#f97316", "#facc15"],
        "pastel": ["#fbbf24", "#fb923c", "#f472b6", "#e879f9", "#c084fc", "#a78bfa", "#93c5fd", "#86efac"],
        "vibrant": ["#0ea5e9", "#06b6d4", "#14b8a6", "#10b981", "#22c55e", "#84cc16", "#eab308", "#f59e0b"],
        "corporate": ["#1e40af", "#2563eb", "#3b82f6", "#60a5fa", "#93bbfc", "#c3d9fe", "#e0ecff", "#f0f6ff"],
        "earth": ["#92400e", "#b45309", "#d97706", "#f59e0b", "#fbbf24", "#fde047", "#fef08a", "#fef9c3"]
    }
    
    # Gradient definitions for modern look
    GRADIENTS = {
        "blue": [{"offset": "0%", "color": "#3b82f6"}, {"offset": "100%", "color": "#1d4ed8"}],
        "green": [{"offset": "0%", "color": "#10b981"}, {"offset": "100%", "color": "#059669"}],
        "purple": [{"offset": "0%", "color": "#8b5cf6"}, {"offset": "100%", "color": "#6d28d9"}],
        "red": [{"offset": "0%", "color": "#ef4444"}, {"offset": "100%", "color": "#dc2626"}],
        "orange": [{"offset": "0%", "color": "#f97316"}, {"offset": "100%", "color": "#ea580c"}],
        "teal": [{"offset": "0%", "color": "#14b8a6"}, {"offset": "100%", "color": "#0f766e"}],
        "pink": [{"offset": "0%", "color": "#ec4899"}, {"offset": "100%", "color": "#db2777"}],
        "indigo": [{"offset": "0%", "color": "#6366f1"}, {"offset": "100%", "color": "#4f46e5"}]
    }
    
    input_schema = {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie", "area", "radar"],
                "description": "Type of chart to generate"
            },
            "data": {
                "type": "array",
                "description": "Array of data points. Each item must have 'name' and 'value' properties (or multiple value keys for multi-series)",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Label for the data point"},
                        "value": {"type": "number", "description": "Numeric value (for single series)"}
                    },
                    "required": ["name"]
                }
            },
            "title": {
                "type": "string",
                "description": "Title for the chart"
            },
            "subtitle": {
                "type": "string", 
                "description": "Subtitle for additional context"
            },
            "theme": {
                "type": "string",
                "enum": ["default", "ocean", "sunset", "forest", "purple", "modern", "pastel", "vibrant", "corporate", "earth"],
                "description": "Color theme for the chart"
            },
            "use_gradient": {
                "type": "boolean",
                "description": "Whether to use gradient fills for bars/areas"
            },
            "show_grid": {
                "type": "boolean",
                "description": "Whether to show grid lines"
            },
            "show_legend": {
                "type": "boolean",
                "description": "Whether to show legend"
            },
            "animation": {
                "type": "boolean",
                "description": "Whether to animate the chart"
            },
            "series_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names for multiple data series"
            }
        },
        "required": ["chart_type", "data"]
    }
    
    def run_impl(
        self,
        tool_input: Dict[str, Any],
        message_history: Optional[MessageHistory] = None,
    ) -> ToolImplOutput:
        """Generate a chart with modern styling."""
        
        chart_type = tool_input["chart_type"]
        data = tool_input["data"]
        title = tool_input.get("title", None)
        subtitle = tool_input.get("subtitle", None)
        theme = tool_input.get("theme", "modern")
        use_gradient = tool_input.get("use_gradient", True)
        show_grid = tool_input.get("show_grid", True)
        show_legend = tool_input.get("show_legend", True)
        animation = tool_input.get("animation", True)
        series_names = tool_input.get("series_names", None)
        
        # Validate data
        if not data:
            return ToolImplOutput(
                tool_output="Error: No data provided for chart",
                tool_result_message="Failed to generate chart: no data provided"
            )
        
        # Detect if multi-series data
        is_multi_series = False
        if data and len(data) > 0:
            first_item = data[0]
            value_keys = [k for k in first_item.keys() if k != 'name' and isinstance(first_item.get(k), (int, float))]
            is_multi_series = len(value_keys) > 1 or 'value' not in first_item
            
            # Use series names if provided, otherwise use value keys
            if is_multi_series and not series_names:
                series_names = value_keys
        
        # Get color palette
        colors = self.COLOR_PALETTES.get(theme, self.COLOR_PALETTES["modern"])
        
        # Build chart specification
        chart_spec = {
            "type": chart_type,
            "data": data,
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "animation": animation,
                "animationDuration": 1000,
                "animationEasing": "easeInOutQuart"
            },
            "style": {
                "colors": colors,
                "gradients": self.GRADIENTS if use_gradient else None,
                "grid": {
                    "show": show_grid,
                    "strokeDasharray": "3 3",
                    "stroke": "#e5e7eb"
                },
                "axis": {
                    "fontSize": 12,
                    "fontFamily": "Inter, system-ui, sans-serif",
                    "color": "#6b7280"
                },
                "tooltip": {
                    "backgroundColor": "rgba(17, 24, 39, 0.95)",
                    "borderRadius": 8,
                    "fontSize": 12,
                    "padding": 12,
                    "boxShadow": "0 10px 15px -3px rgba(0, 0, 0, 0.1)"
                }
            }
        }
        
        if title:
            chart_spec["options"]["title"] = {
                "text": title,
                "fontSize": 20,
                "fontWeight": 600,
                "color": "#111827",
                "marginBottom": subtitle and 4 or 16
            }
            
        if subtitle:
            chart_spec["options"]["subtitle"] = {
                "text": subtitle,
                "fontSize": 14,
                "color": "#6b7280",
                "marginBottom": 16
            }
            
        if show_legend:
            chart_spec["options"]["legend"] = {
                "show": True,
                "position": "bottom",
                "fontSize": 12,
                "color": "#6b7280",
                "itemGap": 16,
                "icon": "circle"
            }
            
        if is_multi_series and series_names:
            chart_spec["options"]["series"] = series_names
            
        # Chart-specific styling
        if chart_type == "bar":
            chart_spec["style"]["bar"] = {
                "radius": [8, 8, 0, 0],  # Rounded top corners
                "maxBarSize": 60
            }
        elif chart_type == "line":
            chart_spec["style"]["line"] = {
                "strokeWidth": 3,
                "dot": {"r": 5, "strokeWidth": 2},
                "activeDot": {"r": 7}
            }
        elif chart_type == "area":
            chart_spec["style"]["area"] = {
                "fillOpacity": 0.3,
                "strokeWidth": 2
            }
        elif chart_type == "pie":
            chart_spec["style"]["pie"] = {
                "innerRadius": "30%",  # Makes it a donut chart
                "outerRadius": "80%",
                "paddingAngle": 2,
                "cornerRadius": 4
            }
        
        # Return in the format expected by the UI
        chart_output = f"```chart\n{json.dumps(chart_spec, indent=2)}\n```"
        
        # Create a descriptive message for logging
        data_summary = f"{len(data)} data points"
        result_message = f"Generated {chart_type} chart with {data_summary} using {theme} theme"
        if title:
            result_message += f' titled "{title}"'
        
        return ToolImplOutput(
            tool_output=chart_output,
            tool_result_message=result_message,
            auxiliary_data={
                "chart_type": chart_type,
                "data_points": len(data),
                "has_title": bool(title),
                "theme": theme,
                "is_multi_series": is_multi_series,
                "use_gradient": use_gradient
            }
        )
    
    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Return a user-friendly message when the tool is called."""
        chart_type = tool_input.get("chart_type", "chart")
        title = tool_input.get("title", "")
        theme = tool_input.get("theme", "modern")
        
        if title:
            return f"Generating {theme}-themed {chart_type} chart: {title}"
        else:
            return f"Generating {theme}-themed {chart_type} chart"