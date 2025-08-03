import { useState, useEffect, useMemo } from "react";
import { X, ChevronDown, RotateCcw,Brain, Zap } from "lucide-react";
import Cookies from "js-cookie";
import { motion } from "framer-motion";

import { Button } from "./ui/button";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Tooltip, TooltipTrigger, TooltipContent } from "./ui/tooltip";
import { AVAILABLE_MODELS, ToolSettings } from "@/typings/agent";
import { useAppContext } from "@/context/app-context";
import { Slider } from "./ui/slider";

interface SettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const SettingsDrawer = ({ isOpen, onClose }: SettingsDrawerProps) => {
  const { state, dispatch } = useAppContext();
  const [toolsExpanded, setToolsExpanded] = useState(true);
  const [reasoningExpanded, setReasoningExpanded] = useState(true);
  const [routingExpanded, setRoutingExpanded] = useState(true);

  const isClaudeModel = useMemo(
    () => state.selectedModel?.toLowerCase().includes("claude"),
    [state.selectedModel]
  );
  // const isOllamaModel = useMemo(
  // () => state.selectedModel?.toLowerCase().includes("ollama"),
  // [state.selectedModel]
  // );

  const handleToolToggle = (tool: keyof ToolSettings) => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        ...state.toolSettings,
        [tool]: !state.toolSettings[tool],
      },
    });
  };

  const resetSettings = () => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        deep_research: false,
        pdf: true,
        media_generation: true,
        audio_generation: true,
        browser: true,
        thinking_tokens: 10000,
        use_routing: false,
        simple_model: "NVIDIA/Nemotron-4-340B-Chat",
        complex_model: state.selectedModel || AVAILABLE_MODELS[0],
        complexity_threshold: 0.5,
      },
    });
    dispatch({ type: "SET_SELECTED_MODEL", payload: AVAILABLE_MODELS[0] });
  };

  const handleReasoningEffortChange = (effort: string) => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        ...state.toolSettings,
        thinking_tokens: effort === "high" ? 10000 : 0,
      },
    });
  };


  useEffect(() => {
    if (state.selectedModel) {
      Cookies.set("selected_model", state.selectedModel, {
        expires: 365, // 1 year
        sameSite: "strict",
        secure: window.location.protocol === "https:",
      });

      // Reset thinking_tokens to 0 for non-Claude models
      if (!isClaudeModel && state.toolSettings.thinking_tokens > 0) {
        dispatch({
          type: "SET_TOOL_SETTINGS",
          payload: { ...state.toolSettings, thinking_tokens: 0 },
        });
      }
    }
  }, [state.selectedModel, isClaudeModel, state.toolSettings, dispatch]);


    const handleRoutingToggle = (enabled: boolean) => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        ...state.toolSettings,
        use_routing: enabled,
        // Set default models if not already set
        simple_model: state.toolSettings.simple_model || "NVIDIA/Nemotron-4-340B-Chat",
        complex_model: state.toolSettings.complex_model || state.selectedModel,
      },
    });
  };

  const handleSimpleModelChange = (model: string) => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        ...state.toolSettings,
        simple_model: model,
      },
    });
  };

  const handleComplexModelChange = (model: string) => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        ...state.toolSettings,
        complex_model: model,
      },
    });
  };

  const handleThresholdChange = (value: number[]) => {
    dispatch({
      type: "SET_TOOL_SETTINGS",
      payload: {
        ...state.toolSettings,
        complexity_threshold: value[0],
      },
    });
  };

  return (
    <>
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />
      )}
      <motion.div
        className={`fixed top-0 right-0 h-full ${
          isOpen ? "w-[400px]" : "w-0"
        } bg-card z-50 shadow-xl overflow-auto`}
        initial={{ x: "100%" }}
        animate={{ x: isOpen ? 0 : "100%" }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
      >
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-foreground">Run settings</h2>
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-full hover:bg-gray-700/50"
                    onClick={resetSettings}
                  >
                    <RotateCcw className="size-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Reset Default Settings</TooltipContent>
              </Tooltip>
              <Button
                variant="ghost"
                size="icon"
                className="rounded-full hover:bg-gray-700/50"
                onClick={onClose}
              >
                <X className="size-5" />
              </Button>
            </div>
          </div>

          <div className="space-y-6">
            {/* Model selector */}
            <div className="space-y-2">
              <Select
                value={state.selectedModel}
                onValueChange={(model) =>
                  dispatch({ type: "SET_SELECTED_MODEL", payload: model })
                }
              >
                <SelectTrigger className="w-full bg-secondary border-[#ffffff0f]">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent className="bg-secondary border-[#ffffff0f]">
                  {AVAILABLE_MODELS.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {/* Intelligent Routing Section */}
            <div className="space-y-4 pt-4 border-t border-gray-700">
              <div
                className="flex justify-between items-center cursor-pointer"
                onClick={() => setRoutingExpanded(!routingExpanded)}
              >
                <h3 className="text-lg font-medium text-foreground flex items-center gap-2">
                  <Brain className="size-5 text-purple-400" />
                  Intelligent Routing
                </h3>
                <ChevronDown
                  className={`size-5 transition-transform ${
                    routingExpanded ? "rotate-180" : ""
                  }`}
                />
              </div>

              {routingExpanded && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="routing-enabled" className="text-gray-300">
                        Enable Intelligent Routing
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Route queries to different models based on complexity
                      </p>
                    </div>
                    <Switch
                      id="routing-enabled"
                      checked={state.toolSettings.use_routing || false}
                      onCheckedChange={handleRoutingToggle}
                    />
                  </div>

                  {state.toolSettings.use_routing && (
                    <>
                      {/* Simple Model Selection */}
                      <div className="space-y-2">
                        <Label className="text-gray-300 flex items-center gap-2">
                          <Zap className="size-4 text-yellow-400" />
                          Simple Queries Model (Fast)
                        </Label>
                        <p className="text-xs text-muted-foreground mb-2">
                          For basic queries and quick responses
                        </p>
                        <Select
                          value={state.toolSettings.simple_model || "NVIDIA/Nemotron-4-340B-Chat"}
                          onValueChange={handleSimpleModelChange}
                        >
                          <SelectTrigger className="w-full bg-secondary border-[#ffffff0f]">
                            <SelectValue placeholder="Select simple model" />
                          </SelectTrigger>
                          <SelectContent className="bg-secondary border-[#ffffff0f]">
                            {AVAILABLE_MODELS.map((model) => (
                              <SelectItem key={model} value={model}>
                                {model}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Complex Model Selection */}
                      <div className="space-y-2">
                        <Label className="text-gray-300 flex items-center gap-2">
                          <Brain className="size-4 text-purple-400" />
                          Complex Queries Model (Smart)
                        </Label>
                        <p className="text-xs text-muted-foreground mb-2">
                          For complex analysis and reasoning tasks
                        </p>
                        <Select
                          value={state.toolSettings.complex_model || state.selectedModel}
                          onValueChange={handleComplexModelChange}
                        >
                          <SelectTrigger className="w-full bg-secondary border-[#ffffff0f]">
                            <SelectValue placeholder="Select complex model" />
                          </SelectTrigger>
                          <SelectContent className="bg-secondary border-[#ffffff0f]">
                            {AVAILABLE_MODELS.map((model) => (
                              <SelectItem key={model} value={model}>
                                {model}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Complexity Threshold */}
                      <div className="space-y-2">
                        <Label className="text-gray-300">
                          Complexity Threshold
                        </Label>
                        <p className="text-xs text-muted-foreground mb-2">
                          Queries above this threshold use the complex model
                        </p>
                        <div className="flex items-center gap-4">
                          <Slider
                            value={[state.toolSettings.complexity_threshold || 0.5]}
                            onValueChange={handleThresholdChange}
                            min={0}
                            max={1}
                            step={0.1}
                            className="flex-1"
                          />
                          <span className="text-sm font-mono w-12 text-right">
                            {(state.toolSettings.complexity_threshold || 0.5).toFixed(1)}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Simple</span>
                          <span>Complex</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
            {/* Reasoning Effort section - only for Claude models */}
            {isClaudeModel && (
              <div className="space-y-4 pt-4 border-t border-gray-700">
                <div
                  className="flex justify-between items-center cursor-pointer"
                  onClick={() => setReasoningExpanded(!reasoningExpanded)}
                >
                  <h3 className="text-lg font-medium text-foreground">
                    Reasoning Effort
                  </h3>
                  <ChevronDown
                    className={`size-5 transition-transform ${
                      reasoningExpanded ? "rotate-180" : ""
                    }`}
                  />
                </div>

                {reasoningExpanded && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label
                        htmlFor="reasoning-effort"
                        className="text-gray-300"
                      >
                        Effort Level
                      </Label>
                      <p className="text-xs text-muted-foreground mb-2">
                        Controls how much effort the model spends on reasoning
                        before responding
                      </p>
                      <Select
                        value={
                          state.toolSettings.thinking_tokens > 0
                            ? "high"
                            : "standard"
                        }
                        onValueChange={handleReasoningEffortChange}
                      >
                        <SelectTrigger className="w-full bg-secondary border-[#ffffff0f]">
                          <SelectValue placeholder="Select effort level" />
                        </SelectTrigger>
                        <SelectContent className="bg-secondary border-[#ffffff0f]">
                          <SelectItem value="standard">Standard</SelectItem>
                          <SelectItem value="high">High-effort</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tools section */}
            <div className="space-y-4 pt-4 border-t border-gray-700">
              <div
                className="flex justify-between items-center cursor-pointer"
                onClick={() => setToolsExpanded(!toolsExpanded)}
              >
                <h3 className="text-lg font-medium text-foreground">Tools</h3>
                <ChevronDown
                  className={`size-5 transition-transform ${
                    toolsExpanded ? "rotate-180" : ""
                  }`}
                />
              </div>

              {toolsExpanded && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="deep-research" className="text-gray-300">
                        Deep Research
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Enable in-depth research capabilities
                      </p>
                    </div>
                    <Switch
                      id="deep-research"
                      checked={state.toolSettings.deep_research}
                      onCheckedChange={() => handleToolToggle("deep_research")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="pdf" className="text-gray-300">
                        PDF Processing
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Extract and analyze PDF documents
                      </p>
                    </div>
                    <Switch
                      id="pdf"
                      checked={state.toolSettings.pdf}
                      onCheckedChange={() => handleToolToggle("pdf")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label
                        htmlFor="media-generation"
                        className="text-gray-300"
                      >
                        Media Generation
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Generate images and videos
                      </p>
                    </div>
                    <Switch
                      id="media-generation"
                      checked={state.toolSettings.media_generation}
                      onCheckedChange={() =>
                        handleToolToggle("media_generation")
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="banking-mode" className="text-gray-300">
                        Banking Mode (TCS BaNCS)
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Enable TCS BaNCS specialist agent for banking operations
                      </p>
                    </div>
                 <Switch
                      id="banking-mode"
                      checked={state.toolSettings.banking_mode || false}
                      onCheckedChange={() => handleToolToggle("banking_mode")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label
                        htmlFor="audio-generation"
                        className="text-gray-300"
                      >
                        Audio Generation
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Generate and process audio content
                      </p>
                    </div>
                    <Switch
                      id="audio-generation"
                      checked={state.toolSettings.audio_generation}
                      onCheckedChange={() =>
                        handleToolToggle("audio_generation")
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="browser" className="text-gray-300">
                        Browser
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Enable web browsing capabilities
                      </p>
                    </div>
                    <Switch
                      id="browser"
                      checked={state.toolSettings.browser}
                      onCheckedChange={() => handleToolToggle("browser")}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </>
  );
};

export default SettingsDrawer;
