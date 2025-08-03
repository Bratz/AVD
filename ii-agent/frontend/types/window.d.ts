// types/window.d.ts
interface Window {
  copilotKit?: {
    // Add the specific properties and methods you use from copilotKit
    // For example:
    sendMessage?: (message: string) => void;
    updateContext?: (context: any) => void;
    // Add other methods/properties as needed
  };
}