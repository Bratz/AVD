// components/message-content.tsx
import React from 'react';
import Markdown from '@/components/markdown';
import { ChartMessage, detectChartData } from '@/components/chart-message';
import { cleanContent } from '@/utils/content-cleaner';
import { getTool } from '@/agentic-ui/ag-ui-client';
import { FallbackComponent } from '@/components/fallback-component';
import { getCurrentUIMode } from '@/utils/ui-mode';

interface MessageContentProps {
  children: string;
  isCopilot?: boolean; // Add this prop
}

interface AGUIComponentSpec {
  type: string;
  props: any;
}

function isAGUIComponentSpec(obj: any): obj is AGUIComponentSpec {
  return obj && typeof obj === 'object' && 'type' in obj && 'props' in obj;
}

export const MessageContent: React.FC<MessageContentProps> = ({ 
  children,
  isCopilot = false 
}) => {
  const rawcontent = typeof children === 'string' ? children : '';
  const uiMode = getCurrentUIMode(isCopilot);
  
  console.log('=== MessageContent Debug ===');
  console.log('Raw children:', children);
  console.log('UI Mode:', uiMode);
  console.log('Is Copilot:', isCopilot);
  console.log('Contains ```chart?', rawcontent.includes('```chart'));
  console.log('Content length:', rawcontent.length);

  // Skip rendering if content is empty
  if (!rawcontent || rawcontent.length === 0) {
    return null;
  }

  // Only try AG-UI parsing in agentic mode
  if (uiMode === 'agentic') {
    try {
      // First, try to clean markdown code blocks if present
      let jsonContent = rawcontent;
      
      // Check if wrapped in ```json blocks and extract
      const jsonBlockMatch = rawcontent.match(/^```json\s*\n?([\s\S]*?)\n?```$/);
      if (jsonBlockMatch) {
        jsonContent = jsonBlockMatch[1].trim();
        console.log('Extracted from code block:', jsonContent);
      }
      
      const parsed = JSON.parse(jsonContent);
      console.log('Parsed JSON:', parsed);
      
      if (isAGUIComponentSpec(parsed)) {
        const Component = getTool(parsed.type.toLowerCase());
        if (Component) {
          console.log(`Rendering AG-UI component: ${parsed.type}`);
          return <Component {...parsed.props} />;
        } else {
          // Use FallbackComponent when component is not found
          return (
            <FallbackComponent
              componentType={parsed.type}
              props={parsed.props}
              error={`Component "${parsed.type}" not found in AG-UI registry`}
            />
          );
        }
      }
      
      // Array of AG-UI components
      if (Array.isArray(parsed) && parsed.every(isAGUIComponentSpec)) {
        return (
          <div className="agui-components space-y-4">
            {parsed.map((spec, index) => {
              const Component = getTool(spec.type.toLowerCase());
              if (Component) {
                return <Component key={index} {...spec.props} />;
              } else {
                return (
                  <FallbackComponent
                    key={index}
                    componentType={spec.type}
                    props={spec.props}
                    error={`Component "${spec.type}" not found`}
                  />
                );
              }
            })}
          </div>
        );
      }
    } catch (e) {
      console.log('Not valid AG-UI JSON, checking for traditional content');
      // Not JSON or parsing failed, continue with traditional rendering
    }
  } else {
    console.log('Traditional mode - skipping AG-UI parsing');
  }

  // Traditional mode or AG-UI parsing failed - check for charts
  const content = cleanContent(rawcontent);
  const chartData = detectChartData(rawcontent);
  console.log('Chart data detected?', !!chartData);
  
  if (chartData) {
    // Split content to handle text before and after chart
    const parts = rawcontent.split(/(```(?:chart|piechart)[\s\S]*?```)/);

    return (
      <div className="message-content">
        {parts.map((part, index) => {
          const partChartData = detectChartData(part);
          
          if (partChartData) {
            return <ChartMessage key={`chart-${index}`} chartData={partChartData} />;
          }
          
          // Clean and check the part
          const cleanPart = cleanContent(part);
          
          if (cleanPart && cleanPart.length > 0) {
            return <Markdown key={`text-${index}`}>{cleanPart}</Markdown>;
          }
          
          return null;
        })}
      </div>
    );
  }
  
  // No chart data, render as regular markdown
  return <Markdown>{content}</Markdown>;
};