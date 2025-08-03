// components/agent-renderer.tsx
import React, { useEffect, useState } from 'react';
import { subscribe, getTool } from '../agentic-ui/ag-ui-client';
import { FallbackComponent } from '@/components/fallback-component';

interface RenderedComponent {
  id: string;
  componentName: string;
  props: any;
}

export function AgentRenderer({ children }: { children?: React.ReactNode }) {
  const [renderedComponents, setRenderedComponents] = useState<RenderedComponent[]>([]);

  useEffect(() => {
    const unsubscribe = subscribe((event) => {
      console.log('AG-UI Event:', event);
      
      // Handle renderComponent events
      if (event.type === 'renderComponent') {
        const { componentName, props } = event.payload;
        setRenderedComponents(prev => [...prev, {
          id: `${componentName}-${Date.now()}`,
          componentName,
          props
        }]);
      }
      
      // Handle agentResponse events
      if (event.type === 'agentResponse') {
        console.log('Agent Response:', event.payload);
      }
      
      // Handle toolCall events
      if (event.type === 'toolCall') {
        console.log('Tool Call:', event.payload);
      }
    });

    return unsubscribe;
  }, []);

  return (
    <div className="agent-renderer">
      {children}
      <div className="agent-components space-y-4">
        {renderedComponents.map((item) => {
          const Component = getTool(item.componentName);
          
          if (!Component) {
            return (
              <FallbackComponent
                key={item.id}
                componentType={item.componentName}
                props={item.props}
                error={`Component "${item.componentName}" not found in registry`}
              />
            );
          }
          
          return <Component key={item.id} {...item.props} />;
        })}
      </div>
    </div>
  );
}