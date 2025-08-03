// components/chart-message.tsx
import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  Filler,
  ChartData as ChartJSData,
  ChartOptions as ChartJSOptions,
  ChartTypeRegistry,
  ScriptableContext,
  ChartArea
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import type { TooltipItem } from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Define our custom data format
interface SimpleDataPoint {
  name: string;
  value: number;
}

// Define chart type
type ChartType = 'bar' | 'line' | 'pie' | 'doughnut' | 'radar' | 'area';
type ChartJSType = keyof ChartTypeRegistry;

// Define proper interfaces for options
interface CustomChartOptions {
  title?: {
    text?: string;
    fontSize?: number;
    fontWeight?: number;
    color?: string;
    marginBottom?: number;
  } | string;
  subtitle?: {
    text?: string;
    fontSize?: number;
    color?: string;
    marginBottom?: number;
  };
  legend?: {
    show?: boolean;
    position?: 'top' | 'bottom' | 'left' | 'right';
  };
  animationDuration?: number;
  animationEasing?: string;
  theme?: string;
  show_grid?: boolean;
  show_legend?: boolean;
}

interface ChartStyle {
  colors?: string[];
  bar?: {
    maxBarSize?: number;
  };
}

export interface ChartData {
  type: ChartType;
  data: SimpleDataPoint[] | ChartJSData<ChartJSType>;
  options?: CustomChartOptions;
  style?: ChartStyle;
}

// Type guard
function isChartJSFormat(data: SimpleDataPoint[] | ChartJSData<ChartJSType>): data is ChartJSData<ChartJSType> {
  return data && typeof data === 'object' && 'datasets' in data;
}

// Fixed detectChartData function that returns ChartData | null
export function detectChartData(content: string): ChartData | null {
   console.log('=== detectChartData ===');
  // Early return for empty or dash-only content
  if (!content || content.trim() === '' || content.trim() === '-' || content.trim() === '--') {
    return null;
  }

  try {
    // const chartMatch = content.match(/```(?:chart|piechart)\n([\s\S]*?)\n```/);
    const chartMatch = content.match(/```chart\s*\n([\s\S]*?)```/);
    if (!chartMatch || !chartMatch[1]) {
      return null;
    }

    const chartContent = chartMatch[1].trim();
    
    // Skip if the content is just dashes or empty
    if (!chartContent || /^-+$/.test(chartContent)) {
      return null;
    }

    try {
      const parsedData = JSON.parse(chartContent);
      console.log('Parsed chart data:', parsedData);
      
      // Handle tool_call format
      if (parsedData.type && parsedData.data) {
        // Transform data if needed
        let chartData = parsedData.data;
        if (chartData.length > 0 && 'name' in chartData[0] && 'value' in chartData[0]) {
          // Data is already in the correct format
        }
        
        return {
          type: parsedData.type || 'bar',
          data: chartData,
          options: parsedData.options || {
            title: parsedData.title,
            subtitle: parsedData.subtitle
          }
        };
      }
      
      return null;
    } catch (e) {
      console.error('Failed to parse chart JSON:', e);
      return null;
    }
  } catch (error) {
    console.error('Error in detectChartData:', error);
    return null;
  }
}

export const ChartMessage: React.FC<{ chartData: ChartData }> = ({ chartData }) => {
  const { type, data, options = {}, style = {} } = chartData;
  
  const [chartId] = React.useState(() => `chart-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
    // ADD: Ref to track if chart is already rendered
    const isRendered = React.useRef(false);
    
    // ADD: Effect to mark as rendered and cleanup
    React.useEffect(() => {
      isRendered.current = true;
      
      return () => {
        isRendered.current = false;
      };
    }, []);


  // Map chart types
  const chartType: ChartJSType = (type === 'area' ? 'line' : type) as ChartJSType;

  // Convert data format based on chart type
  const prepareChartData = (): ChartJSData<ChartJSType> => {
    // Professional muted color palette
    const mutedColors = [
      '#6B7280', // Gray-500
      '#9CA3AF', // Gray-400
      '#60A5FA', // Blue-400
      '#86EFAC', // Green-300
      '#FCA5A5', // Red-300
      '#C7D2FE', // Indigo-300
      '#FDE68A', // Yellow-300
      '#A5B4FC', // Indigo-400
    ];

    // Matching gradient colors (slightly darker)
    const gradientColors = [
      { from: '#6B7280', to: '#4B5563' }, // Gray gradient
      { from: '#9CA3AF', to: '#6B7280' }, // Light gray gradient
      { from: '#60A5FA', to: '#3B82F6' }, // Blue gradient
      { from: '#86EFAC', to: '#4ADE80' }, // Green gradient
      { from: '#FCA5A5', to: '#F87171' }, // Red gradient
      { from: '#C7D2FE', to: '#A5B4FC' }, // Indigo gradient
      { from: '#FDE68A', to: '#FCD34D' }, // Yellow gradient
      { from: '#A5B4FC', to: '#818CF8' }, // Purple gradient
    ];
    
    // If already in Chart.js format, return as is
    if (isChartJSFormat(data)) {
      return data;
    }
    
    // Convert from simple format
    const simpleData = data as SimpleDataPoint[];
    
    if (type === 'pie' || type === 'doughnut') {
      return {
        labels: simpleData.map(item => item.name),
        datasets: [{
          data: simpleData.map(item => item.value),
          backgroundColor: style.colors || mutedColors,
          borderWidth: 0,
          hoverOffset: 4,
          hoverBorderWidth: 2,
          hoverBorderColor: '#ffffff'
        }]
      };
    }

    // For bar charts - create gradient effect
    const createGradient = (ctx: CanvasRenderingContext2D, chartArea: ChartArea, colorIndex: number): string | CanvasGradient => {
      if (!chartArea) return mutedColors[colorIndex];
      
      const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
      const gradientColor = gradientColors[colorIndex % gradientColors.length];
      gradient.addColorStop(0, gradientColor.from);
      gradient.addColorStop(1, gradientColor.to);
      return gradient;
    };

    // For bar, line, area charts
    return {
      labels: simpleData.map(item => item.name),
      datasets: [{
        label: typeof options.title === 'object' ? options.title?.text : options.title || 'Data',
        data: simpleData.map(item => item.value),
        
        // Use direct colors instead of gradient functions for bar charts
        backgroundColor: type === 'bar' ? 
          // Use the vibrant colors array for each bar
          simpleData.map((_, index) => {
            // If custom colors provided in style, use them
            if (style.colors && style.colors.length > 0) {
              return style.colors[index % style.colors.length];
            }
            // Otherwise use the muted colors palette
            return mutedColors[index % mutedColors.length];
          }) : 
          type === 'line' || type === 'area' ? 'rgba(96, 165, 250, 0.1)' : 
          (style.colors?.[0] || mutedColors[0]),
        
        // Match border colors to background colors for bars
        borderColor: type === 'bar' ?
          simpleData.map((_, index) => {
            if (style.colors && style.colors.length > 0) {
              return style.colors[index % style.colors.length];
            }
            return mutedColors[index % mutedColors.length];
          }) :
          (style.colors?.[0] || mutedColors[2]),
        
        borderWidth: type === 'bar' ? 1 : type === 'line' ? 2.5 : 0,
        fill: type === 'area',
        tension: type === 'line' || type === 'area' ? 0.4 : 0,
        borderRadius: type === 'bar' ? 12 : 0,
        maxBarThickness: style.bar?.maxBarSize || 48,
        pointRadius: type === 'line' ? 0 : 6,
        pointHoverRadius: type === 'line' ? 6 : 8,
        pointBackgroundColor: '#ffffff',
        pointBorderColor: mutedColors[2],
        pointBorderWidth: 2,
        pointHoverBorderWidth: 3,
        hoverBackgroundColor: type === 'bar' ?
          simpleData.map((_, index) => {
            const baseColor = style.colors?.[index % (style.colors?.length || 1)] || mutedColors[index % mutedColors.length];
            // Make hover color slightly darker
            return baseColor + 'DD'; // Add transparency for hover effect
          }) : undefined,
      }]
    };
  };
  
  // Prepare Chart.js options
  const prepareOptions = (): ChartJSOptions<ChartJSType> => {
    const isDark = document.documentElement.classList.contains('dark');
    const textColor = isDark ? '#D1D5DB' : '#4B5563';
    const gridColor = isDark ? 'rgba(55, 65, 81, 0.3)' : 'rgba(229, 231, 235, 0.5)';
    
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index' as const,
      },
      animation: {
        duration: options.animationDuration || 750,
        easing: 'easeInOutCubic'
      },
      plugins: {
          title: options.title ? {
            display: true,
            text: typeof options.title === 'string' ? options.title : options.title.text,
            font: {
              size: typeof options.title === 'object' ? (options.title.fontSize || 18) : 18,
              weight: typeof options.title === 'object' ? (options.title.fontWeight || 500) : 500,
              family: "'Inter', 'system-ui', sans-serif"
            },
            color: typeof options.title === 'object' ? (options.title.color || textColor) : textColor,
            padding: {
              top: 0,
              bottom: typeof options.title === 'object' ? (options.title.marginBottom || 20) : 20
            }
          } : {
            display: false
          },
        subtitle: options.subtitle ? {
          display: true,
          text: options.subtitle.text,
          font: {
            size: options.subtitle.fontSize || 13,
            weight: 400,
            family: "'Inter', 'system-ui', sans-serif"
          },
          color: options.subtitle.color || gridColor,
          padding: {
            bottom: options.subtitle.marginBottom || 20
          }
        } : {
          display: false
        },
        legend: {
          display: options.legend?.show !== false && (type === 'pie' || type === 'doughnut'),
          position: options.legend?.position || 'bottom',
          labels: {
            color: textColor,
            font: {
              size: 12,
              weight: 500,
              family: "'Inter', 'system-ui', sans-serif"
            },
            padding: 20,
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        tooltip: {
          enabled: true,
          backgroundColor: isDark ? 'rgba(17, 24, 39, 0.95)' : 'rgba(255, 255, 255, 0.95)',
          titleColor: isDark ? '#F3F4F6' : '#1F2937',
          bodyColor: isDark ? '#D1D5DB' : '#4B5563',
          borderColor: isDark ? 'rgba(75, 85, 99, 0.3)' : 'rgba(229, 231, 235, 0.3)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          displayColors: false,
          titleFont: {
            size: 13,
            weight: 600,
            family: "'Inter', 'system-ui', sans-serif"
          },
          bodyFont: {
            size: 12,
            weight: 400,
            family: "'Inter', 'system-ui', sans-serif"
          },
          callbacks: {
            label: function(context: TooltipItem<ChartJSType>) {
              const label = context.label || context.dataset.label || '';
              let value: number;
              
              // Type-safe value extraction
              if (type === 'pie' || type === 'doughnut') {
                // For pie/doughnut charts, parsed is a number
                value = context.parsed as number;
              } else {
                // For other chart types, parsed is an object with x and y
                const parsed = context.parsed as { x?: number; y?: number } | number;
                
                if (typeof parsed === 'object' && parsed !== null) {
                  // It's a point object
                  value = parsed.y !== undefined ? parsed.y : (parsed.x !== undefined ? parsed.x : NaN);
                } else if (typeof parsed === 'number') {
                  // It's a direct number
                  value = parsed;
                } else if (typeof context.raw === 'number') {
                  // Fallback to raw value
                  value = context.raw;
                } else if (typeof context.raw === 'string') {
                  value = parseFloat(context.raw);
                } else {
                  value = NaN;
                }
              }
              
              if (isNaN(value)) {
                return `${label}: No data`;
              }
              
              const formattedValue = new Intl.NumberFormat('en-US', {
                style: 'decimal',
                minimumFractionDigits: 0,
                maximumFractionDigits: 2
              }).format(value);
              
              if (type === 'pie' || type === 'doughnut') {
                const dataset = context.dataset;
                const dataArray = dataset.data as number[];
                const total = dataArray.reduce((sum: number, val: number | string) => {
                  const num = typeof val === 'number' ? val : parseFloat(String(val));
                  return sum + (isNaN(num) ? 0 : num);
                }, 0);
                
                if (total > 0) {
                  const percentage = ((value / total) * 100).toFixed(1);
                  return `${label}: ${formattedValue} (${percentage}%)`;
                }
              }
              
              return label ? `${label}: ${formattedValue}` : formattedValue;
            },
            title: function(tooltipItems: TooltipItem<ChartJSType>[]) {
              if (tooltipItems.length > 0) {
                if (type === 'pie' || type === 'doughnut') {
                  return '';
                }
                return tooltipItems[0].label || '';
              }
              return '';
            }
          }
        }
      },
      scales: (type === 'pie' || type === 'doughnut' || type === 'radar') ? {} : {
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: textColor,
            font: {
              size: 11,
              weight: 500,
              family: "'Inter', 'system-ui', sans-serif"
            },
            padding: 8
          }
        },
        y: {
          grid: {
            display: true,
            color: gridColor,
          },
          ticks: {
            color: textColor,
            font: {
              size: 11,
              weight: 500,
              family: "'Inter', 'system-ui', sans-serif"
            },
            padding: 8,
            callback: function(value: string | number) {
              if (typeof value === 'number') {
                return new Intl.NumberFormat('en-US', {
                  notation: 'compact',
                  compactDisplay: 'short'
                }).format(value);
              }
              return value;
            }
          }
        }
      }
    };
  };
  
  // Single return statement
  return (
    <div className="chart-container bg-white dark:bg-gray-800 rounded-xl p-6 my-4 border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow duration-200">
      <div style={{ width: '100%', height: 400 }}>
        <Chart
          key={chartId}
          type={chartType}
          data={prepareChartData()} 
          options={prepareOptions()} 
        />
      </div>
    </div>
  );
};