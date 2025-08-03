"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkBreaks from "remark-breaks"; // ADD THIS - Critical for line breaks
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import rehypeMathJax from "rehype-mathjax";
import rehypeKatex from "rehype-katex";

import "katex/dist/katex.min.css";
// import "highlight.js/styles/github-dark.css";
// import "highlight.js/styles/monokai.css"; // Classic dark theme
//import "highlight.js/styles/vs2015.css"; // More neutral colors

import { sanitizeForReact } from '@/utils/content-cleaner';

interface MarkdownProps {
  children: string | null | undefined;
  className?: string;
}

const Markdown = ({ children, className = "" }: MarkdownProps) => {
  const safeContent = sanitizeForReact(children || "");
  
  // Pre-process content to handle special cases WITHOUT removing line breaks
  const processedContent = safeContent
    // Preserve emojis with proper spacing
    // .replace(/([🔹📊💡💰🏦💳📈📉✅❌])/g, ' $1 ')
    // Clean up excessive SPACES ONLY (not newlines!)
    .replace(/[ \t]{3,}/g, ' ')
    .trim();

  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown
        remarkPlugins={[
          remarkGfm, 
          remarkMath,
          remarkBreaks // THIS IS CRITICAL - it converts single line breaks to <br>
        ]}
        rehypePlugins={[rehypeRaw, rehypeHighlight, rehypeMathJax, rehypeKatex]}
        components={{
          // Links
          a: ({ ...props }) => (
            <a 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-blue-600 dark:text-blue-400 hover:underline"
              {...props} 
            />
          ),
          
          // Headers
          h1: ({ ...props }) => <h1 className="text-2xl font-bold mt-6 mb-4" {...props} />,
          h2: ({ ...props }) => <h2 className="text-xl font-bold mt-5 mb-3" {...props} />,
          h3: ({ ...props }) => <h3 className="text-lg font-bold mt-4 mb-2" {...props} />,
          
          // Paragraphs - minimal margin for single line spacing
          p: ({ ...props }) => <p className="my-1 leading-relaxed" {...props} />,
          
          // Strong/Bold
          strong: ({ ...props }) => <strong className="font-bold" {...props} />,
          
          // Lists - single line spacing
          ul: ({ ...props }) => <ul className="my-2 space-y-0.5" {...props} />,
          ol: ({ ...props }) => <ol className="list-decimal list-inside my-2 space-y-0.5" {...props} />,
          li: ({ children, ...props }) => {
            const content = String(children);
            // Check if it starts with an emoji bullet
            const hasEmojiBullet = /^[\s]*[🔹📊💡💰🏦💳📈📉✅❌]/.test(content);
            
            return (
              <li 
                // className={`${hasEmojiBullet ? 'list-none' : 'list-disc list-inside ml-4'} my-0.5`} 
                className="list-none ml-4 my-0.5"
                {...props}
              >
                {children}
              </li>
            );
          },
              pre: ({ children, ...props }) => (
                <pre 
                  className="bg-muted dark:bg-gray-800/50 rounded-lg p-4 overflow-x-auto my-4 border border-border" 
                  {...props}
                >
                  {children}
                </pre>
              ),          
          // Code blocks
          code: ({ className, children, ...props }) => {
              const match = /language-(\w+)/.exec(className || '');
              const isInline = !match && typeof children === 'string' && !children.includes('\n');


              if (isInline) {
                return (
                  <code 
                    className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono"
                    {...props}
                  >
                    {children}
                  </code>
                );
              }
              
              // For code blocks, check if it's JSON and format it
              let content = children;
              if (className === 'language-json' && typeof children === 'string') {
                try {
                  // Parse and re-stringify to fix formatting
                  const parsed = JSON.parse(children);
                  content = JSON.stringify(parsed, null, 2);
                } catch (e) {
                  // If it's not valid JSON, leave it as is
                  content = children;
                }
              }
              
              // Let rehypeHighlight handle the syntax highlighting
            return (
                <code className={className} {...props}>
                  {content}
                </code>
              );
            },
          
          // Line breaks - minimal spacing
          br: () => <br className="leading-tight" />,
          
          // Blockquotes
          blockquote: ({ ...props }) => (
            <blockquote 
              className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 my-4 italic"
              {...props} 
            />
          ),
          
          // Tables
          table: ({ ...props }) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700" {...props} />
            </div>
          ),
          thead: ({ ...props }) => (
            <thead className="bg-gray-50 dark:bg-gray-800" {...props} />
          ),
          tbody: ({ ...props }) => (
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700" {...props} />
          ),
          th: ({ ...props }) => (
            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider" {...props} />
          ),
          td: ({ ...props }) => (
            <td className="px-4 py-2 whitespace-nowrap text-sm" {...props} />
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;