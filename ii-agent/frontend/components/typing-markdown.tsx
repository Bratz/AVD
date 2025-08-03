import { useEffect, useState } from 'react';
import Markdown from './markdown';

interface TypingMarkdownProps {
  text: string;
  speed?: number;
  onComplete?: () => void;
}

export function TypingMarkdown({ text, speed = 15, onComplete }: TypingMarkdownProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        // Stream characters more naturally
        let chunkSize = 1;
        
        // Speed up for spaces and punctuation
        if (text[currentIndex] === ' ') {
          chunkSize = 1;
        }
        // Stream multiple characters for smoother effect
        else if (currentIndex < text.length - 3) {
          chunkSize = Math.min(2 + Math.floor(Math.random() * 2), text.length - currentIndex);
        }
        
        setDisplayedText(prev => prev + text.slice(currentIndex, currentIndex + chunkSize));
        setCurrentIndex(prev => prev + chunkSize);
      }, speed);

      return () => clearTimeout(timer);
    } else if (onComplete) {
      onComplete();
    }
  }, [currentIndex, text, speed, onComplete]);

  // Render markdown as it streams
  return (
    <div className="inline">
      <Markdown>{displayedText}</Markdown>
      {currentIndex < text.length && (
        <span className="inline-block w-[2px] h-4 bg-blue-400 animate-pulse ml-0.5 align-middle" />
      )}
    </div>
  );
}