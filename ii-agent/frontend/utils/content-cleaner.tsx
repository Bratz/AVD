// utils/content-cleaner.ts

/**
 * Cleans content by removing special tags and formatting
 * @param content The raw content string
 * @returns The cleaned content string
 */
export function cleanContent(content: string): string {
  if (!content || typeof content !== 'string') return '';
  
  let cleaned = content;
  
  // STEP 1: Remove system messages and debug info FIRST
  cleaned = cleaned.replace(/Top-level agent planning next step:[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/\(Current token count: \d+\)/g, '');
  cleaned = cleaned.replace(/---+ NEW TURN ---+/g, '');
  cleaned = cleaned.replace(/\[no tools were called\]/g, '');
  cleaned = cleaned.replace(/Sending final response:/g, '');
  cleaned = cleaned.replace(/new phase/gm, '');
  
  // STEP 2: Remove ALL think/thinking tags and their content
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/gim, '');
  cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gim, '');
  cleaned = cleaned.replace(/<\/?think>/gim, '');
  cleaned = cleaned.replace(/<\/?thinking>/gim, '');
  
  // STEP 3: NOW protect remaining blocks
  const protectedBlocks: { placeholder: string; content: string }[] = [];
  let blockIndex = 0;
  
  // Protect chart blocks
  cleaned = cleaned.replace(/(```(?:chart|piechart)[\s\S]*?```)/g, (match) => {
    const placeholder = `__PROTECTED_BLOCK_${blockIndex++}__`;
    protectedBlocks.push({ placeholder, content: match });
    return placeholder;
  });
  
  // Protect code blocks
  cleaned = cleaned.replace(/(```[\s\S]*?```)/g, (match) => {
    const placeholder = `__PROTECTED_BLOCK_${blockIndex++}__`;
    protectedBlocks.push({ placeholder, content: match });
    return placeholder;
  });
  
  // Protect inline code
  cleaned = cleaned.replace(/(`[^`]+`)/g, (match) => {
    const placeholder = `__PROTECTED_BLOCK_${blockIndex++}__`;
    protectedBlocks.push({ placeholder, content: match });
    return placeholder;
  });
  
// STEP 4: Remove tool messages and technical details
  // Remove entire reasoning sections, not just headers
  cleaned = cleaned.replace(/\*\*?Reasoning:\*\*?[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/\*\*?Rationale:\*\*?[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/Reasoning:[\s\S]*?(?=\n\n|$)/gi, '');

  // Remove technical explanations
  cleaned = cleaned.replace(/Now that we've established.*?[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/Let me.*?(?:identify|find|search|look).*?[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/I'll.*?(?:use|call|invoke).*?[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/First,?\s*I.*?(?:need to|will|should).*?[\s\S]*?(?=\n\n|$)/gi, '');

  // Remove tool operation descriptions
  cleaned = cleaned.replace(/I'm using.*?with:[\s\S]*?(?=\n\n|$)/gi, '');
  cleaned = cleaned.replace(/Using.*?tool.*?to.*?$/gmi, '');
  cleaned = cleaned.replace(/This should.*?(?:surface|return|give|provide).*?$/gmi, '');

  // Remove API/technical headers
  cleaned = cleaned.replace(/\*\*?(?:Invoking|Calling|Using|Executing).*?API\*\*?/gi, '');
  cleaned = cleaned.replace(/\*\*?(?:Listing|Getting|Fetching|Retrieving).*?\*\*?/gi, '');

  // Remove technical parameters
  cleaned = cleaned.replace(/(?:tag|search_query|params?)=["'].*?["']/gi, '');
  cleaned = cleaned.replace(/with (?:tag|parameter|query).*?$/gmi, '');

  // Remove step-by-step technical descriptions
  cleaned = cleaned.replace(/Step \d+:.*?$/gmi, '');
  cleaned = cleaned.replace(/^\d+\.\s*(?:First|Then|Next|Finally).*?$/gmi, '');

// STEP 5: Fix bullet points - SURGICAL APPROACH
// STEP 5: Fix bullet points - SIMPLE PRESERVATION OF STRUCTURE
// Replace Unicode bullets with markdown bullets, preserving ALL spacing and indentation
  const bulletChars = '[•·▪▫◦‣⁃◘○●]';

  // Replace any Unicode bullet with a markdown dash, keeping all spacing intact
  cleaned = cleaned.replace(new RegExp(`(^|\\n)(\\s*)${bulletChars}`, 'gm'), '$1$2-');

  // Handle double bullets (e.g., "- •" becomes just "-")
  cleaned = cleaned.replace(/(-\s*)[•·▪▫◦‣⁃◘○●]/g, '-');
  cleaned = cleaned.replace(/([•·▪▫◦‣⁃◘○●]\s*)-/g, '-');
  
  // STEP 6: Clean up formatting
  cleaned = cleaned.replace(/\*\*Final Answer:\*\*\s*\n*/i, '');
  cleaned = cleaned.replace(/^[\t]+$/gm, '');
  // Only reduce excessive newlines (4 or more to 3)
  cleaned = cleaned.replace(/\n{4,}/g, '\n\n\n');
  
  // STEP 7: Restore protected blocks
  protectedBlocks.forEach(({ placeholder, content }) => {
    cleaned = cleaned.replace(placeholder, content);
  });
  
  // STEP 8: Final cleanup
  // Remove any leading/trailing whitespace
  cleaned = cleaned.trim();
  
  return cleaned;
}

/**
 * Sanitizes content for React rendering by escaping problematic tags
 * @param content The content to sanitize
 * @returns The sanitized content
 */
export function sanitizeForReact(content: string): string {
  if (!content || typeof content !== 'string') return '';
  
  // Only escape dangerous tags, don't clean again
  return content
    .replace(/<(think|THINK|thinking|THINKING)>/g, '&lt;$1&gt;')
    .replace(/<\/(think|THINK|thinking|THINKING)>/g, '&lt;/$1&gt;');
}

/**
 * Extracts thinking content from a message
 * @param content The raw content string
 * @returns The thinking content if found, or null
 */
export function extractThinkingContent(content: string): string | null {
  if (!content) return null;
  
  // Try to extract <think> content
  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
  if (thinkMatch) {
    return thinkMatch[1].trim();
  }
  
  // Try to extract <thinking> content
  const thinkingMatch = content.match(/<thinking>([\s\S]*?)<\/thinking>/i);
  if (thinkingMatch) {
    return thinkingMatch[1].trim();
  }
  
  return null;
}

/**
 * Checks if content contains thinking tags
 * @param content The content to check
 * @returns True if thinking tags are found
 */
export function hasThinkingTags(content: string): boolean {
  if (!content) return false;
  
  return /<think>|<\/think>|<thinking>|<\/thinking>/i.test(content);
}

/**
 * Cleans content specifically for AI responses
 * @param content The AI response content
 * @returns The cleaned content suitable for display
 */
export function cleanAIResponse(content: string): string {
  // First do the standard cleaning
  let cleaned = cleanContent(content);
  
  // Additional AI-specific cleaning if needed
  // Remove any remaining artifacts specific to your AI system
  
  return cleaned;
}