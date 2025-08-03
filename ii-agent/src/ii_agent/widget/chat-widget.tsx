// src/ii_agent/widget/chat-widget-vanilla.ts
// Vanilla TypeScript implementation without React dependencies

interface ChatConfig {
  projectId: string;
  apiKey: string;
  host?: string;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  theme?: 'light' | 'dark';
  welcomeMessage?: string;
  placeholder?: string;
  primaryColor?: string;
  width?: number;
  height?: number;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatResponse {
  content: string;
  workflow_id?: string;
  execution_id?: string;
}

// Simple API Client
class IIAgentClient {
  private host: string;
  private apiKey: string;
  private projectId: string;

  constructor(config: { host: string; apiKey: string; projectId: string }) {
    this.host = config.host;
    this.apiKey = config.apiKey;
    this.projectId = config.projectId;
  }

  async chat(messages: Array<{ role: string; content: string }>): Promise<ChatResponse> {
    const response = await fetch(`${this.host}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        project_id: this.projectId,
        messages,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    return response.json();
  }
}

export class ChatWidget {
  private config: ChatConfig;
  private client: IIAgentClient;
  private isOpen: boolean = false;
  private messages: Message[] = [];
  private container: HTMLDivElement | null = null;
  private elements: {
    button?: HTMLButtonElement;
    chat?: HTMLDivElement;
    messagesContainer?: HTMLDivElement;
    input?: HTMLInputElement;
    sendButton?: HTMLButtonElement;
  } = {};

  constructor(config: ChatConfig) {
    this.config = {
      position: 'bottom-right',
      theme: 'light',
      welcomeMessage: 'Hi! How can I help you today?',
      placeholder: 'Type your message...',
      primaryColor: '#007bff',
      width: 380,
      height: 600,
      ...config,
    };

    this.client = new IIAgentClient({
      host: config.host || 'https://api.ii-agent.com',
      apiKey: config.apiKey,
      projectId: config.projectId,
    });

    this.render();
    this.attachEventListeners();
  }

  private render(): void {
    // Create container
    this.container = document.createElement('div');
    this.container.id = 'ii-agent-widget';
    this.container.className = `ii-agent-widget ${this.config.position} ${this.config.theme}`;

    // Create toggle button
    this.elements.button = document.createElement('button');
    this.elements.button.className = 'ii-agent-widget-button';
    this.elements.button.innerHTML = `
      <svg width="30" height="30" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12c0 1.54.36 3 .97 4.29L1 23l6.71-1.97C9 21.64 10.46 22 12 22c5.52 0 10-4.48 10-10S17.52 2 12 2zm0 18c-1.41 0-2.73-.36-3.88-.99l-.28-.15-2.89.85.85-2.89-.15-.28C5.36 14.73 5 13.41 5 12c0-4.41 3.59-8 8-8s8 3.59 8 8-3.59 8-8 8z"/>
        <path d="M7 9h10v2H7zm0 3h7v2H7z"/>
      </svg>
    `;

    // Create chat window
    this.elements.chat = document.createElement('div');
    this.elements.chat.className = 'ii-agent-widget-chat';
    this.elements.chat.style.display = 'none';
    this.elements.chat.innerHTML = `
      <div class="ii-agent-header">
        <h3>Chat Assistant</h3>
        <button class="ii-agent-close">×</button>
      </div>
      <div class="ii-agent-messages" id="ii-agent-messages">
        <div class="ii-agent-message assistant">
          <div class="ii-agent-message-content">
            ${this.config.welcomeMessage}
          </div>
        </div>
      </div>
      <div class="ii-agent-input-container">
        <input 
          type="text" 
          class="ii-agent-input" 
          placeholder="${this.config.placeholder}"
        />
        <button class="ii-agent-send">Send</button>
      </div>
    `;

    // Append elements
    this.container.appendChild(this.elements.button);
    this.container.appendChild(this.elements.chat);
    document.body.appendChild(this.container);

    // Store references
    this.elements.messagesContainer = this.elements.chat.querySelector('.ii-agent-messages') as HTMLDivElement;
    this.elements.input = this.elements.chat.querySelector('.ii-agent-input') as HTMLInputElement;
    this.elements.sendButton = this.elements.chat.querySelector('.ii-agent-send') as HTMLButtonElement;

    // Inject styles
    this.injectStyles();
  }

  private attachEventListeners(): void {
    // Toggle button
    this.elements.button?.addEventListener('click', () => this.toggle());

    // Close button
    const closeButton = this.elements.chat?.querySelector('.ii-agent-close') as HTMLButtonElement;
    closeButton?.addEventListener('click', () => this.close());

    // Send button
    this.elements.sendButton?.addEventListener('click', () => this.sendMessage());

    // Enter key
    this.elements.input?.addEventListener('keypress', (e: KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
  }

  private injectStyles(): void {
    const styleId = 'ii-agent-widget-styles';
    
    if (document.getElementById(styleId)) return;

    const styles = document.createElement('style');
    styles.id = styleId;
    styles.textContent = `
      .ii-agent-widget {
        position: fixed;
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }

      .ii-agent-widget.bottom-right {
        bottom: 20px;
        right: 20px;
      }

      .ii-agent-widget.bottom-left {
        bottom: 20px;
        left: 20px;
      }

      .ii-agent-widget.top-right {
        top: 20px;
        right: 20px;
      }

      .ii-agent-widget.top-left {
        top: 20px;
        left: 20px;
      }

      .ii-agent-widget-button {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: ${this.config.primaryColor};
        color: white;
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, box-shadow 0.2s;
      }

      .ii-agent-widget-button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
      }

      .ii-agent-widget-chat {
        position: absolute;
        width: ${this.config.width}px;
        height: ${this.config.height}px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .ii-agent-widget.bottom-right .ii-agent-widget-chat,
      .ii-agent-widget.bottom-left .ii-agent-widget-chat {
        bottom: 80px;
      }

      .ii-agent-widget.top-right .ii-agent-widget-chat,
      .ii-agent-widget.top-left .ii-agent-widget-chat {
        top: 80px;
      }

      .ii-agent-widget.bottom-right .ii-agent-widget-chat,
      .ii-agent-widget.top-right .ii-agent-widget-chat {
        right: 0;
      }

      .ii-agent-widget.bottom-left .ii-agent-widget-chat,
      .ii-agent-widget.top-left .ii-agent-widget-chat {
        left: 0;
      }

      .ii-agent-header {
        padding: 16px;
        background: ${this.config.primaryColor};
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .ii-agent-header h3 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
      }

      .ii-agent-close {
        background: none;
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        transition: background 0.2s;
      }

      .ii-agent-close:hover {
        background: rgba(255, 255, 255, 0.2);
      }

      .ii-agent-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        background: #f9f9f9;
      }

      .ii-agent-message {
        margin-bottom: 12px;
        display: flex;
        align-items: flex-start;
      }

      .ii-agent-message.user {
        justify-content: flex-end;
      }

      .ii-agent-message-content {
        max-width: 70%;
        padding: 10px 14px;
        border-radius: 18px;
        word-wrap: break-word;
      }

      .ii-agent-message.assistant .ii-agent-message-content {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
      }

      .ii-agent-message.user .ii-agent-message-content {
        background: ${this.config.primaryColor};
        color: white;
      }

      .ii-agent-input-container {
        padding: 16px;
        background: white;
        border-top: 1px solid #e0e0e0;
        display: flex;
        gap: 8px;
      }

      .ii-agent-input {
        flex: 1;
        padding: 10px 14px;
        border: 1px solid #ddd;
        border-radius: 24px;
        font-size: 14px;
        outline: none;
        transition: border-color 0.2s;
      }

      .ii-agent-input:focus {
        border-color: ${this.config.primaryColor};
      }

      .ii-agent-send {
        padding: 8px 20px;
        background: ${this.config.primaryColor};
        color: white;
        border: none;
        border-radius: 24px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
      }

      .ii-agent-send:hover {
        opacity: 0.9;
      }

      .ii-agent-send:disabled {
        background: #ccc;
        cursor: not-allowed;
      }

      /* Dark theme */
      .ii-agent-widget.dark .ii-agent-widget-chat {
        background: #1e1e1e;
        color: white;
      }

      .ii-agent-widget.dark .ii-agent-messages {
        background: #2a2a2a;
      }

      .ii-agent-widget.dark .ii-agent-message.assistant .ii-agent-message-content {
        background: #3a3a3a;
        color: white;
        border-color: #4a4a4a;
      }

      .ii-agent-widget.dark .ii-agent-input-container {
        background: #1e1e1e;
        border-top-color: #3a3a3a;
      }

      .ii-agent-widget.dark .ii-agent-input {
        background: #2a2a2a;
        color: white;
        border-color: #3a3a3a;
      }

      /* Loading animation */
      .ii-agent-loading {
        display: inline-block;
        width: 12px;
        height: 12px;
        border: 2px solid rgba(0, 0, 0, 0.1);
        border-radius: 50%;
        border-top-color: ${this.config.primaryColor};
        animation: ii-agent-spin 1s ease-in-out infinite;
      }

      @keyframes ii-agent-spin {
        to { transform: rotate(360deg); }
      }
    `;

    document.head.appendChild(styles);
  }

  public toggle(): void {
    this.isOpen = !this.isOpen;
    this.updateUI();
  }

  public open(): void {
    this.isOpen = true;
    this.updateUI();
  }

  public close(): void {
    this.isOpen = false;
    this.updateUI();
  }

  private updateUI(): void {
    if (this.elements.chat) {
      this.elements.chat.style.display = this.isOpen ? 'flex' : 'none';
    }
    if (this.elements.button) {
      this.elements.button.style.display = this.isOpen ? 'none' : 'flex';
    }
  }

  private async sendMessage(content?: string): Promise<void> {
    const messageContent = content || this.elements.input?.value.trim();
    
    if (!messageContent) return;

    // Clear input
    if (this.elements.input) {
      this.elements.input.value = '';
    }

    // Add user message
    this.addMessageToUI('user', messageContent);

    // Show loading
    const loadingId = this.addMessageToUI('assistant', '<span class="ii-agent-loading"></span>');

    try {
      // Send to API
      const response = await this.client.chat(
        this.messages.map(m => ({ role: m.role, content: m.content }))
      );

      // Remove loading message
      this.removeMessageFromUI(loadingId);

      // Add assistant response
      this.addMessageToUI('assistant', response.content);

    } catch (error) {
      // Remove loading message
      this.removeMessageFromUI(loadingId);

      // Show error message
      this.addMessageToUI('assistant', 'Sorry, I encountered an error. Please try again.');
      console.error('Chat error:', error);
    }
  }

  private addMessageToUI(role: 'user' | 'assistant', content: string): string {
    const message: Message = {
      id: `msg-${Date.now()}-${Math.random()}`,
      role,
      content,
      timestamp: new Date(),
    };

    this.messages.push(message);

    const messageEl = document.createElement('div');
    messageEl.className = `ii-agent-message ${role}`;
    messageEl.id = message.id;
    messageEl.innerHTML = `
      <div class="ii-agent-message-content">
        ${content}
      </div>
    `;

    this.elements.messagesContainer?.appendChild(messageEl);
    
    // Scroll to bottom
    if (this.elements.messagesContainer) {
      this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }

    return message.id;
  }

  private removeMessageFromUI(messageId: string): void {
    const messageEl = document.getElementById(messageId);
    messageEl?.remove();
    
    this.messages = this.messages.filter(m => m.id !== messageId);
  }

  public destroy(): void {
    this.container?.remove();
  }
}

// Global initialization
declare global {
  interface Window {
    IIAgentChat: {
      init: (config: ChatConfig) => ChatWidget;
      instances: Map<string, ChatWidget>;
    };
  }
}

if (typeof window !== 'undefined') {
  window.IIAgentChat = {
    init: (config: ChatConfig) => {
      const widget = new ChatWidget(config);
      window.IIAgentChat.instances.set(config.projectId, widget);
      return widget;
    },
    instances: new Map(),
  };
}

export default ChatWidget;