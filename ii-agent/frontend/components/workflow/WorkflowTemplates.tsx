// components/rowboat/workflow/WorkflowTemplates.tsx

import React from 'react';
import { 
  Brain,
  MessageSquare,
  Search,
  FileText,
  Code,
  Shield,
  BarChart3,
  Users,
  Sparkles,
  Globe,
  Database,
  GitBranch
} from 'lucide-react';
import type { WorkflowDefinition } from '@/types/rowboat/workflow.types';

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  category: string;
  tags: string[];
  workflow: WorkflowDefinition;
}

interface WorkflowTemplatesProps {
  onSelectTemplate: (template: WorkflowDefinition) => void;
}

export const WorkflowTemplates: React.FC<WorkflowTemplatesProps> = ({ onSelectTemplate }) => {
  const templates: WorkflowTemplate[] = [
    {
      id: 'customer-support',
      name: 'Customer Support System',
      description: 'Multi-tier support system that routes inquiries to specialized agents',
      icon: <MessageSquare className="w-6 h-6" />,
      category: 'Support',
      tags: ['customer service', 'routing', 'classification'],
      workflow: {
        name: 'Customer Support Workflow',
        description: 'Routes customer inquiries to billing or technical support based on content',
        agents: [
          {
            name: 'intake_agent',
            role: 'coordinator',
            instructions: 'Analyze incoming customer inquiries and determine if they are billing-related or technical issues. Route to @billing_specialist for payment/subscription issues or @technical_support for product/feature issues.',
            tools: ['text_classifier', 'sentiment_analyzer'],
            outputVisibility: 'internal'
          },
          {
            name: 'billing_specialist',
            role: 'specialist',
            instructions: 'Handle all billing, payment, and subscription-related inquiries. Access customer account information and process refunds or subscription changes.',
            tools: ['payment_processor', 'account_lookup', 'subscription_manager'],
            outputVisibility: 'user_facing'
          },
          {
            name: 'technical_support',
            role: 'specialist',
            instructions: 'Resolve technical issues, bugs, and feature questions. Search documentation and provide step-by-step solutions.',
            tools: ['knowledge_base_search', 'debug_analyzer', 'solution_generator'],
            outputVisibility: 'user_facing'
          },
          {
            name: 'escalation_manager',
            role: 'reviewer',
            instructions: 'Handle complex cases that require human intervention or management approval. Track escalated issues.',
            tools: ['ticket_creator', 'priority_analyzer'],
            outputVisibility: 'internal'
          }
        ],
        edges: [
          { from_agent: 'intake_agent', to_agent: 'billing_specialist', isMentionBased: true },
          { from_agent: 'intake_agent', to_agent: 'technical_support', isMentionBased: true },
          { from_agent: 'billing_specialist', to_agent: 'escalation_manager', condition: 'requires_escalation' },
          { from_agent: 'technical_support', to_agent: 'escalation_manager', condition: 'requires_escalation' }
        ],
        entry_point: 'intake_agent',
        metadata: { template_id: 'customer-support' }
      }
    },
    {
      id: 'research-assistant',
      name: 'Research Assistant',
      description: 'Comprehensive research workflow with fact-checking and report generation',
      icon: <Search className="w-6 h-6" />,
      category: 'Research',
      tags: ['research', 'analysis', 'reporting'],
      workflow: {
        name: 'Research Assistant Workflow',
        description: 'Research topics, verify facts, and create comprehensive reports',
        agents: [
          {
            name: 'research_coordinator',
            role: 'coordinator',
            instructions: 'Break down research queries into subtopics. Assign @web_researcher to gather information and @fact_checker to verify claims.',
            tools: ['query_analyzer', 'topic_extractor'],
            outputVisibility: 'internal'
          },
          {
            name: 'web_researcher',
            role: 'researcher',
            instructions: 'Search the web for relevant information on assigned topics. Gather data from multiple sources and summarize findings.',
            tools: ['web_search', 'content_extractor', 'summarizer'],
            outputVisibility: 'internal'
          },
          {
            name: 'fact_checker',
            role: 'reviewer',
            instructions: 'Verify claims and data points from research. Cross-reference with authoritative sources and flag any inconsistencies.',
            tools: ['fact_verification', 'source_validator', 'citation_generator'],
            outputVisibility: 'internal'
          },
          {
            name: 'report_writer',
            role: 'writer',
            instructions: 'Compile verified research into a well-structured report with proper citations and executive summary.',
            tools: ['markdown_formatter', 'citation_manager', 'report_template'],
            outputVisibility: 'user_facing'
          }
        ],
        edges: [
          { from_agent: 'research_coordinator', to_agent: 'web_researcher', isMentionBased: true },
          { from_agent: 'web_researcher', to_agent: 'fact_checker' },
          { from_agent: 'fact_checker', to_agent: 'report_writer' },
          { from_agent: 'research_coordinator', to_agent: 'fact_checker', isMentionBased: true }
        ],
        entry_point: 'research_coordinator',
        metadata: { template_id: 'research-assistant' }
      }
    },
    {
      id: 'code-review',
      name: 'Code Review Pipeline',
      description: 'Automated code review with security, performance, and style checks',
      icon: <Code className="w-6 h-6" />,
      category: 'Development',
      tags: ['code review', 'security', 'quality assurance'],
      workflow: {
        name: 'Code Review Workflow',
        description: 'Comprehensive code review checking syntax, security, performance, and architecture',
        agents: [
          {
            name: 'review_coordinator',
            role: 'coordinator',
            instructions: 'Analyze incoming code changes and distribute to specialized reviewers: @syntax_checker, @security_reviewer, @performance_analyst.',
            tools: ['diff_analyzer', 'file_classifier'],
            outputVisibility: 'internal'
          },
          {
            name: 'syntax_checker',
            role: 'analyzer',
            instructions: 'Check code syntax, style compliance, and common anti-patterns. Suggest improvements for readability.',
            tools: ['linter', 'style_checker', 'complexity_analyzer'],
            outputVisibility: 'user_facing'
          },
          {
            name: 'security_reviewer',
            role: 'specialist',
            instructions: 'Scan for security vulnerabilities, check for proper authentication/authorization, and identify potential attack vectors.',
            tools: ['security_scanner', 'dependency_checker', 'vulnerability_database'],
            outputVisibility: 'user_facing'
          },
          {
            name: 'performance_analyst',
            role: 'analyzer',
            instructions: 'Analyze performance implications, identify bottlenecks, and suggest optimizations.',
            tools: ['performance_profiler', 'complexity_calculator', 'benchmark_runner'],
            outputVisibility: 'user_facing'
          },
          {
            name: 'review_summarizer',
            role: 'writer',
            instructions: 'Compile all review feedback into a clear summary with prioritized action items.',
            tools: ['report_generator', 'priority_ranker'],
            outputVisibility: 'user_facing'
          }
        ],
        edges: [
          { from_agent: 'review_coordinator', to_agent: 'syntax_checker', isMentionBased: true },
          { from_agent: 'review_coordinator', to_agent: 'security_reviewer', isMentionBased: true },
          { from_agent: 'review_coordinator', to_agent: 'performance_analyst', isMentionBased: true },
          { from_agent: 'syntax_checker', to_agent: 'review_summarizer' },
          { from_agent: 'security_reviewer', to_agent: 'review_summarizer' },
          { from_agent: 'performance_analyst', to_agent: 'review_summarizer' }
        ],
        entry_point: 'review_coordinator',
        metadata: { template_id: 'code-review' }
      }
    },
    {
      id: 'content-creation',
      name: 'Content Creation Pipeline',
      description: 'End-to-end content creation with SEO optimization and multi-channel publishing',
      icon: <FileText className="w-6 h-6" />,
      category: 'Marketing',
      tags: ['content', 'seo', 'publishing'],
      workflow: {
        name: 'Content Creation Workflow',
        description: 'Create, optimize, and publish content across multiple channels',
        agents: [
          {
            name: 'content_strategist',
            role: 'coordinator',
            instructions: 'Analyze content requirements and create content brief. Assign @content_writer to create draft and @seo_optimizer for optimization.',
            tools: ['keyword_research', 'competitor_analysis', 'topic_planner'],
            outputVisibility: 'internal'
          },
          {
            name: 'content_writer',
            role: 'writer',
            instructions: 'Create engaging content based on brief. Focus on target audience and maintain brand voice.',
            tools: ['grammar_checker', 'readability_analyzer', 'tone_adjuster'],
            outputVisibility: 'internal'
          },
          {
            name: 'seo_optimizer',
            role: 'specialist',
            instructions: 'Optimize content for search engines. Add meta descriptions, optimize headings, and ensure keyword density.',
            tools: ['seo_analyzer', 'keyword_optimizer', 'meta_generator'],
            outputVisibility: 'internal'
          },
          {
            name: 'content_publisher',
            role: 'specialist',
            instructions: 'Format and publish content across specified channels. Track publication status.',
            tools: ['cms_publisher', 'social_media_scheduler', 'analytics_tracker'],
            outputVisibility: 'user_facing'
          }
        ],
        edges: [
          { from_agent: 'content_strategist', to_agent: 'content_writer', isMentionBased: true },
          { from_agent: 'content_writer', to_agent: 'seo_optimizer' },
          { from_agent: 'seo_optimizer', to_agent: 'content_publisher' },
          { from_agent: 'content_strategist', to_agent: 'seo_optimizer', isMentionBased: true }
        ],
        entry_point: 'content_strategist',
        metadata: { template_id: 'content-creation' }
      }
    }
  ];

  const categories = Array.from(new Set(templates.map(t => t.category)));

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Workflow Templates</h2>
        <p className="text-gray-600">Start with a pre-built workflow template and customize it to your needs</p>
      </div>

      {categories.map(category => (
        <div key={category}>
          <h3 className="text-lg font-semibold mb-3 text-gray-700">{category}</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {templates
              .filter(t => t.category === category)
              .map(template => (
                <button
                  key={template.id}
                  onClick={() => onSelectTemplate(template.workflow)}
                  className="bg-white border-2 border-gray-200 rounded-lg p-6 hover:border-blue-500 hover:shadow-lg transition-all text-left group"
                >
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-blue-100 rounded-lg text-blue-600 group-hover:bg-blue-200 transition-colors">
                      {template.icon}
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-lg mb-1">{template.name}</h4>
                      <p className="text-gray-600 text-sm mb-3">{template.description}</p>
                      <div className="flex items-center justify-between">
                        <div className="flex flex-wrap gap-1">
                          {template.tags.map(tag => (
                            <span
                              key={tag}
                              className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-500">
                          <Users className="w-3 h-3" />
                          {template.workflow.agents.length} agents
                          <GitBranch className="w-3 h-3 ml-1" />
                          {template.workflow.edges.length} connections
                        </div>
                      </div>
                    </div>
                  </div>
                </button>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
};