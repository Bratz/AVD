import { registerTool } from "./ag-ui-client"
import {
  MetricCard,
  Chart,
  DataTable,
  DataGrid,
  ConfirmationCard,
  UserForm,
  ToggleSwitch,
  InfoBanner,
  ProgressBar,
  AvatarCard,
  Timeline,
  MultiStepForm,
  SearchWithFilters,
  DateTimeRangePicker,
  RatingSelector,
  KanbanBoard,
  ChecklistWithProgress,
  ApprovalWorkflowCard,
  TeamMemberList,
  ProductCatalogGrid,
  CartSummaryPanel,
  PaymentDetailsForm,
  MessageFeed,
  OrderStatusTracker,
  EditableDataTable,
  CrudDataTable,
  ExpandableRowTable,
  ColumnToggleTable,
  LocationMap,
  RoutePlannerMap,
  ThreadedComments,
  MentionInput,
  TabLayout,
  AccordionContent,
  MarkdownRenderer,
  CodeSnippetViewer,
  ColorPickerPopover,
  ImageGallery,
  EnvironmentSwitcher,
  LanguageSelector,
  ThemeToggle,
  ToastStack,
  ModalPrompt,
  OrgChartViewer,
  AIPromptBuilder,
} from "./components"

// Import ROWBOAT components - using the enhanced versions with original names
import {
  EnhancedROWBOATDashboard as ROWBOATDashboard,
  StreamingWorkflowBuilder as WorkflowBuilder,
  EnhancedWorkflowVisualEditor as WorkflowVisualEditor,
  StreamingWorkflowExecutor as WorkflowExecutor,
  EnhancedWorkflowList as WorkflowList,
} from "@/components/rowboat/rowboat-integration"

// Import the WorkflowChat component and template gallery
import { WorkflowChat, WorkflowTemplateGallery } from "@/components/rowboat/workflow-chat"

// ===== ADD THIS DEBUG SECTION =====
console.log('ROWBOAT Import Check:', {
  ROWBOATDashboard: typeof ROWBOATDashboard,
  WorkflowBuilder: typeof WorkflowBuilder,
  WorkflowVisualEditor: typeof WorkflowVisualEditor,
  WorkflowExecutor: typeof WorkflowExecutor,
  WorkflowList: typeof WorkflowList,
  WorkflowChat: typeof WorkflowChat,
  WorkflowTemplateGallery: typeof WorkflowTemplateGallery,
});

// Check which are undefined
const rowboatComponents = {
  ROWBOATDashboard,
  WorkflowBuilder,
  WorkflowVisualEditor,
  WorkflowExecutor,
  WorkflowList,
  WorkflowChat,
  WorkflowTemplateGallery,
};

const undefinedROWBOAT = Object.entries(rowboatComponents)
  .filter(([name, component]) => component === undefined)
  .map(([name]) => name);

if (undefinedROWBOAT.length > 0) {
  console.error('❌ UNDEFINED ROWBOAT COMPONENTS:', undefinedROWBOAT);
} else {
  console.log('✅ All ROWBOAT components imported successfully');
}
// ===== END DEBUG SECTION =====

// Create a registry object to export
export const registry = {
  MetricCard,
  Chart,
  DataTable,
  DataGrid,
  ConfirmationCard,
  UserForm,
  ToggleSwitch,
  InfoBanner,
  ProgressBar,
  AvatarCard,
  Timeline,
  MultiStepForm,
  SearchWithFilters,
  DateTimeRangePicker,
  RatingSelector,
  KanbanBoard,
  ChecklistWithProgress,
  ApprovalWorkflowCard,
  TeamMemberList,
  ProductCatalogGrid,
  CartSummaryPanel,
  PaymentDetailsForm,
  MessageFeed,
  OrderStatusTracker,
  EditableDataTable,
  CrudDataTable,
  ExpandableRowTable,
  ColumnToggleTable,
  LocationMap,
  RoutePlannerMap,
  ThreadedComments,
  MentionInput,
  TabLayout,
  AccordionContent,
  MarkdownRenderer,
  CodeSnippetViewer,
  ColorPickerPopover,
  ImageGallery,
  EnvironmentSwitcher,
  LanguageSelector,
  ThemeToggle,
  ToastStack,
  ModalPrompt,
  OrgChartViewer,
  AIPromptBuilder,
  // ROWBOAT components
  ROWBOATDashboard,
  WorkflowBuilder,
  WorkflowVisualEditor,
  WorkflowExecutor,
  WorkflowList,
  WorkflowChat,
  WorkflowTemplateGallery,
}

// ===== ADD THIS CHECK =====
console.log('Registry ROWBOAT components:', {
  'registry.ROWBOATDashboard': !!registry.ROWBOATDashboard,
  'typeof': typeof registry.ROWBOATDashboard,
});

/**
 * Registers all agentic UI components with the AG-UI protocol
 * This makes all components available for agents to use
 */
export function registerAllAgentComponents() {
  console.log("Registering all agent components...")

  // Create a map of component names to components
  const componentMap: Record<string, React.ComponentType<any>> = {
    // Ensure all component names are lowercase for consistency
    metriccard: MetricCard,
    chart: Chart,
    datatable: DataTable,
    datagrid: DataGrid,
    confirmationcard: ConfirmationCard,
    userform: UserForm,
    toggleswitch: ToggleSwitch,
    infobanner: InfoBanner,
    progressbar: ProgressBar,
    avatarcard: AvatarCard,
    timeline: Timeline,
    multistepform: MultiStepForm,
    searchwithfilters: SearchWithFilters,
    datetimerangepicker: DateTimeRangePicker,
    ratingselector: RatingSelector,
    kanbanboard: KanbanBoard,
    checklistwithprogress: ChecklistWithProgress,
    approvalworkflowcard: ApprovalWorkflowCard,
    teammemberlist: TeamMemberList,
    productcataloggrid: ProductCatalogGrid,
    cartsummarypanel: CartSummaryPanel,
    paymentdetailsform: PaymentDetailsForm,
    messagefeed: MessageFeed,
    orderstatustracker: OrderStatusTracker,
    editabledatatable: EditableDataTable,
    cruddatatable: CrudDataTable,
    expandablerowtable: ExpandableRowTable,
    columntoggletable: ColumnToggleTable,
    locationmap: LocationMap,
    routeplannermap: RoutePlannerMap,
    threadedcomments: ThreadedComments,
    mentioninput: MentionInput,
    tablayout: TabLayout,
    accordioncontent: AccordionContent,
    markdownrenderer: MarkdownRenderer,
    codesnippetviewer: CodeSnippetViewer,
    colorpickerpopover: ColorPickerPopover,
    imagegallery: ImageGallery,
    environmentswitcher: EnvironmentSwitcher,
    languageselector: LanguageSelector,
    themetoggle: ThemeToggle,
    toaststack: ToastStack,
    modalprompt: ModalPrompt,
    orgchartviewer: OrgChartViewer,
    aipromptbuilder: AIPromptBuilder,
    // ROWBOAT components
    rowboatdashboard: ROWBOATDashboard,
    workflowbuilder: WorkflowBuilder,
    workflowvisualeditor: WorkflowVisualEditor,
    workflowexecutor: WorkflowExecutor,
    workflowlist: WorkflowList,
    workflowchat: WorkflowChat,
    workflowtemplategallery: WorkflowTemplateGallery,
  }

  console.log("Components to register:", Object.keys(componentMap).length)

  // Register each component and log for debugging
  Object.entries(componentMap).forEach(([name, Component]) => {
    try {
      registerTool(name, Component)
      console.log(`Registered component: ${name}`)
    } catch (error) {
      console.error(`Failed to register component ${name}:`, error)
    }
  })

  console.log("All components registered successfully")
}