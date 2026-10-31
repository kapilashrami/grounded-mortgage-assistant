const list = document.querySelector('#turns');
const detail = document.querySelector('#detail');
const status = document.querySelector('#list-status');
const summary = document.querySelector('#summary');
const consoleRoot = document.querySelector('.console');
let selected = new URLSearchParams(location.search).get('trace');
let traces = [];
let detailRequest = 0;
let activeTab = 'input';
let selectedStage = null;
let selectedFlowNode = null;
let openEvents = new Set();

const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const value = item => item === null || item === undefined ? 'Unknown' : String(item);
const get = async path => {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(response.status === 404 ? 'This trace has expired.' : 'Local trace unavailable.');
  return response.json();
};
const number = amount => Number.isFinite(amount) ? amount.toLocaleString('en-US') : null;
const duration = ms => !Number.isFinite(ms) ? null : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
const shortId = id => String(id || '').slice(0, 8);
// Mortgage turns are stored as a safe projection: readable, with amounts masked.
const displayQuestion = trace => {
  if (trace.question) return trace.question;
  return trace.status === 'REJECTED' ? 'Message withheld by the privacy check' : 'No retained question';
};
const isProjection = trace => Boolean(trace.question) && trace.question_projection === true;
const sentence = text => { const value = String(text || '').toLowerCase(); return value.charAt(0).toUpperCase() + value.slice(1); };
const usd = amount => Number.isFinite(amount) ? `$${Number(amount.toFixed(4))}` : null;
const pill = (text, className = '', title) => {
  const item = element('span', text, `pill ${className}`.trim());
  if (title) item.title = title;
  return item;
};
const textBlock = text => element('p', text || 'Not recorded', 'answer');
const section = (title, content, open = true) => {
  const details = element('details', undefined, 'io-section');
  details.open = open;
  details.append(element('summary', title), content);
  return details;
};
const dataList = entries => {
  const list = element('dl', undefined, 'data-list');
  entries.forEach(([label, content]) => list.append(element('dt', label), element('dd', value(content))));
  return list;
};
const eventRow = (event, index) => {
  const details = element('details', undefined, 'event-row');
  details.open = openEvents.has(index);
  details.addEventListener('toggle', () => {
    if (details.open) openEvents.add(index);
    else openEvents.delete(index);
  });
  const row = element('summary');
  row.append(
    element('strong', event.name || 'Recorded event'),
    element('span', event.result || 'recorded'),
    element('span', event.duration_ms === undefined ? 'No duration' : `${event.duration_ms} ms`)
  );
  details.append(row, element('pre', JSON.stringify(event, null, 2), 'event-json'));
  return details;
};

function renderSummary() {
  const tokenSummary = field => {
    const unknown = traces.some(trace => !Number.isFinite(trace.usage?.[field]));
    const known = traces.reduce((total, trace) => total + (Number.isFinite(trace.usage?.[field]) ? trace.usage[field] : 0), 0);
    return unknown ? `${number(known)} partial` : number(known);
  };
  summary.replaceChildren(
    pill(`${traces.length} trace${traces.length === 1 ? '' : 's'}`),
    pill(`${tokenSummary('input_tokens')} in`, '', 'Input tokens'),
    pill(`${tokenSummary('output_tokens')} out`, '', 'Output tokens'),
    pill(`${costSummary()} external API`, 'cost')
  );
}

function costSummary() {
  const unknown = traces.some(trace => !Number.isFinite(trace.cost?.external_api_usd));
  const known = traces.reduce((total, trace) => total + (Number.isFinite(trace.cost?.external_api_usd) ? trace.cost.external_api_usd : 0), 0);
  const amount = `$${Number(known.toFixed(4))}`;
  return unknown ? `${amount} partial` : amount;
}

function renderList() {
  list.replaceChildren();
  const groups = new Map();
  traces.forEach(trace => groups.set(trace.session_id, [...(groups.get(trace.session_id) || []), trace]));
  for (const [sessionId, turns] of groups) {
    const heading = element('p', undefined, 'session-heading');
    heading.title = `Session ${sessionId}`;
    heading.append(element('span', 'Session'), element('code', shortId(sessionId)), element('span', `${turns.length}`, 'session-count'));
    list.append(heading);
    turns.forEach(trace => {
      const button = element('button', undefined, 'turn');
      button.type = 'button';
      button.setAttribute('aria-pressed', String(trace.trace_id === selected));
      const tone = outcomeTone(trace.status);
      const meta = element('span', undefined, 'turn-meta');
      meta.append(element('span', value(trace.status), `turn-status ${tone}`), element('span', duration(trace.duration_ms) || '', 'turn-time'));
      const question = element('span', displayQuestion(trace), 'turn-question');
      if (isProjection(trace)) question.title = 'Safe projection of the message: amounts are masked';
      button.append(element('i', undefined, `status-dot ${tone}`), question, meta);
      button.addEventListener('click', () => show(trace.trace_id));
      list.append(button);
    });
  }
}

function createTabs(trace, inspector) {
  const tabs = element('div', undefined, 'tabs');
  tabs.role = 'tablist';
  const retrievalEvents = (trace.events || []).filter(event => event.name === 'frozen_retrieval');
  const specs = [
    ['input', 'Input & output'],
    ['flow', 'Flow'],
    ['evidence', `Evidence (${retrievalEvents.reduce((count, event) => count + (event.final_ids?.length || 0), 0)})`],
    ['diagnostics', 'Diagnostics'],
    ['metadata', 'Metadata'],
    ['raw', 'Raw'],
  ];
  specs.forEach(([name, label]) => {
    const tab = element('button', label, 'tab');
    tab.type = 'button'; tab.role = 'tab'; tab.id = `tab-${name}`; tab.dataset.tab = name;
    tab.setAttribute('aria-controls', 'tab-panel');
    tab.addEventListener('click', () => selectTab(trace, name, inspector));
    tabs.append(tab);
  });
  tabs.addEventListener('keydown', event => {
    const all = [...tabs.querySelectorAll('[role=tab]')];
    const at = all.indexOf(document.activeElement);
    let next = at;
    if (event.key === 'ArrowRight') next = (at + 1) % all.length;
    else if (event.key === 'ArrowLeft') next = (at - 1 + all.length) % all.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = all.length - 1;
    else return;
    event.preventDefault();
    all[next].focus();
    selectTab(trace, all[next].dataset.tab, inspector);
  });
  return tabs;
}

const FLOW_LABELS = {
  privacy: 'Privacy check',
  public_projection: 'Public question check',
  explicit_ltv_request: 'Explicit LTV request',
  interpret_turn: 'Interpret request',
  select_evidence: 'Select evidence',
  frozen_retrieval: 'Retrieve evidence',
  memory_commit: 'Save to session',
  clarify_decision: 'Clarification here',
};
const FLOW_LAYERS = {
  evidence: 'Evidence check',
  public_projection: 'Public question routing',
  missing_fact: 'Required detail check',
  calculation: 'Calculation check',
  draft: 'Draft preparation',
  interpretation: 'Request interpretation',
  source_validation: 'Source validation',
  memory_validation: 'Memory validation',
};
const FLOW_REASONS = {
  ambiguous: 'Public topic needs clarification',
  unsupported: 'Public topic is unsupported',
  evidence_unavailable: 'Evidence is unavailable',
  calculation_inputs: 'Loan amount or property value is missing',
  calculation_unavailable: 'Calculation details are unavailable',
  draft_unavailable: 'Draft details are unavailable',
  ambiguous_or_negated: 'The request is ambiguous or negated',
  proposal_rejected: 'The proposed memory update was rejected',
  pending_question: 'A required detail is missing',
  scope_unavailable: 'Draft scope is unavailable',
  interpretation_unavailable: 'Request interpretation is unavailable',
};
const PRIVACY_REASONS = {
  email_address: 'Email address detected',
  personal_information: 'Personal information detected',
  detector_unavailable: 'Privacy check unavailable',
};
const ERROR_MESSAGES = {
  SENSITIVE_TEXT: 'Privacy controls rejected this message.',
  PII_CHECK_UNAVAILABLE: 'Privacy check unavailable.',
  PROCESSING_FAILED: 'The request could not be processed.',
  UNSUPPORTED_TEXT: 'This message format is unsupported.',
  INVALID_MONEY: 'The amount could not be processed.',
};
const STAGE_ROLES = {
  interpret_turn: { label: 'LLM', kind: 'llm' },
  select_evidence: { label: 'LLM', kind: 'llm' },
  frozen_retrieval: { label: 'RAG', kind: 'rag' },
  public_projection: { label: 'Routing', kind: 'routing' },
  explicit_ltv_request: { label: 'Routing', kind: 'routing' },
  clarify_decision: { label: 'Routing', kind: 'routing' },
  privacy: { label: 'Guardrail', kind: 'guardrail' },
  memory_commit: { label: 'State', kind: 'state' },
};
function stageRole(event) {
  return STAGE_ROLES[event.name] || { label: 'Recorded', kind: 'recorded' };
}
function roleBadge(event) {
  const role = stageRole(event);
  return element('span', role.label, `role-badge ${role.kind}`);
}
function eventResult(event) {
  return event.result || event.kind || 'recorded';
}
function modelEvents(events) {
  return events.filter(event => stageRole(event).kind === 'llm');
}
function retrievalEvents(events) {
  return events.filter(event => stageRole(event).kind === 'rag');
}
function explicitEarlyStop(trace, events) {
  const privacyBlocked = trace.status === 'REJECTED'
    && events.some(event => event.name === 'privacy' && ['blocked', 'unavailable'].includes(event.result));
  const publicClarification = events.some(event => event.name === 'public_projection'
    && ['ambiguous', 'unsupported'].includes(event.kind));
  return privacyBlocked || publicClarification;
}
function roleSummaryCard(label, kind, events, trace, stopped) {
  const card = element('section', undefined, `role-summary ${kind}`);
  card.append(element('strong', label));
  if (!events.length) {
    card.append(element('span', stopped ? `${label} not reached` : `No ${label} event recorded`));
    return card;
  }
  const outcomes = events.map(eventResult).join(', ');
  card.append(element('span', `${events.length} retained stage${events.length === 1 ? '' : 's'}: ${outcomes}`));
  if (kind === 'rag') {
    const candidates = events.reduce((count, event) => count + (Number.isFinite(event.count) ? event.count : 0), 0);
    const selected = Number.isFinite(trace.retrieval?.count) ? trace.retrieval.count : 'Unknown';
    card.append(element('small', `${candidates} retrieved candidates, ${selected} selected evidence`));
  }
  return card;
}
function flowRoleSummary(trace, events) {
  const summary = element('section', undefined, 'role-summary-strip');
  const llm = modelEvents(events);
  const rag = retrievalEvents(events);
  const stopped = !llm.length && !rag.length && explicitEarlyStop(trace, events);
  summary.append(
    roleSummaryCard('LLM', 'llm', llm, trace, stopped),
    roleSummaryCard('RAG', 'rag', rag, trace, stopped)
  );
  return summary;
}
function privacyReason(event) {
  if (event?.name !== 'privacy') return null;
  if (PRIVACY_REASONS[event.reason]) return PRIVACY_REASONS[event.reason];
  return ['blocked', 'unavailable'].includes(event.result) ? 'Privacy rejection reason was not recorded.' : null;
}
function tracePrivacyReason(trace) {
  return (trace.events || []).map(privacyReason).find(Boolean) || 'No reason recorded.';
}
function errorMessage(code) {
  return ERROR_MESSAGES[code] || 'A local processing error was recorded.';
}
function flowTone(event) {
  if (event.name === 'clarify_decision') return 'decision';
  if (['failure', 'error', 'failed', 'unavailable', 'blocked'].includes(String(event.result).toLowerCase())) return 'error';
  if (event.result === 'success' || event.result === 'pass') return 'success';
  return 'neutral';
}
function flowNode(event, index) {
  const node = element('button', undefined, `flow-node ${flowTone(event)}`);
  node.type = 'button';
  node.dataset.flowNode = String(index);
  node.setAttribute('aria-pressed', String(selectedFlowNode === index));
  const title = event.name === 'clarify_decision'
    ? (FLOW_LAYERS[event.layer] || 'Recorded decision layer')
    : (FLOW_LABELS[event.name] || 'Recorded event');
  const subtitle = event.name === 'clarify_decision'
    ? `Clarification here - ${FLOW_REASONS[event.reason] || 'Clarification was recorded'}`
    : event.name === 'privacy' && privacyReason(event)
      ? privacyReason(event)
    : (event.result || 'Recorded');
  const heading = element('strong', undefined, 'flow-node-title');
  heading.append(document.createTextNode(title), roleBadge(event));
  node.append(heading, element('span', subtitle));
  node.addEventListener('click', () => { selectedFlowNode = index; });
  return node;
}
function outcomeTone(status) {
  if (status === 'ANSWERED') return 'success';
  if (status === 'CLARIFY') return 'decision';
  if (status === 'ERROR' || status === 'REJECTED') return 'error';
  return 'neutral';
}
function outcomeNode(trace) {
  const node = element('button', undefined, `flow-node ${outcomeTone(trace.status)}`);
  node.type = 'button'; node.dataset.flowNode = 'outcome';
  node.setAttribute('aria-pressed', String(selectedFlowNode === 'outcome'));
  node.append(element('strong', 'Response outcome'), element('span', trace.status ? sentence(trace.status) : 'Not recorded'));
  return node;
}
function flowPanel(trace, inspector) {
  const panel = element('div', undefined, 'flow-panel');
  const events = trace.events || [];
  const decisions = events.filter(event => event.name === 'clarify_decision');
  panel.append(flowRoleSummary(trace, events));
  const diagram = element('div', undefined, 'flow-diagram');
  if (!events.length) diagram.append(element('p', 'No execution events were retained.', 'json-purpose'));
  events.forEach((event, index) => {
    const node = flowNode(event, index);
    node.addEventListener('click', () => {
      const detail = panel.querySelector('.flow-detail');
      const role = stageRole(event);
      const details = [
        ['Role', role.label],
        ['Result', eventResult(event)],
        ['Duration', Number.isFinite(event.duration_ms) ? `${event.duration_ms} ms` : 'Not recorded'],
      ];
      if (role.kind === 'llm') {
        details.push(
          ['Model', event.model || trace.model?.id || 'Not recorded'],
          ['Provider', trace.model?.provider || 'Not recorded'],
          ['Provider input tokens', Number.isFinite(event.provider_input_tokens) ? event.provider_input_tokens : 'Not recorded'],
          ['Output tokens', Number.isFinite(event.output_tokens) ? event.output_tokens : 'Not recorded']
        );
      }
      if (role.kind === 'rag') {
        details.push(
          ['Retrieved candidates', Number.isFinite(event.count) ? event.count : 'Not recorded'],
          ['Selected evidence', Number.isFinite(trace.retrieval?.count) ? trace.retrieval.count : 'Not recorded']
        );
      }
      if (event.name === 'privacy') details.push(['Reason', privacyReason(event) || 'No reason recorded.']);
      if (event.name === 'clarify_decision') {
        details.push(['Reason', FLOW_REASONS[event.reason] || 'Clarification was recorded.']);
      }
      const remaining = Object.entries(event).filter(([key]) => ![
        'name', 'result', 'kind', 'duration_ms', 'model', 'provider_input_tokens', 'output_tokens', 'count', 'reason',
      ].includes(key));
      details.push(...remaining.map(([key, content]) => [key, typeof content === 'object' ? JSON.stringify(content) : content]));
      const actions = element('div', undefined, 'flow-actions');
      if (role.kind === 'llm') {
        const action = element('button', 'View model diagnostics', 'flow-action');
        action.type = 'button';
        action.addEventListener('click', () => {
          selectedStage = event;
          selectTab(trace, 'diagnostics', inspector, true);
        });
        actions.append(action);
      }
      if (role.kind === 'rag') {
        const action = element('button', 'View retrieval evidence', 'flow-action');
        action.type = 'button';
        action.addEventListener('click', () => selectTab(trace, 'evidence', inspector, true));
        actions.append(action);
      }
      detail.replaceChildren(element('h4', event.name === 'clarify_decision' ? 'Clarification decision' : (FLOW_LABELS[event.name] || 'Recorded event')),
        dataList(details), actions);
      diagram.querySelectorAll('.flow-node').forEach(item => item.setAttribute('aria-pressed', String(item.dataset.flowNode === String(index))));
    });
    diagram.append(node);
  });
  const outcome = outcomeNode(trace);
  outcome.addEventListener('click', () => {
    selectedFlowNode = 'outcome';
    const detail = panel.querySelector('.flow-detail');
    detail.replaceChildren(
      element('h4', 'Response outcome'),
      dataList([
        ['Status', trace.status || 'Not recorded'],
        ['Reason', tracePrivacyReason(trace)],
        ['Error code', trace.error || 'None recorded'],
        ['Description', trace.error ? errorMessage(trace.error) : 'None recorded'],
      ])
    );
    diagram.querySelectorAll('.flow-node').forEach(item => item.setAttribute('aria-pressed', String(item.dataset.flowNode === 'outcome')));
  });
  diagram.append(outcome);
  panel.append(diagram);
  if (!decisions.length && trace.status === 'CLARIFY') panel.append(element('p', 'Decision layer not recorded for this legacy trace.', 'outside-note'));
  const detail = element('section', undefined, 'flow-detail');
  detail.append(element('p', 'Select a step to see its details.', 'json-purpose'));
  panel.append(detail);
  if (selectedFlowNode === 'outcome') {
    outcome.click();
  } else if (Number.isInteger(selectedFlowNode) && selectedFlowNode >= 0 && selectedFlowNode < events.length) {
    diagram.querySelector(`[data-flow-node="${selectedFlowNode}"]`).click();
  }
  return panel;
}

function selectTab(trace, name, inspector, focus = false) {
  activeTab = name;
  detail.querySelectorAll('[role=tab]').forEach(tab => {
    const on = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
  });
  const panel = element('section', undefined, 'tab-panel');
  panel.id = 'tab-panel'; panel.role = 'tabpanel'; panel.tabIndex = -1;
  panel.setAttribute('aria-labelledby', `tab-${name}`);
  const events = trace.events || [];
  const retrievalEvents = events.filter(event => event.name === 'frozen_retrieval');
  const ids = retrievalEvents.flatMap(event => Array.isArray(event.final_ids) ? event.final_ids : []);
  if (name === 'input') {
    panel.append(section(isProjection(trace) ? 'Input (safe projection)' : 'Input', textBlock(trace.question ? displayQuestion(trace) : null)), section('Output', textBlock(trace.answer)));
    if (trace.answer_clipped) panel.append(element('p', 'Answer display was clipped to the local safety limit.', 'notice'));
  }
  if (name === 'flow') panel.append(flowPanel(trace, inspector));
  if (name === 'evidence') {
    panel.append(
      element('h3', 'Evidence', 'panel-title'),
      dataList([['Selected evidence', trace.retrieval?.count], ['Retrieved record IDs', ids.length ? ids.join(', ') : 'None recorded']]),
      element('p', 'Only retrieved record IDs and counts are retained. Evidence snippets are not recorded.', 'json-purpose')
    );
  }
  if (name === 'diagnostics') {
    const modelEvents = events.filter(event => ['interpret_turn', 'select_evidence'].includes(event.name));
    const selectedModelEvent = selectedStage && stageRole(selectedStage).kind === 'llm'
      ? selectedStage
      : null;
    const diagnosticReason = (events || []).map(privacyReason).find(Boolean);
    const diagnosticResult = trace.error || (selectedModelEvent ? eventResult(selectedModelEvent) : 'No diagnostics recorded.');
    const diagnosticDescription = trace.error
      ? errorMessage(trace.error)
      : selectedModelEvent
        ? selectedModelEvent.result === 'success'
          ? 'A model tool result was recorded.'
          : 'A model tool failure was recorded.'
        : 'No diagnostics recorded.';
    const diagnosticDetails = [
      ['Result', diagnosticResult],
      ...(diagnosticReason ? [['Reason', diagnosticReason]] : []),
      ['Description', diagnosticDescription],
      ['Selected stage', selectedStage?.name || 'Trace root'],
    ];
    panel.append(
      element('h3', 'Diagnostics', 'panel-title'),
      dataList(diagnosticDetails),
      ...(modelEvents.length ? modelEvents.map((event, index) => eventRow(event, events.indexOf(event) + index)) : [textBlock('No model event was recorded.')])
    );
  }
  if (name === 'metadata') {
    panel.append(
      element('h3', 'Metadata', 'panel-title'),
      dataList([
        ['Trace ID', trace.trace_id], ['Session ID', trace.session_id], ['Message ID', trace.message_id],
        ['Request kind', trace.request_kind], ['Recorded at', trace.created_at], ['Model provider', trace.model?.provider], ['Model ID', trace.model?.id],
      ]),
      element('p', trace.cost?.note || 'External API cost is not recorded.', 'notice')
    );
  }
  if (name === 'raw') {
    panel.append(
      element('h3', 'Raw', 'panel-title'),
      element('p', 'The safe, bounded trace retained in RAM. Input rejected by privacy controls is never recorded.', 'json-purpose'),
      element('pre', JSON.stringify(trace, null, 2), 'event-json')
    );
  }
  inspector.replaceChildren(panel);
  if (focus) panel.focus();
}

function traceTree(trace, inspector) {
  const events = trace.events || [];
  const tree = element('nav', undefined, 'trace-tree');
  tree.setAttribute('aria-label', 'Trace steps');
  tree.append(element('p', 'Trace', 'tree-label'));
  const root = element('button', undefined, 'tree-row tree-root');
  root.type = 'button';
  root.setAttribute('aria-pressed', String(selectedStage === null));
  root.append(element('i', undefined, `status-dot ${outcomeTone(trace.status)}`), element('span', 'Turn', 'tree-name'), element('span', duration(trace.duration_ms) || '', 'tree-time'));
  root.addEventListener('click', () => { selectedStage = null; renderTree(); selectTab(trace, 'input', inspector, true); });
  const steps = element('ol', undefined, 'tree-steps');
  if (!events.length) steps.append(element('li', 'No steps were retained.', 'json-purpose'));
  const max = Math.max(1, ...events.map(event => Number.isFinite(event.duration_ms) ? event.duration_ms : 0));
  events.forEach((event, index) => {
    const item = element('li');
    const button = element('button', undefined, `tree-row ${flowTone(event)}`);
    button.type = 'button'; button.dataset.stage = String(index);
    const clarification = event.name === 'clarify_decision';
    const target = clarification ? 'flow' : event.name === 'frozen_retrieval' ? 'evidence' : ['interpret_turn', 'select_evidence'].includes(event.name) ? 'diagnostics' : 'input';
    const label = clarification ? (FLOW_LAYERS[event.layer] || 'Clarification') : (FLOW_LABELS[event.name] || event.name || 'Recorded step');
    const ms = Number.isFinite(event.duration_ms) ? event.duration_ms : null;
    button.append(roleBadge(event), element('span', label, 'tree-name'), element('span', duration(ms) || '', 'tree-time'));
    if (ms !== null) {
      const bar = element('span', undefined, 'tree-bar');
      const fill = element('span'); fill.style.width = `${Math.max(4, Math.round((ms / max) * 100))}%`;
      bar.append(fill); button.append(bar);
    }
    button.title = `${event.name || 'step'}: ${eventResult(event)}`;
    button.addEventListener('click', () => { selectedStage = event; renderTree(); selectTab(trace, target, inspector, true); });
    item.append(button);
    steps.append(item);
  });
  const renderTree = () => tree.querySelectorAll('.tree-row').forEach(row => {
    const on = row.classList.contains('tree-root') ? selectedStage === null : events[Number(row.dataset.stage)] === selectedStage;
    row.setAttribute('aria-pressed', String(on));
  });
  tree.append(root, steps);
  renderTree();
  return tree;
}

function renderTrace(trace, preserve = false) {
  const scrollTop = preserve ? detail.scrollTop : 0;
  const back = element('button', 'Back to traces', 'back-button');
  back.type = 'button';
  back.addEventListener('click', () => {
    consoleRoot.classList.remove('show-detail');
    (list.querySelector('button[aria-pressed="true"]') || document.querySelector('.run-list')).focus();
  });
  const header = element('header', undefined, 'trace-header');
  const pills = element('div', undefined, 'header-pills');
  const tokensIn = number(trace.usage?.input_tokens);
  const tokensOut = number(trace.usage?.output_tokens);
  pills.append(pill(sentence(trace.status || 'Unknown'), `status-badge ${outcomeTone(trace.status)}`));
  if (duration(trace.duration_ms)) pills.append(pill(duration(trace.duration_ms), 'mono', 'Server latency'));
  if (tokensIn !== null || tokensOut !== null) pills.append(pill(`${tokensIn ?? '?'} → ${tokensOut ?? '?'} tokens`, 'mono', `Input → output tokens (${trace.usage?.input_token_source || 'source not recorded'})`));
  pills.append(pill(usd(trace.cost?.external_api_usd) || 'Not priced', 'mono', trace.cost?.note || 'External API cost'));
  if (Number.isFinite(trace.retrieval?.count) && trace.retrieval.count > 0) pills.append(pill(`${trace.retrieval.count} evidence`, '', 'Selected evidence records'));
  if (trace.model?.id && (trace.events || []).some(event => stageRole(event).kind === 'llm')) pills.append(pill(trace.model.id, 'model', `Model provider: ${trace.model.provider || 'unknown'}`));
  const title = element('h2', displayQuestion(trace), 'trace-question');
  if (isProjection(trace)) { title.title = 'Safe projection of the message: amounts are masked'; pills.append(pill('Safe projection', '', 'The original message is not stored; amounts are masked')); }
  header.append(title, pills);
  const inspector = element('div', undefined, 'inspector');
  const main = element('div', undefined, 'trace-main');
  main.append(createTabs(trace, inspector), inspector);
  const body = element('div', undefined, 'trace-body');
  body.append(traceTree(trace, inspector), main);
  detail.replaceChildren(back, header, body);
  selectTab(trace, activeTab, inspector);
  if (preserve) detail.scrollTop = scrollTop;
}

async function show(id, preserve = false, openDetail = true) {
  const request = ++detailRequest;
  try {
    const trace = await get(`/api/traces/${encodeURIComponent(id)}`);
    if (request !== detailRequest) return;
    const changed = selected !== id;
    selected = id;
    if (changed) { activeTab = 'input'; selectedStage = null; selectedFlowNode = null; openEvents = new Set(); }
    history.replaceState(null, '', `/inspect?trace=${encodeURIComponent(id)}`);
    renderList(); renderTrace(trace, preserve && !changed);
    if (openDetail) consoleRoot.classList.add('show-detail');
  } catch (error) {
    if (request !== detailRequest) return;
    selected = null; consoleRoot.classList.remove('show-detail'); detail.replaceChildren(element('p', error.message, 'notice')); renderList();
  }
}

async function refresh() {
  try {
    const preserveListView = !consoleRoot.classList.contains('show-detail') && Boolean(detail.querySelector('.trace-header'));
    traces = await get('/api/traces'); renderSummary(); renderList();
    status.textContent = `${traces.length} in memory`;
    if (selected && traces.some(trace => trace.trace_id === selected)) await show(selected, true, !preserveListView);
    else if (selected) { selected = null; consoleRoot.classList.remove('show-detail'); history.replaceState(null, '', '/inspect'); detail.replaceChildren(element('p', 'The selected trace has expired.', 'notice')); }
  } catch (error) { status.textContent = error.message; }
}
document.querySelector('#refresh').addEventListener('click', refresh);
refresh();
