'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const state = { expanded: new Set(), selectedPart: null, hoverPart: null, rewardSource: 'verifier' };
// Static placements. Opening, closing, and comparing never move or replace a card.
const nodes = new Map();
const partLabels = {
  targets: 'Targets', filter: 'Filter', ce: 'CE', pairs: 'Pairs', ref: 'Ref LLM',
  pair: 'Pair loss', teacher: 'Teacher LLM', tokens: 'Tokens', logits: 'Logits', kl: 'KL',
  critic: 'Critic LLM', group: 'Group', clip: 'Clip loss', weights: 'Clip weights', actor: 'Actor LLM'
};
const objectives = new Set(['ce', 'pair', 'kl', 'clip', 'weights']);
const llmParts = new Set(['ref', 'teacher', 'critic', 'actor']);
const partLabel = part => part === 'reward' ? (state.rewardSource === 'model' ? 'Reward LLM' : 'Verifier') : partLabels[part];
const isLLM = part => llmParts.has(part) || (part === 'reward' && state.rewardSource === 'model');
const math = (part, tex) => '\\htmlClass{math-part part-' + part + '}{' + tex + '}';
const actorPolicy = math('actor', '\\pi_\\theta');
const logPolicy = math('ce', '\\log ' + actorPolicy + '(y_t\\mid h_t)');
const clipTerm = math('clip', '\\ell_{\\mathrm{clip}}');
const groupTerm = math('group', '\\frac{r_i-\\operatorname{mean}(\\mathbf r)}{\\operatorname{std}(\\mathbf r)+\\delta}');
const klTerm = math('kl', 'D_{\\mathrm{KL}}\\!\\left(p_T(\\cdot\\mid h_t)\\,\\|\\,' + actorPolicy + '(\\cdot\\mid h_t)\\right)');
const tokenRatio = '\\rho_t=\\frac{' + actorPolicy + '(y_t\\mid h_t)}{\\pi_{\\mathrm{old}}(y_t\\mid h_t)}';
const groupRatio = '\\rho_{i,t}=\\frac{' + actorPolicy + '(y_{i,t}\\mid h_{i,t})}{\\pi_{\\mathrm{old}}(y_{i,t}\\mid h_{i,t})}';
const refPlus = math('ref', '\\pi_{\\mathrm{ref}}(y^+\\mid x)');
const refMinus = math('ref', '\\pi_{\\mathrm{ref}}(y^-\\mid x)');
const formulas = {
  clip: '\\ell_{\\mathrm{clip}}(\\rho,A)=\\min\\!\\left(\\rho A,\\operatorname{clip}(\\rho,1-\\epsilon,1+\\epsilon)A\\right)'
};
const methods = [
  {
    id: 'sft', title: 'SFT', full: 'Supervised Fine-Tuning', cell: 'no-reward-external',
    critic: false, parts: ['targets','ce','actor'], connectors: ['→','→'],
    formula: '\\mathcal L_{\\mathrm{SFT}}=-\\mathbb E_{(x,y)\\sim\\mathcal D}\\!\\left[\\frac1{|y|}\\sum_t ' + logPolicy + '\\right]',
    source: ['SFT / RLHF · InstructGPT', 'https://arxiv.org/abs/2203.02155']
  },
  {
    id: 'rft', title: 'RFT', full: 'Rejection Sampling Fine-Tuning', cell: 'reward-external',
    critic: false, parts: ['reward','filter','ce','actor'], connectors: ['→','→','→'],
    formula: '\\mathcal L_{\\mathrm{RFT}}=-\\mathbb E_{(x,y)\\sim\\mathcal D_{\\mathrm{keep}}}\\!\\left[\\frac1{|y|}\\sum_t ' + logPolicy + '\\right]',
    definitions: ['\\mathcal D_{\\mathrm{keep}}=' + math('filter', '\\{(x,y):r(x,y)\\geq\\tau\\}')],
    source: ['RFT / GRPO · DeepSeekMath', 'https://arxiv.org/abs/2402.03300']
  },
  {
    id: 'dpo', title: 'DPO', full: 'Direct Preference Optimization', cell: 'no-reward-external',
    critic: false, parts: ['pairs','ref','pair','actor'], connectors: ['+','→','→'],
    formula: '\\mathcal L_{\\mathrm{DPO}}=-\\mathbb E_{\\mathcal D}\\!\\left[' + math('pair', '\\log\\sigma\\!\\left(\\beta\\left[\\log\\frac{' + actorPolicy + '(y^+\\mid x)}{' + refPlus + '}-\\log\\frac{' + actorPolicy + '(y^-\\mid x)}{' + refMinus + '}\\right]\\right)') + '\\right]',
    source: ['DPO · Rafailov et al.', 'https://arxiv.org/abs/2305.18290']
  },
  {
    id: 'kd-hard-off', title: 'Hard distillation', full: 'Off-Policy Hard Distillation', cell: 'no-reward-external',
    critic: false, parts: ['teacher','tokens','ce','actor'], connectors: ['→','→','→'],
    formula: '\\mathcal L_{\\mathrm{hard}}=-\\mathbb E_{x,\\,y\\sim ' + math('teacher', 'p_T') + '}\\!\\left[\\frac1{|y|}\\sum_t ' + logPolicy + '\\right]',
    source: ['Hard distillation · Sequence-Level Knowledge Distillation', 'https://arxiv.org/abs/1606.07947']
  },
  {
    id: 'kd-off', title: 'Soft distillation', full: 'Off-Policy Soft Distillation', cell: 'no-reward-external',
    critic: false, parts: ['teacher','logits','kl','actor'], connectors: ['→','→','→'],
    formula: '\\mathcal L_{\\mathrm{KD}}=\\mathbb E_{x,\\,y\\sim ' + math('teacher', 'p_T') + '}\\!\\left[\\frac1{|y|}\\sum_t ' + klTerm + '\\right]',
    source: ['Soft distillation · Generalized Knowledge Distillation', 'https://arxiv.org/abs/2306.13649']
  },
  {
    id: 'kd-hard-on', title: 'Hard distillation', full: 'On-Policy Hard Distillation · Teacher Labels', cell: 'no-reward-current',
    critic: false, parts: ['teacher','tokens','ce','actor'], connectors: ['→','→','→'],
    formula: '\\mathcal L_{\\mathrm{hard,on}}=-\\mathbb E_{x,\\,y\\sim\\operatorname{sg}(' + actorPolicy + ')}\\!\\left[\\frac1{|y|}\\sum_t ' + math('ce', '\\log ' + actorPolicy + '(\\tilde y_t\\mid h_t)') + '\\right]',
    definitions: [math('tokens', '\\tilde y_t') + '=\\operatorname*{arg\\,max}_v\\,' + math('teacher', 'p_T(v\\mid h_t)')],
    source: ['On-policy hard labels · DAgger-style imitation', 'https://arxiv.org/abs/1011.0686']
  },
  {
    id: 'kd-on', title: 'Soft distillation', full: 'On-Policy Soft Distillation · Direct KL (GKD)', cell: 'no-reward-current',
    critic: false, parts: ['teacher','logits','kl','actor'], connectors: ['→','→','→'],
    formula: '\\mathcal L_{\\mathrm{GKD}}=\\mathbb E_{x,\\,y\\sim\\operatorname{sg}(' + actorPolicy + ')}\\!\\left[\\frac1{|y|}\\sum_t ' + klTerm + '\\right]',
    source: ['On-policy KL matching · GKD', 'https://arxiv.org/abs/2306.13649']
  },
  {
    id: 'ppo', title: 'PPO', full: 'Proximal Policy Optimization', cell: 'reward-current',
    critic: true, parts: ['ref','reward','critic','clip','actor'], connectors: ['+','+','→','→'],
    formula: '\\mathcal J_{\\mathrm{PPO}}=\\mathbb E_t[' + clipTerm + '(' + math('actor', '\\rho_t') + ',\\hat A_t)]',
    definitions: [tokenRatio, '\\hat A_t=\\operatorname{GAE}(r_t^{\\mathrm{reg}},' + math('critic', 'V_\\psi') + ')'],
    source: ['PPO · Schulman et al.', 'https://arxiv.org/abs/1707.06347']
  },
  {
    id: 'grpo', title: 'GRPO', full: 'Group Relative Policy Optimization', cell: 'reward-current',
    critic: false, parts: ['ref','reward','group','clip','actor'], connectors: ['+','+','→','→'],
    formula: '\\mathcal J_{\\mathrm{GRPO}}=\\mathbb E\\!\\left[\\frac1G\\sum_i\\frac1{|y_i|}\\sum_t\\bigl(' + clipTerm + '(' + math('actor', '\\rho_{i,t}') + ',\\hat A_i)-\\beta K_{i,t}\\bigr)\\right]',
    definitions: [groupRatio, '\\hat A_i=' + groupTerm],
    source: ['GRPO · DeepSeekMath', 'https://arxiv.org/abs/2402.03300']
  },
  {
    id: 'cispo', title: 'CISPO', full: 'Clipped IS-weight Policy Optimization', cell: 'reward-current',
    critic: false, parts: ['reward','group','weights','actor'], connectors: ['+','→','→'],
    formula: '\\mathcal J_{\\mathrm{CISPO}}=\\mathbb E\\!\\left[\\frac1{\\sum_i|y_i|}\\sum_{i,t}\\operatorname{sg}(\\bar\\rho_{i,t})\\,\\hat A_i\\log ' + actorPolicy + '(y_{i,t}\\mid h_{i,t})\\right]',
    definitions: [groupRatio, '\\hat A_i=' + groupTerm, '\\bar\\rho_{i,t}=' + math('weights', '\\operatorname{clip}(\\rho_{i,t},1-\\epsilon_\\ell,1+\\epsilon_h)')],
    source: ['CISPO · MiniMax-M1', 'https://arxiv.org/abs/2506.13585']
  }
];

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function componentHTML(part) {
  const label = partLabel(part);
  const title = part === 'actor' ? 'Trainable policy πθ; the loss updates its weights' : 'Highlight ' + label;
  const classes = 'part-button' + (objectives.has(part) ? ' objective' : '') + (isLLM(part) ? ' llm-component' : '') + (part === 'actor' ? ' actor-component' : '');
  return '<button class="' + classes + '" data-part="' + part + '" data-component-type="' + (isLLM(part) ? 'llm' : 'function') + '" aria-label="Highlight shared component: ' + escapeHTML(label) + (part === 'actor' ? ', trainable policy' : '') + '" aria-pressed="false" title="' + escapeHTML(title) + '">' + escapeHTML(label) + '</button>';
}

function cardHTML(m) {
  return '<article class="method-card" id="card-' + m.id + '" data-id="' + m.id + '" data-critic="' + (m.critic ? 'yes' : 'none') + '">' +
    '<div class="method-header"><button class="method-toggle" data-toggle="' + m.id + '" aria-expanded="false" aria-controls="detail-' + m.id + '">' +
      '<i class="critic-dot ' + (m.critic ? 'has-critic' : 'no-critic') + '" aria-label="' + (m.critic ? 'Critic required' : 'No critic') + '"></i><span class="method-name">' + escapeHTML(m.title) + '</span><span class="chevron" aria-hidden="true">⌄</span></button>' +
      '<div class="pipeline" aria-label="Training signal and actor update">' + m.parts.map((part, i) => (i ? '<span class="part-connector" aria-hidden="true">' + m.connectors[i-1] + '</span>' : '') + componentHTML(part)).join('') + '</div>' +
    '</div><div class="method-detail" id="detail-' + m.id + '" hidden><h3 class="formula-title">' + escapeHTML(m.full) + '</h3><div class="formula" data-method-formula="' + m.id + '"></div>' + (m.definitions || []).map((_, i) => '<div class="formula definition" data-definition-index="' + i + '"></div>').join('') + '</div></article>';
}

function renderFormula(el, tex) {
  if (!el || el.dataset.rendered === 'yes') return;
  try {
    katex.render(tex, el, {
      displayMode: true, throwOnError: true, output: 'htmlAndMathml',
      trust: context => context.command === '\\htmlClass',
      strict: code => code === 'htmlExtension' ? 'ignore' : 'warn'
    });
    el.dataset.rendered = 'yes';
  } catch (error) {
    el.textContent = tex; el.classList.add('formula-error');
    console.error('Formula rendering failed', error);
  }
}

function fitFormulas() {
  $$('.formula').forEach(el => {
    if (!el.offsetWidth) return;
    const mathEl = $('.katex-html', el);
    if (!mathEl) return;
    el.style.fontSize = '';
    const computed = getComputedStyle(el);
    const base = parseFloat(computed.fontSize);
    const available = el.clientWidth - parseFloat(computed.paddingLeft) - parseFloat(computed.paddingRight);
    const width = Math.max(mathEl.scrollWidth, mathEl.getBoundingClientRect().width);
    if (width > available) el.style.fontSize = Math.max(10, Math.floor(base * available / width * .98)) + 'px';
  });
}

function applyHighlights() {
  $$('[data-part]').forEach(el => {
    el.classList.toggle('shared-hover', el.dataset.part === state.hoverPart);
    el.classList.toggle('shared-active', el.dataset.part === state.selectedPart);
    el.setAttribute('aria-pressed', el.dataset.part === state.selectedPart);
  });
  $$('.math-part').forEach(el => {
    el.classList.toggle('shared-hover', !!state.hoverPart && el.classList.contains('part-' + state.hoverPart));
    el.classList.toggle('shared-active', !!state.selectedPart && el.classList.contains('part-' + state.selectedPart));
  });
  methods.forEach(m => {
    const card = nodes.get(m.id).card;
    card.classList.toggle('shared-hover-method', m.parts.includes(state.hoverPart));
    card.classList.toggle('shared-active-method', m.parts.includes(state.selectedPart));
  });
}

function updateGlobalButton() {
  const all = state.expanded.size === methods.length;
  $('#expand-all').textContent = all ? 'Collapse all' : 'Expand all';
  $('#expand-all').setAttribute('aria-pressed', all);
}

function setExpanded(id, expanded) {
  const node = nodes.get(id);
  if (!node) return;
  if (expanded) state.expanded.add(id); else state.expanded.delete(id);
  node.detail.hidden = !expanded;
  node.toggle.setAttribute('aria-expanded', expanded);
  node.card.classList.toggle('expanded', expanded);
  if (expanded) {
    renderFormula(node.formula, node.method.formula);
    $$('.definition', node.card).forEach(el => renderFormula(el, node.method.definitions[Number(el.dataset.definitionIndex)]));
    applyHighlights();
  }
  updateGlobalButton();
  requestAnimationFrame(fitFormulas);
}

// Build each card once. Future actions only update its own hidden/ARIA state.
methods.forEach(m => {
  $('#' + m.cell + ' .card-list').insertAdjacentHTML('beforeend', cardHTML(m));
  const card = $('#card-' + m.id);
  nodes.set(m.id, { card, method:m, detail:$('.method-detail',card), toggle:$('.method-toggle',card), formula:$('.formula',card) });
  $('.method-header', card).addEventListener('click', event => {
    if (event.target.closest('[data-part]')) return;
    setExpanded(m.id, !state.expanded.has(m.id));
  });
});

$('#matrix').addEventListener('click', event => {
  const part = event.target.closest('[data-part]');
  if (!part) return;
  state.selectedPart = state.selectedPart === part.dataset.part ? null : part.dataset.part;
  applyHighlights();
});
$('#matrix').addEventListener('pointerover', event => {
  const part = event.target.closest('[data-part]');
  if (!part || state.hoverPart === part.dataset.part) return;
  state.hoverPart = part.dataset.part; applyHighlights();
});
$('#matrix').addEventListener('pointerout', event => {
  const part = event.target.closest('[data-part]');
  if (!part || part.contains(event.relatedTarget)) return;
  state.hoverPart = null; applyHighlights();
});
$('#matrix').addEventListener('focusin', event => {
  const part = event.target.closest('[data-part]');
  if (!part) return;
  state.hoverPart = part.dataset.part; applyHighlights();
});
$('#matrix').addEventListener('focusout', event => {
  if (!event.target.closest('[data-part]')) return;
  state.hoverPart = null; applyHighlights();
});
$('#expand-all').addEventListener('click', () => {
  const open = state.expanded.size !== methods.length;
  methods.forEach(m => setExpanded(m.id, open));
});
$$('[data-reward-source]').forEach(button => button.addEventListener('click', () => {
  state.rewardSource = button.dataset.rewardSource;
  $$('[data-reward-source]').forEach(el => el.setAttribute('aria-pressed', el.dataset.rewardSource === state.rewardSource));
  $$('[data-part="reward"]').forEach(el => {
    const label = partLabel('reward');
    el.textContent = label;
    el.classList.toggle('llm-component', isLLM('reward'));
    el.dataset.componentType = isLLM('reward') ? 'llm' : 'function';
    el.setAttribute('aria-label', 'Highlight shared component: ' + label);
    el.title = 'Highlight ' + label;
  });
  applyHighlights();
}));
$('#notes-toggle').addEventListener('click', () => {
  const show = $('#reading-notes').hidden;
  $('#reading-notes').hidden = !show;
  $('#notes-toggle').setAttribute('aria-expanded', show);
  if (show) renderFormula($('[data-formula="clip"]'), formulas.clip);
  requestAnimationFrame(fitFormulas);
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  state.selectedPart = null; state.hoverPart = null; applyHighlights();
});
let resizeFrame;
window.addEventListener('resize', () => {
  cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(fitFormulas);
});
const sourceSeen = new Set();
methods.forEach(m => {
  if (sourceSeen.has(m.source[1])) return;
  sourceSeen.add(m.source[1]);
  $('#source-list').insertAdjacentHTML('beforeend','<a class="source-link" href="' + m.source[1] + '" target="_blank" rel="noreferrer">' + escapeHTML(m.source[0]) + ' ↗</a>');
});
if (document.fonts) {
  document.fonts.ready.then(fitFormulas);
  document.fonts.addEventListener('loadingdone', () => requestAnimationFrame(fitFormulas));
}
