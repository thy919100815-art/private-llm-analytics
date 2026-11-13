const fs = require('fs');

const html = fs.readFileSync('src/private_llm_analytics/web/templates/index.html', 'utf8');
const js = fs.readFileSync('src/private_llm_analytics/web/static/app.js', 'utf8');

// Parse IDs from HTML
const idRegex = /id=["']([a-zA-Z0-9_\-]+)["']/g;
const elementIds = new Set();
let match;
while ((match = idRegex.exec(html)) !== null) {
  elementIds.add(match[1]);
}

console.log(`Loaded ${elementIds.size} IDs from index.html`);

// Build mock DOM elements
const elements = {};
function createMockElement(id) {
  return {
    id: id,
    style: {},
    classList: {
      add: () => {},
      remove: () => {},
      toggle: () => {},
      contains: () => false
    },
    textContent: '',
    innerHTML: '',
    value: id === 'input-epsilon' ? '1.0' : id === 'input-delta' ? '1e-6' : '',
    checked: false,
    disabled: false,
    appendChild: (child) => child,
    removeChild: () => {},
    remove: () => {},
    addEventListener: (event, handler) => {
      // Record listener
    },
    setAttribute: () => {},
    getAttribute: () => null,
    querySelectorAll: () => [],
    querySelector: () => null,
    closest: () => null,
    dataset: {}
  };
}

elementIds.forEach(id => {
  elements[id] = createMockElement(id);
});

// Setup global mock environment
const eventListeners = {};
global.window = {
  addEventListener: (event, handler) => {
    if (!eventListeners[event]) eventListeners[event] = [];
    eventListeners[event].push(handler);
  }
};
global.document = {
  addEventListener: (event, handler) => {
    if (!eventListeners[event]) eventListeners[event] = [];
    eventListeners[event].push(handler);
  },
  getElementById: (id) => elements[id] || null,
  querySelectorAll: (selector) => [],
  querySelector: (selector) => null,
  createElement: (tag) => createMockElement('temp_' + tag),
  createTextNode: (text) => ({ text }),
  getElementsByName: (name) => [
    { value: 'interaction_volume', checked: true }
  ],
  body: createMockElement('body')
};

global.fetch = async (url) => {
  if (url.includes('/api/config')) {
    return {
      ok: true,
      json: async () => ({
        providers: ['provider_a', 'provider_b', 'provider_c'],
        tasks: ['document', 'technical', 'analysis'],
        metrics: [{ id: 'interaction_volume', name_zh: '模型交互量' }],
        recipients: ['provider_a', 'internal'],
        recipient_details: [
          { recipient_id: 'provider_a', display_name: 'OpenAI', role: 'external' }
        ],
        candidate_windows: [7, 30, 90],
        demo_mode: true
      })
    };
  }
  return {
    ok: true,
    json: async () => ({})
  };
};

global.autoRenderMath = () => {};

try {
  eval(js);
  console.log('✓ app.js evaluated successfully without syntax or execution errors.');
  
  // Trigger DOMContentLoaded
  if (eventListeners['DOMContentLoaded']) {
    console.log(`Triggering ${eventListeners['DOMContentLoaded'].length} DOMContentLoaded listener(s)...`);
    for (const fn of eventListeners['DOMContentLoaded']) {
      fn();
    }
    console.log('✓ DOMContentLoaded listener(s) executed without errors!');
  }
} catch (err) {
  console.error('❌ ERROR executing app.js:', err);
  process.exit(1);
}
