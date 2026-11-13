const fs = require('fs');
const js = fs.readFileSync('src/private_llm_analytics/web/static/app.js', 'utf8');

global.window = {
  addEventListener: () => {}
};
global.document = {
  getElementById: (id) => ({
    id,
    disabled: false,
    title: '',
    style: {},
    classList: { add: ()=>{}, remove: ()=>{} },
    textContent: '',
    innerHTML: '',
    appendChild: ()=>{}
  }),
  createElement: (tag) => ({
    tagName: tag,
    style: {},
    click: () => { global.clickOccurred = true; },
    parentNode: { removeChild: ()=>{} },
    remove: ()=>{}
  }),
  body: {
    appendChild: (el) => { global.appendedChild = el; },
    removeChild: () => {}
  },
  addEventListener: ()=>{}
};
global.window.URL = {
  createObjectURL: () => 'blob:http://localhost/test',
  revokeObjectURL: () => {}
};
global.fetch = async (url) => {
  return {
    ok: true,
    status: 200,
    headers: { get: () => 'attachment; filename="test_release.csv"' },
    blob: async () => ({ size: 123 }),
    json: async () => ({
      release_id: 'test-123',
      metric: 'interaction_volume',
      recipient: 'provider_a',
      results: { cell_volumes: { 'provider_a:document': 100 } }
    })
  };
};

eval(js);

console.log('Testing showNotification...');
showNotification('test msg');

console.log('Testing downloadReleaseCsv...');
global.clickOccurred = false;
downloadReleaseCsv('test-123', 'provider_a');
setTimeout(() => {
  console.log('Click occurred in downloadReleaseCsv:', global.clickOccurred);
  console.log('Appended child download attr:', global.appendedChild && global.appendedChild.download);

  console.log('Testing window bindings:');
  console.log('window.downloadReleaseCsv exists:', typeof window.downloadReleaseCsv === 'function');
  console.log('window.downloadReleaseJson exists:', typeof window.downloadReleaseJson === 'function');
  console.log('window.loadAndDisplayRelease exists:', typeof window.loadAndDisplayRelease === 'function');
}, 200);
