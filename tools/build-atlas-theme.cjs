// Keep the UI asset self-contained: the server already serves terrain_toolbar.js.
const fs = require('node:fs');
const path = require('node:path');
const web = path.resolve(__dirname, '../web');
const image = fs.readFileSync(path.join(web, 'assets/atlas-icons.webp')).toString('base64');
const css = fs.readFileSync(path.join(web, 'assets/atlas-theme.css'), 'utf8')
  .replace('--atlas-image:none', `--atlas-image:url("data:image/webp;base64,${image}")`);
const file = path.join(web, 'terrain_toolbar.js');
const source = fs.readFileSync(file, 'utf8').replace(/^const ATLAS_CSS=.*;\r?\n/, '');
fs.writeFileSync(file, `const ATLAS_CSS=${JSON.stringify(css)};\n${source}`);
