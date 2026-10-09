const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
require(NEURAL_EARTH_ROOT + '/tools/verification/verify_natural.cjs');
