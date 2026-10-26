// Separate real hydrological complement. Same freeze/plan/run contract; no
// capture is launched by default. Existing 900-frame land corpus is untouched.
process.env.REVIEW_REFERENCE_BASE='output/render-v2-hydrology-reference/captures-reference';
require('./review_render_v2_reference.cjs');
