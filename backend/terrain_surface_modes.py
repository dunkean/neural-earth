"""Display-only Render v2 modes, separate from source-only atlas diagnostics."""

RENDER_V2_MODES = ('render-v2', 'render-v2-substrate', 'render-v2-moisture',
                   'render-v2-vegetation', 'render-v2-snow')

RENDER_V2_LEGENDS = {
    'render-v2': 'Render v2 · seasonal satellite landscape',
    'render-v2-substrate': 'Render v2 substrate · locally refined parent rock and soil',
    'render-v2-moisture': 'Render v2 moisture · seasonal local habitat proxy',
    'render-v2-vegetation': 'Render v2 vegetation · local natural cover',
    'render-v2-snow': 'Render v2 snow and ice · seasonal coverage',
}
