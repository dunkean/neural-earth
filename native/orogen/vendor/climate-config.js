// Canonical settings shared with Python validation and the compact form.
import parameters from '../climate-parameters.json' with {type:'json'};
export const climate=Object.fromEntries(Object.entries(parameters).map(([key,spec])=>[
    key.slice(7), globalThis.OROGEN_CLIMATE?.[key.slice(7)] ?? spec.default
]));
