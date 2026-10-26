import {mountIllustration, captureViewport} from '/illustration/index.mjs';

mountIllustration({
  host: document.getElementById('controls'),
  capture: () => window.captureNeuralIllustration(captureViewport),
});
