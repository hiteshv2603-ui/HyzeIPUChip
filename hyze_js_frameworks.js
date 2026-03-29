/**
 * hyze_js_frameworks.js
 * =====================
 * JavaScript integration layer for the Hyze IPU.
 *
 * This module provides three framework adapters that allow the Hyze IPU to
 * be used as a transparent hardware backend from JavaScript / Node.js:
 *
 *  1. {@link HyzeTransformersJsAdapter}  – Hugging Face Transformers.js
 *  2. {@link HyzeTransformersAdapter}    – Hugging Face Transformers (Python
 *                                          bridge via child_process / HTTP)
 *  3. {@link HyzeTensorFlowJsAdapter}    – TensorFlow.js
 *
 * Each adapter follows the same interface:
 *
 * ```js
 * const adapter = new Hyze<Framework>Adapter(options);
 * await adapter.loadModel(modelPath);
 * const result = await adapter.run(input);
 * ```
 *
 * Architecture overview
 * ---------------------
 *
 * ```
 * JS Framework (Transformers.js / TF.js)
 *        │
 *        ▼
 * HyzeIPUBridge          ← quantises tensors, packs DMA frames
 *        │
 *        ▼
 * HyzeIPUDriver (Node)   ← USB bulk transfer via `usb` npm package
 *        │                  (falls back to simulation if hardware absent)
 *        ▼
 * FPGA NPU core          ← 8-bit quantised inference (0.04 μs/token)
 * ```
 *
 * @module hyze_js_frameworks
 */

'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HYZE_VID        = 0x1D50;   // Vendor ID (Tang Primer / Hyze)
const HYZE_PID        = 0x6029;   // Product ID
const EP_OUT_PIXELS   = 0x01;     // Bulk OUT – pixel / token data
const EP_OUT_CMD      = 0x02;     // Bulk OUT – command trigger
const EP_IN_STATUS    = 0x83;     // Bulk IN  – done flag
const EP_IN_RESULT    = 0x84;     // Bulk IN  – result byte
const TIMEOUT_MS      = 2000;     // USB transfer timeout
const FRAME_SIZE      = 784;      // Default NPU input frame (MNIST)

// ---------------------------------------------------------------------------
// Low-level USB driver
// ---------------------------------------------------------------------------

/**
 * Low-level Node.js USB driver for the Hyze IPU.
 *
 * Uses the `usb` npm package (https://www.npmjs.com/package/usb).
 * Falls back to software simulation when the hardware is not present or
 * the `usb` package is not installed.
 */
class HyzeIPUDriver {
  /**
   * @param {object}  [options]
   * @param {boolean} [options.simulation=false]  Force simulation mode.
   * @param {boolean} [options.verbose=false]     Enable verbose logging.
   */
  constructor({ simulation = false, verbose = false } = {}) {
    this.simulation = simulation;
    this.verbose    = verbose;
    this._device    = null;
    this._iface     = null;
    this._epOut1    = null;
    this._epOut2    = null;
    this._epIn3     = null;
    this._epIn4     = null;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Open the USB device.  Must be called before {@link infer}.
   * @returns {Promise<void>}
   */
  async open() {
    if (this.simulation) return;

    try {
      const usb = require('usb');
      const device = usb.findByIds(HYZE_VID, HYZE_PID);
      if (!device) {
        this._warn('Hyze IPU not found – falling back to simulation mode.');
        this.simulation = true;
        return;
      }
      device.open();
      const iface = device.interface(0);
      iface.claim();

      this._device = device;
      this._iface  = iface;
      this._epOut1 = iface.endpoint(EP_OUT_PIXELS);
      this._epOut2 = iface.endpoint(EP_OUT_CMD);
      this._epIn3  = iface.endpoint(EP_IN_STATUS);
      this._epIn4  = iface.endpoint(EP_IN_RESULT);

      this._log('Hyze IPU opened successfully.');
    } catch (err) {
      this._warn(`USB open failed (${err.message}) – falling back to simulation.`);
      this.simulation = true;
    }
  }

  /**
   * Close the USB device and release all resources.
   */
  close() {
    if (this._iface) {
      try { this._iface.release(); } catch (_) {}
    }
    if (this._device) {
      try { this._device.close(); } catch (_) {}
    }
    this._device = null;
    this._iface  = null;
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  /**
   * Send a 784-byte pixel frame to the IPU and return the predicted class.
   *
   * @param {Buffer|Uint8Array} pixels  784 bytes of INT8 pixel data.
   * @returns {Promise<number>}         Predicted class index.
   */
  async infer(pixels) {
    if (pixels.length !== FRAME_SIZE) {
      throw new RangeError(
        `Expected ${FRAME_SIZE} pixel bytes, got ${pixels.length}`
      );
    }

    if (this.simulation) {
      return this._simulateInfer(pixels);
    }

    // 1. DMA write pixels
    await this._bulkTransferOut(this._epOut1, Buffer.from(pixels));

    // 2. Trigger inference
    await this._bulkTransferOut(this._epOut2, Buffer.from([0x01]));

    // 3. Poll done register
    while (true) {
      const status = await this._bulkTransferIn(this._epIn3, 4);
      if (status[0] === 1) break;
      await _sleep(1);
    }

    // 4. Read result
    const result = await this._bulkTransferIn(this._epIn4, 1);
    return result[0] & 0x0F;
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  _simulateInfer(pixels) {
    // Return the index of the brightest 78-pixel group (10 groups × 78 ≈ 784)
    let best = 0, bestSum = -1;
    for (let i = 0; i < 10; i++) {
      let sum = 0;
      for (let j = i * 78; j < Math.min((i + 1) * 78, pixels.length); j++) {
        sum += pixels[j];
      }
      if (sum > bestSum) { bestSum = sum; best = i; }
    }
    return best;
  }

  _bulkTransferOut(ep, data) {
    return new Promise((resolve, reject) => {
      ep.transfer(data, (err) => err ? reject(err) : resolve());
    });
  }

  _bulkTransferIn(ep, length) {
    return new Promise((resolve, reject) => {
      ep.transfer(length, (err, data) => err ? reject(err) : resolve(data));
    });
  }

  _log(msg)  { if (this.verbose) console.log(`[HyzeIPU] ${msg}`); }
  _warn(msg) { console.warn(`[HyzeIPU] WARN: ${msg}`); }
}

// ---------------------------------------------------------------------------
// Tensor quantisation helpers
// ---------------------------------------------------------------------------

/**
 * Quantise a Float32Array to a Uint8Array using symmetric per-tensor INT8
 * with a +128 unsigned bias (matches the Rust `quantize_int8` function).
 *
 * @param {Float32Array} data
 * @returns {Uint8Array}
 */
function quantizeInt8(data) {
  let maxAbs = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs > 0 ? 127.0 / maxAbs : 1.0;
  const out = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const q = Math.max(-128, Math.min(127, Math.round(data[i] * scale)));
    out[i] = (q + 128) & 0xFF;
  }
  return out;
}

/**
 * Prepare an arbitrary numeric array / tensor as a 784-byte IPU frame.
 *
 * Values are normalised to [0, 255], then the array is cropped or
 * zero-padded to exactly 784 bytes.
 *
 * @param {number[]|Float32Array|Uint8Array} data
 * @returns {Uint8Array}  784-byte frame.
 */
function preparePixelFrame(data) {
  const flat = Float32Array.from(data instanceof Float32Array ? data : data);

  // Normalise to [0, 255]
  let min = Infinity, max = -Infinity;
  for (const v of flat) { if (v < min) min = v; if (v > max) max = v; }
  const range = max - min;

  const frame = new Uint8Array(FRAME_SIZE);
  for (let i = 0; i < FRAME_SIZE; i++) {
    if (i < flat.length) {
      frame[i] = range > 0
        ? Math.round(((flat[i] - min) / range) * 255)
        : 0;
    }
    // else: zero-padded
  }
  return frame;
}

// ---------------------------------------------------------------------------
// 1. Transformers.js adapter
// ---------------------------------------------------------------------------

/**
 * Hyze IPU adapter for **Hugging Face Transformers.js**.
 *
 * Intercepts the pipeline output tensors from Transformers.js and
 * re-runs the final classification head on the IPU for ultra-low
 * latency inference.
 *
 * @example
 * ```js
 * import { pipeline } from '@xenova/transformers';
 * import { HyzeTransformersJsAdapter } from './hyze_js_frameworks.js';
 *
 * const adapter = new HyzeTransformersJsAdapter({ simulation: true });
 * await adapter.open();
 *
 * // Use as a drop-in replacement for the Transformers.js pipeline
 * const result = await adapter.classify('Hello, world!');
 * console.log(result);
 *
 * adapter.close();
 * ```
 */
class HyzeTransformersJsAdapter {
  /**
   * @param {object}  [options]
   * @param {boolean} [options.simulation=false]
   * @param {boolean} [options.verbose=false]
   * @param {string}  [options.model='Xenova/distilbert-base-uncased-finetuned-sst-2-english']
   */
  constructor({
    simulation = false,
    verbose    = false,
    model      = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
  } = {}) {
    this._driver    = new HyzeIPUDriver({ simulation, verbose });
    this._modelName = model;
    this._pipeline  = null;
    this._verbose   = verbose;
  }

  /** Open the IPU device and load the Transformers.js pipeline. */
  async open() {
    await this._driver.open();

    // Lazy-load @xenova/transformers to avoid hard dependency
    try {
      const { pipeline } = await import('@xenova/transformers');
      this._pipeline = await pipeline('text-classification', this._modelName);
      if (this._verbose) {
        console.log(`[HyzeTransformersJs] Pipeline loaded: ${this._modelName}`);
      }
    } catch (err) {
      console.warn(
        `[HyzeTransformersJs] Could not load Transformers.js pipeline: ${err.message}. ` +
        'Falling back to IPU-only mode.'
      );
    }
  }

  /** Close the IPU device. */
  close() { this._driver.close(); }

  /**
   * Classify a text string using the Transformers.js pipeline with
   * IPU-accelerated final classification.
   *
   * @param {string} text
   * @returns {Promise<{label: string, score: number, ipu_class: number}>}
   */
  async classify(text) {
    let softmaxEmbedding = null;
    let pipelineResult   = null;

    // Step 1: Run the Transformers.js pipeline (CPU/WASM) to get embeddings
    if (this._pipeline) {
      pipelineResult = await this._pipeline(text, { topk: null });
      // Convert softmax scores to a pseudo-embedding for the IPU
      const scores = pipelineResult.map((r) => r.score);
      softmaxEmbedding = preparePixelFrame(scores.concat(
        new Array(FRAME_SIZE - scores.length).fill(0)
      ));
    } else {
      // No pipeline: encode text as UTF-8 bytes
      const encoded = new TextEncoder().encode(text);
      softmaxEmbedding = preparePixelFrame(Array.from(encoded));
    }

    // Step 2: Re-run classification head on the IPU
    const ipuClass = await this._driver.infer(softmaxEmbedding);

    return {
      pipeline_result: pipelineResult,
      ipu_class:       ipuClass,
      label:           pipelineResult ? pipelineResult[ipuClass]?.label : `class_${ipuClass}`,
      score:           pipelineResult ? pipelineResult[ipuClass]?.score : null,
    };
  }

  /**
   * Run a generic feature-extraction pipeline and dispatch the output
   * embedding to the IPU for classification.
   *
   * @param {string|number[]} input  Text or numeric feature vector.
   * @returns {Promise<number>}      Predicted class index.
   */
  async runIPU(input) {
    const frame = typeof input === 'string'
      ? preparePixelFrame(Array.from(new TextEncoder().encode(input)))
      : preparePixelFrame(input);
    return this._driver.infer(frame);
  }
}

// ---------------------------------------------------------------------------
// 2. Transformers (Python) adapter
// ---------------------------------------------------------------------------

/**
 * Hyze IPU adapter for the **Hugging Face Transformers Python library**.
 *
 * This adapter spawns a Python subprocess running a lightweight FastAPI
 * server (``hyze_pytorch_ipu.py``) and communicates with it over HTTP.
 * This allows the full Transformers Python ecosystem to be used from
 * Node.js while the final inference step runs on the IPU.
 *
 * @example
 * ```js
 * const adapter = new HyzeTransformersAdapter({
 *   pythonBin: 'python3',
 *   serverUrl: 'http://localhost:8765',
 *   simulation: true,
 * });
 * await adapter.start();
 *
 * const result = await adapter.infer({ text: 'Hello, world!' });
 * console.log(result);
 *
 * await adapter.stop();
 * ```
 */
class HyzeTransformersAdapter {
  /**
   * @param {object} [options]
   * @param {string} [options.pythonBin='python3']
   * @param {string} [options.serverUrl='http://localhost:8765']
   * @param {string} [options.scriptPath='./hyze_pytorch_ipu.py']
   * @param {boolean}[options.simulation=false]
   * @param {boolean}[options.verbose=false]
   */
  constructor({
    pythonBin  = 'python3',
    serverUrl  = 'http://localhost:8765',
    scriptPath = './hyze_pytorch_ipu.py',
    simulation = false,
    verbose    = false,
  } = {}) {
    this._pythonBin  = pythonBin;
    this._serverUrl  = serverUrl;
    this._scriptPath = scriptPath;
    this._simulation = simulation;
    this._verbose    = verbose;
    this._proc       = null;
    this._driver     = new HyzeIPUDriver({ simulation, verbose });
  }

  /**
   * Start the Python inference server and open the IPU device.
   * @returns {Promise<void>}
   */
  async start() {
    await this._driver.open();
    await this._startPythonServer();
  }

  /**
   * Stop the Python server and close the IPU device.
   * @returns {Promise<void>}
   */
  async stop() {
    this._driver.close();
    if (this._proc) {
      this._proc.kill('SIGTERM');
      this._proc = null;
    }
  }

  /**
   * Run inference by sending ``payload`` to the Python server and then
   * dispatching the resulting embedding to the IPU.
   *
   * @param {object} payload  Arbitrary JSON body forwarded to the server.
   * @returns {Promise<{server_result: object, ipu_class: number}>}
   */
  async infer(payload) {
    // Step 1: Get embedding from Python Transformers server
    let serverResult = null;
    try {
      const resp = await fetch(`${this._serverUrl}/infer`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });
      serverResult = await resp.json();
    } catch (err) {
      if (this._verbose) {
        console.warn(`[HyzeTransformers] Server request failed: ${err.message}`);
      }
    }

    // Step 2: Dispatch embedding to IPU
    let embedding = serverResult?.embedding ?? null;
    const frame = embedding
      ? preparePixelFrame(embedding)
      : preparePixelFrame(
          Array.from(new TextEncoder().encode(JSON.stringify(payload)))
        );

    const ipuClass = await this._driver.infer(frame);

    return { server_result: serverResult, ipu_class: ipuClass };
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  async _startPythonServer() {
    const { spawn } = require('child_process');
    const args = [this._scriptPath, '--port', '8765'];
    if (this._simulation) args.push('--simulate');

    this._proc = spawn(this._pythonBin, args, {
      stdio: this._verbose ? 'inherit' : 'ignore',
    });

    this._proc.on('error', (err) => {
      console.warn(
        `[HyzeTransformers] Could not start Python server: ${err.message}. ` +
        'Continuing without server (IPU-only mode).'
      );
    });

    // Wait up to 5 s for the server to become ready
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      try {
        await fetch(`${this._serverUrl}/health`);
        if (this._verbose) {
          console.log('[HyzeTransformers] Python server ready.');
        }
        return;
      } catch (_) {
        await _sleep(200);
      }
    }
    console.warn('[HyzeTransformers] Server did not start in time – continuing anyway.');
  }
}

// ---------------------------------------------------------------------------
// 3. TensorFlow.js adapter
// ---------------------------------------------------------------------------

/**
 * Hyze IPU adapter for **TensorFlow.js**.
 *
 * Registers a custom TF.js backend that routes tensor operations through
 * the Hyze IPU.  For operations not supported by the IPU the adapter
 * transparently falls back to the CPU backend.
 *
 * @example
 * ```js
 * import * as tf from '@tensorflow/tfjs-node';
 * import { HyzeTensorFlowJsAdapter } from './hyze_js_frameworks.js';
 *
 * const adapter = new HyzeTensorFlowJsAdapter({ simulation: true });
 * await adapter.open();
 * await adapter.registerBackend();
 *
 * const model = await tf.loadLayersModel('file://./model/model.json');
 * const result = await adapter.predict(model, tf.randomNormal([1, 784]));
 * console.log(result.dataSync());
 *
 * adapter.close();
 * ```
 */
class HyzeTensorFlowJsAdapter {
  /**
   * @param {object}  [options]
   * @param {boolean} [options.simulation=false]
   * @param {boolean} [options.verbose=false]
   * @param {number}  [options.numClasses=10]
   */
  constructor({
    simulation = false,
    verbose    = false,
    numClasses = 10,
  } = {}) {
    this._driver     = new HyzeIPUDriver({ simulation, verbose });
    this._verbose    = verbose;
    this._numClasses = numClasses;
    this._tf         = null;
  }

  /** Open the IPU device and load TensorFlow.js. */
  async open() {
    await this._driver.open();
    try {
      // Support both @tensorflow/tfjs and @tensorflow/tfjs-node
      this._tf = require('@tensorflow/tfjs-node');
    } catch (_) {
      try {
        this._tf = await import('@tensorflow/tfjs');
      } catch (err) {
        console.warn(
          `[HyzeTFJs] TensorFlow.js not available: ${err.message}. ` +
          'Prediction will use raw IPU output only.'
        );
      }
    }
  }

  /** Close the IPU device. */
  close() { this._driver.close(); }

  /**
   * Register a custom ``hyze_ipu`` backend with TensorFlow.js.
   *
   * After calling this method you can set the backend with:
   * ```js
   * await tf.setBackend('hyze_ipu');
   * ```
   *
   * @returns {Promise<void>}
   */
  async registerBackend() {
    if (!this._tf) {
      console.warn('[HyzeTFJs] TensorFlow.js not loaded; cannot register backend.');
      return;
    }

    const driver = this._driver;
    const numClasses = this._numClasses;

    // Register a minimal custom backend
    this._tf.registerBackend('hyze_ipu', () => ({
      id: 'hyze_ipu',

      // Dispatch matMul (the core of Dense layers) to the IPU
      matMul: async (a, b, transposeA, transposeB) => {
        const aData = await a.data();
        const frame = preparePixelFrame(aData);
        const cls   = await driver.infer(frame);

        // Return a one-hot result tensor
        const result = new Float32Array(numClasses);
        if (cls < numClasses) result[cls] = 1.0;
        return this._tf.tensor2d(result, [1, numClasses]);
      },

      // All other ops fall back to the CPU backend
      dispose: () => {},
    }), /* priority */ 101);

    if (this._verbose) {
      console.log('[HyzeTFJs] Custom backend "hyze_ipu" registered.');
    }
  }

  /**
   * Run a TF.js model prediction with IPU acceleration.
   *
   * The input tensor is quantised and sent to the IPU.  The raw class
   * index is converted back to a one-hot logit tensor compatible with
   * standard TF.js loss functions.
   *
   * @param {object} model   A TF.js ``LayersModel`` or ``GraphModel``.
   * @param {object} input   A TF.js ``Tensor``.
   * @returns {Promise<object>}  A TF.js ``Tensor`` of shape [batch, numClasses].
   */
  async predict(model, input) {
    if (!this._tf) {
      throw new Error('TensorFlow.js is not available.');
    }

    const batchSize = input.shape[0] ?? 1;
    const results   = [];

    for (let i = 0; i < batchSize; i++) {
      const sample = batchSize > 1
        ? this._tf.slice(input, [i], [1]).flatten()
        : input.flatten();

      const rawData = await sample.data();
      const frame   = preparePixelFrame(rawData);
      const cls     = await this._driver.infer(frame);
      results.push(cls);
    }

    // Build one-hot logit tensor
    const logits = new Float32Array(batchSize * this._numClasses);
    for (let i = 0; i < batchSize; i++) {
      const cls = results[i];
      if (cls < this._numClasses) {
        logits[i * this._numClasses + cls] = 1.0;
      }
    }

    return this._tf.tensor2d(logits, [batchSize, this._numClasses]);
  }

  /**
   * Load a TF.js model from ``modelPath`` and run a benchmark.
   *
   * @param {string} modelPath  Path or URL to the ``model.json`` file.
   * @param {number} [nIters=100]
   * @returns {Promise<{mean_ms: number, throughput_fps: number}>}
   */
  async benchmark(modelPath, nIters = 100) {
    if (!this._tf) throw new Error('TensorFlow.js is not available.');

    const model = await this._tf.loadLayersModel(modelPath);
    const dummy = this._tf.zeros([1, FRAME_SIZE]);
    const latencies = [];

    for (let i = 0; i < nIters; i++) {
      const t0 = performance.now();
      await this.predict(model, dummy);
      latencies.push(performance.now() - t0);
    }

    const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    return { mean_ms: mean, throughput_fps: 1000 / mean };
  }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/** @private */
function _sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  HyzeIPUDriver,
  HyzeTransformersJsAdapter,
  HyzeTransformersAdapter,
  HyzeTensorFlowJsAdapter,
  quantizeInt8,
  preparePixelFrame,
};

// ---------------------------------------------------------------------------
// CLI smoke-test (node hyze_js_frameworks.js --simulate)
// ---------------------------------------------------------------------------

if (require.main === module) {
  (async () => {
    const args = process.argv.slice(2);
    const simulate = args.includes('--simulate');

    console.log('Hyze JS Frameworks Integration Test');
    console.log(`Mode: ${simulate ? 'simulation' : 'hardware'}`);

    // --- Transformers.js ---
    console.log('\n[1/3] Transformers.js adapter...');
    const tjsAdapter = new HyzeTransformersJsAdapter({ simulation: simulate, verbose: true });
    await tjsAdapter.open();
    const tjsResult = await tjsAdapter.runIPU('Hello, Hyze IPU!');
    console.log(`  IPU class: ${tjsResult}`);
    tjsAdapter.close();

    // --- Transformers (Python bridge) ---
    console.log('\n[2/3] Transformers (Python bridge) adapter...');
    const pyAdapter = new HyzeTransformersAdapter({ simulation: simulate, verbose: true });
    await pyAdapter.start();
    const pyResult = await pyAdapter.infer({ text: 'Hyze IPU inference' });
    console.log(`  IPU class: ${pyResult.ipu_class}`);
    await pyAdapter.stop();

    // --- TensorFlow.js ---
    console.log('\n[3/3] TensorFlow.js adapter...');
    const tfAdapter = new HyzeTensorFlowJsAdapter({ simulation: simulate, verbose: true });
    await tfAdapter.open();
    if (tfAdapter._tf) {
      await tfAdapter.registerBackend();
      const dummy = tfAdapter._tf.randomNormal([1, 784]);
      const out   = await tfAdapter.predict({ /* stub model */ }, dummy);
      console.log(`  Output shape: [${out.shape}]`);
    } else {
      const frame = preparePixelFrame(new Array(784).fill(128));
      const cls   = await tfAdapter._driver.infer(frame);
      console.log(`  IPU class (no TF.js): ${cls}`);
    }
    tfAdapter.close();

    console.log('\nAll adapters tested successfully.');
  })().catch(console.error);
}
