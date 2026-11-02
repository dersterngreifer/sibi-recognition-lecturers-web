import * as tf from "@tensorflow/tfjs";
import {
  ACTIONS,
  ACTIONS_FULL,
  SEQUENCE_LENGTH,
  KEYPOINTS_CONFIG,
  MODEL_PATH,
  MODEL_PATH_FULL,
} from "@/config/modelConfig";

let model: tf.LayersModel | null = null;
let currentLoadedPath: string | null = null;
let activeActions: readonly string[] = ACTIONS;
let loadPromise: Promise<void> | null = null;

/**
 * AttentionLayer — port dari Keras (model30class / BiLSTM + Attention).
 * Keras: e = tanh(x @ W + b); a = softmax(e, axis=1); out = sum(x * a, axis=1)
 */
class AttentionLayer extends tf.layers.Layer {
  static className = "AttentionLayer";

  private W_?: tf.LayerVariable;
  private b_?: tf.LayerVariable;

  build(inputShape: tf.Shape | tf.Shape[]) {
    const shape: tf.Shape = Array.isArray(inputShape[0])
      ? (inputShape as tf.Shape[])[0]
      : (inputShape as tf.Shape);
    const inputLen = shape[shape.length - 1];
    const seqLen = shape[shape.length - 2];
    this.W_ = this.addWeight("variable_2", [inputLen, 1], "float32", tf.initializers.glorotUniform({ seed: 0 }));
    this.b_ = this.addWeight("variable_3", [seqLen, 1], "float32", tf.initializers.zeros());
    super.build(inputShape);
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const W = this.W_?.read() as tf.Tensor;
    const b = this.b_?.read() as tf.Tensor;

    // e = tanh(x @ W + b)  => [batch, seq, 1]
    const e = tf.tanh(x.matMul(W).add(b));
    // a = softmax over axis=1 (tf.softmax hanya mendukung sumbu terakhir)
    const maxE = e.max(1, true);
    const expE = e.sub(maxE).exp();
    const a = expE.div(expE.sum(1, true));
    // out = sum(x * a, axis=1) => [batch, features]
    return x.mul(a).sum(1);
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    const shape = inputShape as tf.Shape;
    return [shape[0], shape[shape.length - 1]];
  }

  getClassName(): string {
    return AttentionLayer.className;
  }
}

class CustomAttentionLayer extends AttentionLayer {
  static className = "Custom>AttentionLayer";
  getClassName(): string {
    return CustomAttentionLayer.className;
  }
}

// Daftarkan layer custom agar tf.loadLayersModel() bisa membaca "AttentionLayer" & "Custom>AttentionLayer"
if (typeof window !== "undefined" || typeof process !== "undefined") {
  try {
    tf.serialization.registerClass(AttentionLayer);
    tf.serialization.registerClass(CustomAttentionLayer);
  } catch {
    // Ignore duplicate registration
  }
}

export async function loadModel(
  customPath?: string,
  customActions?: readonly string[]
): Promise<boolean> {
  const targetPath = customPath || MODEL_PATH;

  if (model && currentLoadedPath === targetPath) {
    if (customActions) activeActions = customActions;
    return true;
  }

  if (model && currentLoadedPath !== targetPath) {
    model.dispose();
    model = null;
    currentLoadedPath = null;
  }

  if (!loadPromise) {
    loadPromise = (async () => {
      const loaded = await tf.loadLayersModel(targetPath);
      model = loaded;
      currentLoadedPath = targetPath;
      activeActions =
        customActions || (targetPath === MODEL_PATH_FULL ? ACTIONS_FULL : ACTIONS);
    })();
  }

  try {
    await loadPromise;
    return !!model;
  } catch (error) {
    console.error("Gagal memuat model TFJS:", error);
    model = null;
    currentLoadedPath = null;
    return false;
  } finally {
    loadPromise = null;
  }
}

export function isModelLoaded(customPath?: string): boolean {
  if (!model) return false;
  if (customPath && currentLoadedPath !== customPath) return false;
  return true;
}

export function getCurrentModelPath(): string | null {
  return currentLoadedPath;
}

export function disposeModel(): void {
  if (model) {
    model.dispose();
    model = null;
    currentLoadedPath = null;
  }
}

export type PredictionResult = {
  predictions: number[];
  confidence: number;
  gesture: string;
  allProbabilities: number[];
};

export async function predictGesture(
  sequence: Float32Array[],
  customActions?: readonly string[]
): Promise<PredictionResult> {
  const actionsList = customActions || activeActions || ACTIONS;
  const seqLen = SEQUENCE_LENGTH;
  const total = KEYPOINTS_CONFIG.TOTAL_FEATURES;

  if (!model || !seqLen || !total || sequence.length < seqLen) {
    return {
      predictions: [],
      confidence: 0,
      gesture: "Menunggu konfigurasi model...",
      allProbabilities: [],
    };
  }

  const flat = new Float32Array(seqLen * total);
  for (let i = 0; i < seqLen; i++) {
    flat.set(sequence[i].subarray(0, total), i * total);
  }

  const input = tf.tensor(flat, [1, seqLen, total], "float32");
  let result: tf.Tensor | tf.Tensor[] | undefined;

  try {
    result = model.predict(input);
    const output = Array.isArray(result) ? result[0] : result;
    const probs = Array.from(await output.data());

    let bestIndex = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIndex]) bestIndex = i;
    }

    return {
      predictions: [bestIndex],
      confidence: probs[bestIndex],
      gesture: actionsList[bestIndex] ?? "Unknown",
      allProbabilities: probs,
    };
  } catch (error) {
    console.error("Gagal menjalankan prediksi:", error);
    return {
      predictions: [],
      confidence: 0,
      gesture: "Error prediksi",
      allProbabilities: [],
    };
  } finally {
    input.dispose();
    if (result) {
      if (Array.isArray(result)) result.forEach((t) => t.dispose());
      else result.dispose();
    }
  }
}
