const tf = require("@tensorflow/tfjs");
const fs = require("fs");
tf.setBackend("cpu");

class AttentionLayer extends tf.layers.Layer {
  static className = "Custom>AttentionLayer";
  constructor(config) { super(config || {}); }
  build(inputShape) {
    const shape = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    const inputLen = shape[shape.length - 1];
    const seqLen = shape[shape.length - 2];
    this.W_ = this.addWeight("variable", [inputLen, 1], "float32", tf.initializers.glorotUniform({ seed: 0 }));
    this.b_ = this.addWeight("variable_1", [seqLen, 1], "float32", tf.initializers.zeros());
    super.build(inputShape);
  }
  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const W = this.W_.read();
    const b = this.b_.read();
    const e = tf.tanh(x.matMul(W).add(b));
    const maxE = e.max(1, true);
    const expE = e.sub(maxE).exp();
    const a = expE.div(expE.sum(1, true));
    return x.mul(a).sum(1);
  }
  computeOutputShape(inputShape) {
    const shape = inputShape;
    return [shape[0], shape[shape.length - 1]];
  }
  getClassName() { return AttentionLayer.className; }
  static getClassName() { return AttentionLayer.className; }
}
tf.serialization.registerClass(AttentionLayer);

const ACTIONS = ['Adik','Agustus','Apa','April','Ayah','Baik','Berapa','Bertemu','Cantik','Dari','Desember','Dia','Dimana','Februari','Ganteng','Gemuk','Hallo','Hobi','Ibu','Januari','Juli','Jumat','Juni','Jurusan','Kabar'];

async function main() {
  const http = require("http");
  const url = require("url");
  const MODELDIR = "D:/Projects/SIBI Recognition Lecturers/Frontend/public/models/tfjs_model25class";
  const mime = { ".json": "application/json", ".bin": "application/octet-stream" };
  const server = http.createServer((req, res) => {
    const p = url.parse(req.url).pathname;
    const f = require("path").join(MODELDIR, p.replace(/^\//, ""));
    try {
      const data = fs.readFileSync(f);
      res.writeHead(200, { "Content-Type": mime[require("path").extname(f)] || "application/octet-stream" });
      res.end(data);
    } catch (e) { res.writeHead(404); res.end("not found"); }
  });
  await new Promise(r => server.listen(18765, r));

  const model = await tf.loadLayersModel("http://localhost:18765/model.json");
  console.log("model loaded, input:", model.inputs[0].shape);

  const binDir = "C:/Users/agung/AppData/Local/Temp/opencode/seqs";
  const cases = ["Hallo_0","Apa_0","Ayah_0","Gemuk_0","Jumat_0","Cantik_0","Adik_0","Berapa_0","Kabar_0","Januari_0"];
  for (const c of cases) {
    const buf = fs.readFileSync(require("path").join(binDir, c + ".bin"));
    const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const input = tf.tensor(arr, [1, 30, 258], "float32");
    const out = model.predict(input);
    const probs = Array.from(await out.data());
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    console.log(`${c.padEnd(10)} -> tfjs pred: ${ACTIONS[best].padEnd(10)} p=${probs[best].toFixed(4)}  top5=${top5(probs)}`);
    input.dispose(); out.dispose();
  }

  const z = tf.zeros([1,30,258],"float32");
  const zo = model.predict(z);
  const zp = Array.from(await zo.data());
  let b=0; for(let i=1;i<zp.length;i++) if(zp[i]>zp[b]) b=i;
  console.log("zeros      -> tfjs pred:", ACTIONS[b], "p=", zp[b].toFixed(4));
  z.dispose(); zo.dispose();
  server.close();

  function top5(p){ const idx = p.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,5); return idx.map(([v,i])=>`${ACTIONS[i]}=${v.toFixed(3)}`).join(" "); }
}

main().catch(e => { console.error(e); process.exit(1); });