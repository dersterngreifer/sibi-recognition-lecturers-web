// TEMP DEBUG - hapus setelah verifikasi selesai
// Bandingkan sequence live (hasil dump browser) vs sequence training (.bin
// yang sudah dipakai test_tfjs.js) → cari perbedaan distribusi per-channel.
const fs = require("fs");
const path = require("path");

const LIVE_DIR = "C:/Users/agung/AppData/Local/Temp/opencode/live_seqs";
const TRAIN_DIR = "C:/Users/agung/AppData/Local/Temp/opencode/seqs";

const POSE_DIM = 132; // 33 * 4
const LH_DIM = 63;
const RH_DIM = 63;
const FEATURES = 258;
const FRAMES = 30;

function loadSeq(file) {
  const buf = fs.readFileSync(file);
  const arr = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 4));
  return arr;
}

function stats(arr, label) {
  let sum = 0, sum2 = 0, mn = Infinity, mx = -Infinity, nz = 0;
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    sum += v;
    sum2 += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    if (v !== 0) nz++;
  }
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
  return { label, n, mean, std, mn, mx, nz, pctNz: ((nz / n) * 100).toFixed(1) };
}

function slice(arr, start, end) {
  return Array.from(arr.slice(start, end));
}

function compareSlices(name, trainFile, liveFile, trainArr, liveArr) {
  const blocks = [
    ["pose_ALL(xyz+vis)", 0, 132],
    ["pose x,y,z        ", 0, 99],
    ["pose visibility   ", 99, 132],
    ["left_hand x,y,z   ", 132, 195],
    ["right_hand x,y,z  ", 195, 258],
  ];
  console.log(`\n=== ${name}: training(${path.basename(trainFile)}) vs live(${path.basename(liveFile)}) ===`);
  for (const [lbl, s, e] of blocks) {
    const t = stats(slice(trainArr, s, e), lbl);
    const l = stats(slice(liveArr, s, e), lbl);
    const meanDelta = (l.mean - t.mean).toFixed(4);
    const stdDelta = (l.std - t.std).toFixed(4);
    console.log(
      `${lbl} | train mean=${t.mean.toFixed(4)} std=${t.std.toFixed(4)} min=${t.mn.toFixed(3)} max=${t.mx.toFixed(3)} nz=${t.pctNz}%` +
      ` | live mean=${l.mean.toFixed(4)} std=${l.std.toFixed(4)} min=${l.mn.toFixed(3)} max=${l.mx.toFixed(3)} nz=${l.pctNz}%` +
      ` | Δmean=${meanDelta} Δstd=${stdDelta}`,
    );
  }
}

const liveFiles = fs.existsSync(LIVE_DIR)
  ? fs.readdirSync(LIVE_DIR).filter((f) => f.endsWith(".bin")).sort()
  : [];
const trainFiles = fs.existsSync(TRAIN_DIR)
  ? fs.readdirSync(TRAIN_DIR).filter((f) => f.endsWith(".bin")).sort()
  : [];

if (liveFiles.length === 0) {
  console.log("Belum ada sequence live. Jalankan app, peragakan gesture, lalu jalankan ulang script ini.");
  console.log("Pastikan logs/seq_dump_server.js sedang berjalan.");
  process.exit(0);
}

console.log(`Live sequences : ${liveFiles.join(", ")}`);
console.log(`Training refs  : ${trainFiles.join(", ")}`);

// Match live word to a training ref of the same word when possible.
const liveByWord = {};
for (const f of liveFiles) {
  const w = f.split("_")[0]; // Hallo_123_rawN.bin -> Hallo
  (liveByWord[w] ||= []).push(f);
}

let fileIdx = 0;
for (const [word, files] of Object.entries(liveByWord)) {
  const ref = trainFiles.find((t) => t.startsWith(word + "_"));
  for (const lf of files) {
    const liveArr = loadSeq(path.join(LIVE_DIR, lf));
    const lbl = `${word}#${fileIdx++}`;
    if (ref) {
      const trainArr = loadSeq(path.join(TRAIN_DIR, ref));
      compareSlices(lbl, ref, lf, trainArr, liveArr);
    } else {
      const t = stats(liveArr, lbl);
      console.log(`\n=== ${lbl}: live(${lf}) [no matching training ref] ===`);
      console.log(`all: mean=${t.mean.toFixed(4)} std=${t.std.toFixed(4)} min=${t.mn.toFixed(3)} max=${t.mx.toFixed(3)} nz=${t.pctNz}%`);
    }
  }
}