// TEMP DEBUG - hapus setelah verifikasi selesai
// Menerima POST binary sequence (Float32Array 30x258) dari browser dan
// menyimpannya sebagai .bin untuk dibandingkan dengan data training.
const http = require("http");
const fs = require("fs");
const path = require("path");

const OUT_DIR = "C:/Users/agung/AppData/Local/Temp/opencode/live_seqs";
fs.mkdirSync(OUT_DIR, { recursive: true });

const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url.startsWith("/seqs/")) {
    const name = path.basename(req.url.replace(/^\/seqs\//, "")).replace(/\.bin$/, "") + ".bin";
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const buf = Buffer.concat(chunks);
      const nFloat = Math.floor(buf.length / 4);
      if (nFloat !== 30 * 258) {
        res.writeHead(400);
        res.end(`expected ${30 * 258} floats (${30 * 258 * 4} bytes), got ${nFloat}`);
        return;
      }
      const file = path.join(OUT_DIR, name);
      fs.writeFileSync(file, buf);
      const arr = new Float32Array(buf.buffer, buf.byteOffset, nFloat);
      let nz = 0;
      for (let i = 0; i < arr.length; i++) if (arr[i] !== 0) nz++;
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end(`saved ${file.slice(-80)} (nonzero=${nz}/${arr.length})`);
    });
    return;
  }

  if (req.method === "GET") {
    const files = fs.existsSync(OUT_DIR)
      ? fs.readdirSync(OUT_DIR).map((f) => `${f} (${fs.statSync(path.join(OUT_DIR, f)).size}b)`)
      : [];
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("Dumped sequences:\n" + (files.join("\n") || "(none)"));
    return;
  }

  res.writeHead(405);
  res.end("method not allowed");
});

server.listen(18767, () => {
  console.log("seq-dump server listening on http://localhost:18767");
});