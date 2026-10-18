// node web/parity_test.js  — checks the JS port against Python outputs in parity_fixture.json
const fs = require("fs");
const path = require("path");
const D = require("./detector.js");

const model = D.makeModel(JSON.parse(fs.readFileSync(path.join(__dirname, "model.json"), "utf8")));
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "parity_fixture.json"), "utf8"));

let scrubDiff = 0, ruleDiff = 0, maxDp = 0, labelFlips = 0;
const examples = [];
for (const f of fixture) {
  const clean = D.scrub(f.raw);
  if (clean !== f.clean) { scrubDiff++; if (examples.length < 5) examples.push({ py: f.clean, js: clean }); }
  if (Math.abs(D.checkRules(f.clean).score - f.rule_score) > 1e-9) ruleDiff++;
  const p = model.score(f.clean).p;
  maxDp = Math.max(maxDp, Math.abs(p - f.p));
  if ((p >= 0.5) !== (f.p >= 0.5)) labelFlips++;
}
console.log(`messages: ${fixture.length}`);
console.log(`scrub mismatches: ${scrubDiff}`);
console.log(`rule-score mismatches: ${ruleDiff}`);
console.log(`max |p_js - p_py|: ${maxDp.toExponential(2)}  label flips at 0.5: ${labelFlips}`);
for (const e of examples) console.log("  py:", e.py, "\n  js:", e.js);
process.exit(scrubDiff || ruleDiff || labelFlips || maxDp > 1e-3 ? 1 : 0);
