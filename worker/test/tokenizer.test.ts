/**
 * Parity test: our hand-written WordPieceTokenizer must reproduce the exact
 * `input_ids` Hugging Face's tokenizer produces, otherwise the model receives
 * the wrong tokens. Expected ids are generated from the real tokenizer in
 * `expected_tokens.json`. Run with: `node --experimental-strip-types tokenizer.test.ts`.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { WordPieceTokenizer } from "../src/tokenizer.ts";

const here = dirname(fileURLToPath(import.meta.url));
const vocab = readFileSync(
  join(here, "..", "assets", "models", "distilbert-sst2", "vocab.txt"),
  "utf-8",
);
const expected = JSON.parse(
  readFileSync(join(here, "expected_tokens.json"), "utf-8"),
) as Array<{ text: string; input_ids: number[] }>;

const tokenizer = new WordPieceTokenizer(vocab);

let failures = 0;
for (const { text, input_ids } of expected) {
  const got = tokenizer.encode(text).inputIds.map((b) => Number(b));
  const ok = got.length === input_ids.length && got.every((v, i) => v === input_ids[i]);
  console.log(`${ok ? "PASS" : "FAIL"}  "${text.slice(0, 44)}"`);
  if (!ok) {
    failures++;
    console.log(`   expected: ${input_ids.join(", ")}`);
    console.log(`   got:      ${got.join(", ")}`);
  }
}

console.log(`\n${expected.length - failures}/${expected.length} cases match HF tokenizer.`);
process.exit(failures === 0 ? 0 : 1);
