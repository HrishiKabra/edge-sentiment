/**
 * A minimal, faithful WordPiece tokenizer for `distilbert-base-uncased`.
 *
 * This mirrors Hugging Face's `BertTokenizer` (uncased) closely enough to
 * reproduce the exact `input_ids` the model was trained on for English text:
 *   1. BasicTokenizer — clean text, lowercase, strip accents, split on
 *      whitespace and punctuation.
 *   2. WordpieceTokenizer — greedy longest-match-first subword segmentation with
 *      "##" continuation markers and an [UNK] fallback.
 *
 * We implement it by hand (rather than bundling @xenova/transformers) to keep the
 * Worker bundle small and dependency-light. The vocabulary is loaded once from
 * the static-assets binding and cached.
 */

export interface Encoding {
  inputIds: bigint[];
  attentionMask: bigint[];
}

const MAX_LEN = 128;
const MAX_CHARS_PER_WORD = 100;

export class WordPieceTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly unkId: number;
  private readonly clsId: number;
  private readonly sepId: number;

  constructor(vocabText: string) {
    this.vocab = new Map();
    const lines = vocabText.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i]?.replace(/\r$/, "") ?? "";
      // Trailing blank line at EOF must not become a token.
      if (token === "" && i === lines.length - 1) continue;
      this.vocab.set(token, i);
    }
    this.unkId = this.vocab.get("[UNK]") ?? 100;
    this.clsId = this.vocab.get("[CLS]") ?? 101;
    this.sepId = this.vocab.get("[SEP]") ?? 102;
  }

  /** Tokenize text into model input ids + attention mask, with truncation. */
  encode(text: string, maxLength: number = MAX_LEN): Encoding {
    const basicTokens = this.basicTokenize(text);
    const wordpieceIds: number[] = [];
    for (const token of basicTokens) {
      for (const id of this.wordpiece(token)) {
        wordpieceIds.push(id);
      }
    }

    // Reserve two slots for [CLS] and [SEP].
    const truncated = wordpieceIds.slice(0, Math.max(0, maxLength - 2));
    const ids = [this.clsId, ...truncated, this.sepId];

    return {
      inputIds: ids.map((id) => BigInt(id)),
      attentionMask: ids.map(() => 1n),
    };
  }

  // --- BasicTokenizer ------------------------------------------------------ //

  private basicTokenize(text: string): string[] {
    const cleaned = this.cleanText(text);
    const tokens: string[] = [];
    for (const rawToken of cleaned.trim().split(/\s+/)) {
      if (rawToken === "") continue;
      const lowered = this.stripAccents(rawToken.toLowerCase());
      tokens.push(...this.splitOnPunctuation(lowered));
    }
    return tokens;
  }

  /** Drop control characters and normalize all whitespace to single spaces. */
  private cleanText(text: string): string {
    let out = "";
    for (const char of text) {
      const cp = char.codePointAt(0) ?? 0;
      if (cp === 0 || cp === 0xfffd || this.isControl(char)) continue;
      out += this.isWhitespace(char) ? " " : char;
    }
    return out;
  }

  private stripAccents(text: string): string {
    return text.normalize("NFD").replace(/\p{Mn}/gu, "");
  }

  private splitOnPunctuation(text: string): string[] {
    const result: string[] = [];
    let current = "";
    for (const char of text) {
      if (this.isPunctuation(char)) {
        if (current) {
          result.push(current);
          current = "";
        }
        result.push(char);
      } else {
        current += char;
      }
    }
    if (current) result.push(current);
    return result;
  }

  // --- WordpieceTokenizer -------------------------------------------------- //

  private wordpiece(token: string): number[] {
    const chars = Array.from(token);
    if (chars.length > MAX_CHARS_PER_WORD) return [this.unkId];

    const subTokenIds: number[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let currentId: number | null = null;
      while (start < end) {
        let substr = chars.slice(start, end).join("");
        if (start > 0) substr = "##" + substr;
        const id = this.vocab.get(substr);
        if (id !== undefined) {
          currentId = id;
          break;
        }
        end -= 1;
      }
      if (currentId === null) {
        // Any unmatchable piece makes the whole word [UNK].
        return [this.unkId];
      }
      subTokenIds.push(currentId);
      start = end;
    }
    return subTokenIds;
  }

  // --- Character class helpers (mirror BERT's definitions) ----------------- //

  private isWhitespace(char: string): boolean {
    if (char === " " || char === "\t" || char === "\n" || char === "\r") return true;
    return /\p{Zs}/u.test(char);
  }

  private isControl(char: string): boolean {
    if (char === "\t" || char === "\n" || char === "\r") return false;
    return /\p{Cc}|\p{Cf}/u.test(char);
  }

  private isPunctuation(char: string): boolean {
    const cp = char.codePointAt(0) ?? 0;
    // BERT treats all non-alphanumeric ASCII as punctuation, plus any Unicode P.
    if (
      (cp >= 33 && cp <= 47) ||
      (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) ||
      (cp >= 123 && cp <= 126)
    ) {
      return true;
    }
    return /\p{P}|\p{S}/u.test(char);
  }
}
