/* MoMo scam detector, in the browser. A line-for-line port of src/scrub.py, src/rules.py,
   the TF-IDF + logistic regression model (src/text_model.py) and src/predictor.py.
   web/parity_test.js checks it gives the same scores as the Python code. */
(function (root) {
  "use strict";

  // Python's str-pattern \w and \b are Unicode-aware; JS's are ASCII-only even with /u.
  // pyRe rewrites \w, \W and \b into Unicode classes so both languages agree.
  const W = "\\p{L}\\p{N}_";
  const B = `(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))`;
  function pyRe(src, flags) {
    let out = "", inClass = false;
    for (let i = 0; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") {
        const n = src[i + 1];
        i++;
        if (n === "w") out += inClass ? W : `[${W}]`;
        else if (n === "W" && !inClass) out += `[^${W}]`;
        else if (n === "b" && !inClass) out += B;
        else out += "\\" + n;
        continue;
      }
      if (c === "[" && !inClass) inClass = true;
      else if (c === "]" && inClass) inClass = false;
      out += c;
    }
    return new RegExp(out, (flags || "") + "u");
  }

  // ---------------------------------------------------------------- scrub.py
  const CURRENCY = String.raw`(?:GH[S₵C¢]|GH\s?₵|GHC|GHS|₵|¢|cedis?|MZN|MTS?|MT|KES|KSH|NGN|₦|ZAR|AUD|USD|Rs\.?|INR|£|\$|€)`;
  const AMOUNT_NUM = String.raw`\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?`;
  const PATTERNS = [
    ["<URL>", pyRe(String.raw`\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|ly|gh|co|info|xyz|link|top|site|online)(?:/\S*)?\b`, "gi")],
    ["<EMAIL>", pyRe(String.raw`\b[\w.+-]+@[\w-]+\.[\w.-]+\b`, "g")],
    ["<AMOUNT>", pyRe(`${CURRENCY}\\s?(?:${AMOUNT_NUM})|(?:${AMOUNT_NUM})\\s?${CURRENCY}(?![a-z])`, "gi")],
    ["<DATE>", pyRe(String.raw`\b\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?\b`, "g")],
    ["<TIME>", pyRe(String.raw`\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AP]M)?\b`, "gi")],
    ["<PHONE>", pyRe(String.raw`(?<![\w])(?:\+?\d{1,3}[\s-]?)?\(?0?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{3,4}(?![\w])`, "g")],
    ["<TXID>", pyRe(String.raw`\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6,}\b|\b\d{8,}\b`, "g")],
    ["<NUM>", pyRe(String.raw`\b\d+(?:[.,]\d+)*\b`, "g")],
  ];
  const NOT_NAMES = String.raw`(?!(?:MTN|TELECEL|VODAFONE|AIRTELTIGO|AT|GCB|ECG|DSTV|BOLT|MOMO|GHS|GRA|SSNIT|NHIS)\b)`;
  const NAME = pyRe(
    String.raw`(?<=\bfrom\s)` + NOT_NAMES + String.raw`(?:[A-Z][A-Z'.-]+\s?){1,4}(?=\W|$)` +
    String.raw`|(?<=\bto\s)` + NOT_NAMES + String.raw`(?:[A-Z][A-Z'.-]+\s){1,3}[A-Z][A-Z'.-]+(?=\W|$)` +
    String.raw`|(?<=<PHONE> - )(?:[A-Z][A-Z'.-]+\s?){1,4}`, "g");
  const USSD = pyRe(String.raw`\*\d+(?:\*\d+)*#`, "g");
  const LABELLED_ID = pyRe(String.raw`\b((?:financial\s+)?(?:transaction|trans|txn)\s*id|ref(?:erence)?(?:\s*no)?)\s*[:.]?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b`, "gi");
  const MM_BRAND = pyRe(String.raw`\b(m[-_ ]?pesa|m[-_ ]?weight|mpesa|e-?mola|mkesh|ponto ?24|point ?24|p24|mobile money|momo|telecel cash|vodafone cash|airteltigo money|at money)\b`, "gi");

  function scrub(text) {
    if (typeof text !== "string") return "";
    text = text.replace(MM_BRAND, "MoMo");
    text = text.replace(LABELLED_ID, "$1: <TXID>");
    const ussd = text.match(USSD) || [];
    text = text.replace(USSD, "\x00");
    for (const [token, pat] of PATTERNS) text = text.replace(pat, token);
    for (const code of ussd) text = text.replace("\x00", () => code);
    text = text.replace(NAME, "<NAME> ");
    return text.replace(/\s+/gu, " ").trim();
  }

  // ---------------------------------------------------------------- rules.py
  const R = (p) => pyRe(p, "i");
  const RULES = [
    ["pin_otp_request", "Asks for your PIN, OTP or verification code. No genuine provider will ask for these.",
      R(String.raw`\b(send|share|give|enter|reply|tell|provide|confirm|dial)\b.{0,40}\b(pin|otp|one[- ]time|verification code|secret code|code)\b|\b(pin|otp)\b.{0,30}\b(send|share|reply|confirm)`), 2.0],
    ["mistaken_transfer", "Claims money was sent to you by mistake. This is the most common MoMo scam script.",
      R(String.raw`\b(by mistake|mistakenly|wrong(ly)? (number|transfer|transaction|account)|in error|erroneous(ly)?)\b`), 2.0],
    ["send_back", "Asks you to send or return money.",
      R(String.raw`\b(send|return|refund|reverse|transfer)\b.{0,25}\b(it |the money |the amount )?back\b|\b(kindly|please|pls)\b.{0,20}\b(send|return|refund)\b`), 1.5],
    ["account_threat", "Threatens to block, suspend or close your account.",
      R(String.raw`\b(block(ed)?|suspend(ed)?|deactivat\w*|clos(e|ed)|frozen|freeze|restrict\w*)\b.{0,40}\b(account|wallet|sim|line|number|momo)\b|\b(account|wallet|sim|line|momo)\b.{0,40}\b(block(ed)?|suspend(ed)?|deactivat\w*|frozen|restrict\w*)\b`), 1.5],
    ["urgency", "Pressures you to act immediately.",
      R(String.raw`\b(urgent(ly)?|immediately|right now|asap|within \S+ (hours?|minutes?|mins?)|today only|act now|expires?|last chance|now or)\b`), 1.0],
    ["prize_reward", "Claims you have won a prize, promotion or reward.",
      R(String.raw`\b(you('ve| have)? (won|been selected)|winner|congratulations|congrats|prize|lottery|jackpot|promo(tion)? (winner|reward)|claim your|bonus (of|reward))\b`), 1.5],
    ["fee_to_receive", "Asks you to pay a fee before you can receive money or a reward.",
      R(String.raw`\b(processing|registration|activation|clearance|release|delivery|customs|service) (fee|charge)\b|\bpay\b.{0,30}\b(to (receive|claim|release|unlock))\b`), 1.5],
    ["link", "Contains a link. Genuine MoMo alerts do not ask you to click links.",
      R(String.raw`<URL>|https?://|www\.`), 1.0],
    ["call_number", "Asks you to call or contact a phone number.",
      R(String.raw`\b(call|contact|whatsapp|text|sms|reach)\b.{0,20}(<PHONE>|\b0\d{9}\b)`), 1.0],
    ["verify_account", "Asks you to verify, update or reactivate your account details.",
      R(String.raw`\b(verify|update|confirm|validate|reactivate|unblock|kyc)\b.{0,30}\b(account|details|information|identity|ghana card|wallet|sim)\b`), 1.5],
    ["impersonation", "Claims to come from a provider, agent or customer care.",
      R(String.raw`\b(customer (care|service)|mtn|telecel|vodafone|airteltigo|momo (agent|office|team)|mobile money (office|team)|head office|security team|fraud (team|department))\b`), 0.5],
    ["dial_code", "Tells you to dial a USSD code (scammers use this to make you authorise payments).",
      R(String.raw`\b(dial|press|enter)\b.{0,15}\*\d+`), 1.5],
  ].map(([name, reason, pattern, weight]) => ({ name, reason, pattern, weight }));
  const RULE_THRESHOLD = 2.0;

  function checkRules(text) {
    const hits = RULES.filter((r) => r.pattern.test(text || ""));
    return { score: hits.reduce((s, r) => s + r.weight, 0), hits };
  }

  // ----------------------------------------------------- TF-IDF + LR model
  const TOKEN = pyRe(String.raw`\b\w\w+\b`, "g");

  function wordFeatures(text) {
    const toks = text.toLowerCase().match(TOKEN) || [];
    const grams = toks.slice();
    for (let i = 0; i + 1 < toks.length; i++) grams.push(toks[i] + " " + toks[i + 1]);
    return grams;
  }

  function charFeatures(text) {
    // sklearn char_wb: each whitespace-separated word padded with one space, n = 3..5.
    const grams = [];
    for (const word of text.toLowerCase().split(/\s+/u).filter(Boolean)) {
      const w = Array.from(" " + word + " ");
      for (let n = 3; n <= 5; n++) {
        let offset = 0;
        grams.push(w.slice(offset, offset + n).join(""));
        while (offset + n < w.length) {
          offset += 1;
          grams.push(w.slice(offset, offset + n).join(""));
        }
        if (offset === 0) break;
      }
    }
    return grams;
  }

  // Sublinear TF x IDF, L2-normalised, as sklearn's TfidfVectorizer(sublinear_tf=True) does.
  function vectorise(grams, table) {
    const counts = new Map();
    for (const g of grams) if (table[g]) counts.set(g, (counts.get(g) || 0) + 1);
    const vals = [];
    let norm = 0;
    for (const [g, c] of counts) {
      const v = (1 + Math.log(c)) * table[g][0];
      vals.push([g, v]);
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    return vals.map(([g, v]) => [g, v / norm, table[g][1]]);
  }

  function makeModel(json) {
    return {
      score(clean) {
        const w = vectorise(wordFeatures(clean), json.word);
        const c = vectorise(charFeatures(clean), json.char);
        let z = json.intercept;
        for (const [, x, coef] of w) z += x * coef;
        for (const [, x, coef] of c) z += x * coef;
        const contrib = w.map(([g, x, coef]) => [g, x * coef]).sort((a, b) => b[1] - a[1]);
        return { p: 1 / (1 + Math.exp(-z)), contrib };
      },
    };
  }

  // ------------------------------------------------------------- predictor.py
  const ADVICE = {
    mistaken_transfer: "Check your real balance and history through the official MoMo menu (*170# on MTN) or the official app. A genuine wrong transfer is reversed by the network, never by you sending money.",
    send_back: "Do not send money just because a message or caller asks you to. Check your real transaction history first.",
    pin_otp_request: "Never share your PIN or OTP with anyone, including people who say they work for MTN or Telecel.",
    dial_code: "Do not dial a code or approve a prompt that a stranger tells you to. It may authorise a payment out of your wallet.",
    account_threat: "Providers do not close accounts by SMS from a personal number. Call your provider's official customer care line yourself, using a number from their official website.",
    verify_account: "Update your details only at an official service centre or in the official app, never through a link or a number in an SMS.",
    prize_reward: "Genuine promotions never ask you to pay a fee or share a PIN to collect a prize.",
    fee_to_receive: "Paying a fee to receive money is a classic scam pattern.",
    link: "Do not open links in unexpected SMS messages.",
  };
  const DEFAULT_ADVICE = "If in doubt, check your balance through the official MoMo menu and report suspicious messages to your provider (MTN: 1515).";
  const PLACEHOLDER_WORDS = new Set(["phone", "amount", "num", "txid", "url", "email", "name", "date", "time"]);

  function analyse(model, text) {
    const clean = scrub(text);
    const { p, contrib } = model.score(clean);
    const { score: ruleScore, hits } = checkRules(clean);
    const risk = Math.max(p, Math.min(1, ruleScore / 5));
    const level = risk >= 0.7 ? "high" : risk >= 0.4 ? "medium" : "low";
    const advice = hits.filter((h) => ADVICE[h.name]).map((h) => ADVICE[h.name]);
    return {
      risk, level, modelProbability: p, ruleScore,
      reasons: hits.map((h) => h.reason),
      ruleNames: hits.map((h) => h.name),
      advice: advice.length ? [...new Set(advice)].slice(0, 3) : [DEFAULT_ADVICE],
      influentialWords: contrib.slice(0, 15)
        .filter(([g, c]) => c > 0 && g.length > 2 && !g.split(" ").some((t) => PLACEHOLDER_WORDS.has(t)))
        .slice(0, 6).map(([g]) => g),
      scrubbedText: clean,
    };
  }

  const api = { scrub, checkRules, makeModel, analyse, RULE_THRESHOLD };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MoMoDetector = api;
})(typeof self !== "undefined" ? self : this);
