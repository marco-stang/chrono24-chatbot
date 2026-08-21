const messagesEl = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const examplesEl = document.getElementById("examples");

const history = [];

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// Rendert das Markdown-Subset der Bot-Antworten (Fett, Listen, Absätze).
// Eingabe wird zuerst HTML-escaped — erst danach entsteht Markup.
function renderMarkdown(text) {
  const inline = (s) => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const blocks = [];
  let listItems = [];
  const flushList = () => {
    if (listItems.length) {
      blocks.push(`<ul>${listItems.map((li) => `<li>${li}</li>`).join("")}</ul>`);
      listItems = [];
    }
  };
  for (const line of text.split("\n")) {
    const item = line.match(/^\s*[-*]\s+(.*)/);
    if (item) {
      listItems.push(inline(item[1]));
    } else {
      flushList();
      if (line.trim()) blocks.push(`<p>${inline(line)}</p>`);
    }
  }
  flushList();
  return blocks.join("");
}

function addMessage(role, text = "") {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addSources(items) {
  const div = document.createElement("div");
  div.className = "sources";
  const links = items.map((s) => {
    const a = document.createElement("a");
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `[${s.n}] ${s.title}`;
    return a;
  });
  div.append("Quellen: ", ...links);
  messagesEl.appendChild(div);
}

function addRetrievalDetails(docs) {
  if (!docs.length) return;
  const details = document.createElement("details");
  details.className = "retrieval";
  const summary = document.createElement("summary");
  summary.textContent = `Retrieval-Details (${docs.length} Treffer)`;
  const list = document.createElement("ul");
  for (const d of docs) {
    const li = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = d.rerank == null
      ? d.score.toFixed(3)
      : `RRF ${d.score.toFixed(3)} → Rerank ${d.rerank.toFixed(2)}`;
    li.append(code, ` ${d.title}`);
    list.appendChild(li);
  }
  details.append(summary, list);
  messagesEl.appendChild(details);
}

const STATUS_ICON = { PASS: "✅", WEAK: "🟡", FAIL: "🔴" };

function addValidationDetails(sentences) {
  if (!sentences.length) return;
  const counts = { PASS: 0, WEAK: 0, FAIL: 0 };
  for (const s of sentences) counts[s.status]++;
  const details = document.createElement("details");
  details.className = "validation";
  const summary = document.createElement("summary");
  const parts = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .map(([status, n]) => `${n} ${STATUS_ICON[status]}`);
  summary.textContent = `Faithfulness-Check (${parts.join(" / ")})`;
  const list = document.createElement("ul");
  for (const s of sentences) {
    const li = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = s.score.toFixed(2);
    li.append(`${STATUS_ICON[s.status]} `, code, ` ${s.text}`);
    list.appendChild(li);
  }
  const legend = document.createElement("p");
  legend.className = "legend";
  legend.textContent =
    "✅ Wortlaut deckt sich mit der Quelle · 🟡 paraphrasiert oder ohne Zitat · 🔴 nicht gedeckt";
  details.append(summary, list, legend);
  messagesEl.appendChild(details);
}

const handoverBtn = document.getElementById("handover");
const NOT_FOUND_TEXT = "Dazu finde ich nichts in den Chrono24-Hilfeseiten.";
const STATUS_LABEL = { ok: "✅ geprüft", rejected: "⛔ abgelehnt" };

function lineTooltip(lines, ids) {
  const byId = new Map(lines.map((l) => [l.id, `${l.actor}: ${l.text}`]));
  return ids.map((id) => `${id} — ${byId.get(id) || "?"}`).join("\n");
}

function briefingRow(label, text, check, lines) {
  const row = document.createElement("div");
  row.className = "briefing-row";
  const icon = check ? STATUS_ICON[check.status] : "";
  const strong = document.createElement("strong");
  strong.textContent = label + ": ";
  row.append(`${icon} `, strong, text);
  if (check && check.sources.length) {
    const ids = document.createElement("span");
    ids.className = "line-ids";
    ids.textContent = ` [${check.sources.join(", ")}]`;
    ids.title = lineTooltip(lines, check.sources);
    row.appendChild(ids);
  }
  return row;
}

function addBriefingCard(result) {
  const card = document.createElement("div");
  card.className = "briefing";
  const head = document.createElement("div");
  head.className = "briefing-head";
  head.textContent = `Übergabe-Briefing · ${STATUS_LABEL[result.status]}`;
  card.appendChild(head);

  if (result.status === "rejected") {
    const p = document.createElement("p");
    p.textContent =
      `Briefing nicht belegbar — der Validator hat ${result.failed_claims.length} ` +
      "Aussage(n) abgelehnt. Der Roh-Verlauf würde übergeben. " +
      "(Demo: es findet keine echte Weiterleitung statt.)";
    card.appendChild(p);
    messagesEl.appendChild(card);
    return;
  }

  // validation-Reihenfolge = situation, history, sentiment-quote, open_question, claims
  const v = result.validation;
  const b = result.briefing;
  card.appendChild(briefingRow("Situation", b.situation.text, v[0], result.lines));
  card.appendChild(briefingRow("Verlauf", b.history.text, v[1], result.lines));
  card.appendChild(briefingRow("Stimmung",
    `${b.sentiment.label} — „${b.sentiment.quote}"`, v[2], result.lines));
  card.appendChild(briefingRow("Offene Frage", b.open_question.text, v[3], result.lines));
  b.claims.forEach((claim, i) => {
    card.appendChild(briefingRow("Aussage", claim.text, v[4 + i], result.lines));
  });
  const legend = document.createElement("p");
  legend.className = "legend";
  legend.textContent =
    "✅ Wortlaut deckt sich mit dem Chat · 🟡 paraphrasiert · 🔴 nicht belegt · (Demo: keine echte Weiterleitung)";
  card.appendChild(legend);
  messagesEl.appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

let handoverInFlight = false;

async function requestHandover() {
  if (handoverInFlight) return;
  handoverInFlight = true;
  handoverBtn.disabled = true;
  handoverBtn.textContent = "Übergebe …";
  try {
    const response = await fetch("/api/handover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-20) }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      addMessage("bot", body.detail || "Übergabe gerade nicht möglich — bitte später erneut versuchen.");
      return;
    }
    addBriefingCard(await response.json());
  } catch {
    addMessage("bot", "Verbindungsfehler bei der Übergabe — bitte gleich nochmal versuchen.");
  } finally {
    handoverInFlight = false;
    handoverBtn.disabled = false;
    handoverBtn.textContent = "An Support übergeben";
  }
}

handoverBtn.addEventListener("click", requestHandover);

function offerHandover() {
  const div = document.createElement("div");
  div.className = "handover-offer";
  div.append("Der Bot weiß hier nicht weiter — ");
  const link = document.createElement("button");
  link.type = "button";
  link.className = "handover-link";
  link.textContent = "an einen Menschen übergeben?";
  link.addEventListener("click", requestHandover);
  div.appendChild(link);
  messagesEl.appendChild(div);
}

async function ask(question) {
  input.value = "";
  sendBtn.disabled = true;
  examplesEl.style.display = "none";
  addMessage("user", question);
  history.push({ role: "user", content: question });
  const botEl = addMessage("bot", "…");
  let answer = "";
  let sourceItems = null;
  let validationSentences = null;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-20) }),
    });
    if (response.status === 429) {
      const body = await response.json().catch(() => ({}));
      botEl.textContent = body.detail || "Zu viele Anfragen — bitte später erneut versuchen.";
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop();
      for (const chunk of chunks) {
        if (!chunk.startsWith("data: ")) continue;
        const event = JSON.parse(chunk.slice(6));
        if (event.type === "token") {
          answer += event.text;
          botEl.innerHTML = renderMarkdown(answer);
        } else if (event.type === "retrieval") {
          addRetrievalDetails(event.docs);
        } else if (event.type === "sources") {
          sourceItems = event.items;
        } else if (event.type === "validation") {
          validationSentences = event.sentences;
        } else if (event.type === "error") {
          botEl.textContent = event.message;
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }
    // Nur die im Antworttext zitierten Quellen [n] anzeigen — alle 5 Kandidaten
    // bleiben im Retrieval-Details-Panel sichtbar.
    if (sourceItems) {
      const cited = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
      const used = sourceItems.filter((s) => cited.has(s.n));
      addSources(used.length ? used : sourceItems.slice(0, 1));
    }
    // Nach den Quellen rendern, damit die [n]-Ampeln unter der Liste stehen,
    // die erklärt, worauf sich [n] bezieht.
    if (validationSentences) addValidationDetails(validationSentences);
    if (answer) {
      history.push({ role: "assistant", content: answer });
      handoverBtn.hidden = false;
      if (answer.trim().endsWith(NOT_FOUND_TEXT)) offerHandover();
    }
  } catch {
    botEl.textContent = "Verbindungsfehler — bitte gleich nochmal versuchen.";
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

examplesEl.addEventListener("click", (e) => {
  if (!sendBtn.disabled && e.target.classList.contains("example")) ask(e.target.textContent);
});
