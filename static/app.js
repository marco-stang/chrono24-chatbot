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

function briefingRow(label, text, check, lines) {
  const row = document.createElement("div");
  row.className = "briefing-row";
  const icon = check ? STATUS_ICON[check.status] : "";
  const strong = document.createElement("strong");
  strong.textContent = label + ": ";
  row.append(`${icon} `, strong, text);
  if (check && check.sources.length) {
    // "Beleg prüfen" wie im Schwesterprojekt: wörtliches Zeilenzitat + Score.
    const details = document.createElement("details");
    details.className = "belege";
    const summary = document.createElement("summary");
    summary.textContent =
      `Beleg prüfen [${check.sources.join(", ")}] · Score ${check.score.toFixed(2)}`;
    details.appendChild(summary);
    const byId = new Map(lines.map((l) => [l.id, l]));
    for (const id of check.sources) {
      const quote = document.createElement("blockquote");
      const line = byId.get(id);
      quote.textContent = line
        ? `${id} · ${line.actor}: ${line.text}`
        : `${id} — Zeile nicht gefunden`;
      details.appendChild(quote);
    }
    row.appendChild(details);
  }
  return row;
}

function addBriefingCard(result, target = "Support") {
  const card = document.createElement("div");
  card.className = "briefing";
  const head = document.createElement("div");
  head.className = "briefing-head";
  head.textContent = `Übergabe-Briefing · für: ${target} · ${STATUS_LABEL[result.status]}`;
  card.appendChild(head);

  if (result.status === "rejected") {
    const p = document.createElement("p");
    p.textContent =
      `Briefing nicht belegbar — der Validator hat ${result.failed_claims.length} ` +
      "Aussage(n) abgelehnt. Der Roh-Verlauf würde übergeben. " +
      "(Demo: es findet keine echte Weiterleitung statt.)";
    card.appendChild(p);
    scenarioContainer().appendChild(card);
    return;
  }

  // Der Mehrwert in einem Satz — wie im Schwesterprojekt.
  const merit = document.createElement("p");
  merit.className = "merit";
  merit.textContent =
    `Statt ${result.lines.length} Nachrichten zu lesen, bekommt der ` +
    "übernehmende Agent dieses Briefing — jede Aussage mit geprüfter Quelle.";
  card.appendChild(merit);

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
  scenarioContainer().appendChild(card);
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
      // detail ist bei 422 ein Array von Objekten — nur echte Strings anzeigen.
      const detail = typeof body.detail === "string" ? body.detail : null;
      addMessage("bot", detail || "Übergabe gerade nicht möglich — bitte später erneut versuchen.");
      return;
    }
    addBriefingCard(await response.json(), pendingHandoverTarget || "Support");
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

// --- Demo-Szenarien: drei gestellte Sackgassen-Verläufe in Akten. Akt 1 zeigt
// jeweils den funktionierenden Bot, danach läuft das Gespräch in die Sackgasse —
// die Übergabe mit live geprüftem Briefing ist die Auflösung. Max. 12
// Nachrichten pro Szenario, damit die M-Badges der Server-Nummerierung
// entsprechen (der Endpoint trimmt auf die letzten 12). ---
const ACTOR_LABEL = { user: "Kunde", assistant: "Bot", agent: "Support" };
const ROLE_CLASS = { user: "user", assistant: "bot", agent: "agent" };

const SCENARIOS = [
  {
    label: "🎬 Uhr nicht angekommen",
    acts: [
      { title: "Der Bot funktioniert", messages: [
        { role: "user", content: "Ich habe vor zwei Wochen eine Omega Speedmaster bei einem Händler gekauft. Wie lange dauert der Versand normalerweise?" },
        { role: "assistant", content: "Der Verkäufer versendet in der Regel innerhalb weniger Werktage, der Versand ist versichert [1]. Über den Sendungsstatus wirst du per E-Mail informiert [2]." },
      ] },
      { title: "Es wird konkret — die FAQ reicht nicht mehr", messages: [
        { role: "user", content: "Es sind jetzt aber schon 14 Tage. Bestellnummer C24-88123. Der Händler antwortet nicht auf meine Nachrichten." },
        { role: "assistant", content: "Der Käuferschutz sichert deine Zahlung ab — das Geld liegt beim Treuhandservice, bis du die Uhr erhalten hast [1]. Bei Problemen kontaktiere das Support-Team [2]." },
        { role: "user", content: "Das habe ich alles gelesen. Ich will wissen, was jetzt mit MEINER Bestellung passiert." },
        { role: "assistant", content: "Auf einzelne Bestellungen habe ich keinen Zugriff — der Käuferschutz gilt, solange die Zahlung über Chrono24 abgewickelt wurde [1]." },
      ] },
      { title: "Die Sackgasse", handoverTarget: "Tier-1-Support", messages: [
        { role: "user", content: "Kannst du prüfen, ob mein Geld noch beim Treuhandservice liegt, und den Händler mahnen?" },
        { role: "assistant", content: NOT_FOUND_TEXT },
      ] },
    ],
  },
  {
    label: "🎬 Zollfrage aus der Schweiz",
    acts: [
      { title: "Der Bot funktioniert", messages: [
        { role: "user", content: "Ich kaufe aus der Schweiz. Ist der Versand dorthin möglich?" },
        { role: "assistant", content: "Internationaler Versand ist möglich und versichert [1]. Je nach Land können Zollgebühren anfallen [2]." },
      ] },
      { title: "Die Rückgabe wird kompliziert", messages: [
        { role: "user", content: "Die Uhr passt mir nicht, ich will sie zurückgeben. Wer zahlt den Rückversand in die EU?" },
        { role: "assistant", content: "Innerhalb der 14-tägigen Widerrufsfrist kannst du den Kauf rückgängig machen [1]. Die Details zum Rückversand klärst du mit dem Verkäufer [2]." },
        { role: "user", content: "Und die Schweizer Zollgebühren, die ich beim Import schon bezahlt habe — bekomme ich die zurück?" },
        { role: "assistant", content: "In den Hilfeseiten steht nur, dass Zollgebühren anfallen können [2] — zu deren Erstattung finde ich dort keine Angaben." },
      ] },
      { title: "Die Sackgasse", handoverTarget: "Tier-1-Support", messages: [
        { role: "user", content: "Welches Formular brauche ich beim Schweizer Zoll für die Rückerstattung, und wen kontaktiere ich dort?" },
        { role: "assistant", content: NOT_FOUND_TEXT },
      ] },
    ],
  },
  {
    label: "🎬 Widersprüchliche Angaben",
    acts: [
      { title: "Der Bot funktioniert", messages: [
        { role: "user", content: "Wie kann ich bei Chrono24 bezahlen?" },
        { role: "assistant", content: "Per Banküberweisung oder Kreditkarte über den Treuhandservice [1] — das Geld wird erst nach Erhalt der Uhr an den Verkäufer freigegeben [2]." },
      ] },
      { title: "Die Angaben widersprechen sich", messages: [
        { role: "user", content: "Ich habe vor zwei Wochen per Überweisung bezahlt, aber der Status zeigt immer noch 'Zahlung ausstehend'." },
        { role: "assistant", content: "Banküberweisungen brauchen einige Werktage, bis sie beim Treuhandservice eingehen [1]." },
        { role: "user", content: "Moment — es war doch letzte Woche mit Kreditkarte, glaube ich. Auf jeden Fall ist das Geld weg." },
        { role: "assistant", content: "Kreditkartenzahlungen werden sofort verbucht [1]. Den konkreten Zahlungsstatus deiner Bestellung kann ich nicht einsehen." },
      ] },
      { title: "Die Sackgasse", handoverTarget: "Tier-1-Support", messages: [
        { role: "user", content: "Egal wie — wo ist mein Geld? Kann das bitte jemand prüfen?" },
        { role: "assistant", content: NOT_FOUND_TEXT },
      ] },
      { title: "Tier 1 übernimmt — und muss selbst eskalieren", handoverTarget: "Tier-2-Support", messages: [
        { role: "agent", content: "Hallo, hier Tier-1-Support. Ich sehe zwei unterschiedliche Angaben: Überweisung vor zwei Wochen und Kreditkarte letzte Woche. Welche stimmt?" },
        { role: "user", content: "Ich bin nicht sicher — vielleicht beides? Mein Mann hat eventuell auch noch etwas überwiesen." },
        { role: "agent", content: "Das kann ich auf Tier 1 nicht auflösen, dafür braucht es die Buchhaltung mit Kontoeinsicht." },
        { role: "user", content: "Dann gebt es bitte weiter — ich will einfach mein Geld zurück." },
      ] },
    ],
  },
];

// Durchklickbar in Akten wie im Schwesterprojekt, an einer vertikalen
// Zeitachse: Akte sind ●-Stationen, Übergaben ◆-Meilensteine.
let activeScenario = null;
let scenarioStep = 0;       // Anzahl bereits gezeigter Akte
let scenarioMsgCount = 0;   // fortlaufender M-Index über alle Akte
let pendingHandoverTarget = null;

function scenarioContainer() {
  return (activeScenario && document.getElementById("scenario-timeline")) || messagesEl;
}

function addScenarioNote() {
  const div = document.createElement("div");
  div.className = "scenario-note";
  div.textContent =
    "🎬 Gestelltes Szenario in Akten — Akt 1 zeigt den funktionierenden Bot, " +
    "dann läuft das Gespräch in die Sackgasse. An jedem ◆-Meilenstein " +
    "übergibst du: das Briefing entsteht live und jede Aussage wird gegen " +
    "den Verlauf geprüft.";
  messagesEl.appendChild(div);
}

function removeScenarioNav() {
  const nav = document.getElementById("scenario-nav");
  if (nav) nav.remove();
}

function renderActMessages(messages) {
  const container = scenarioContainer();
  for (const m of messages) {
    history.push(m);
    const el = addMessage(ROLE_CLASS[m.role], m.content);
    container.appendChild(el);
    scenarioMsgCount++;
    // Sichtbare Zeilen-ID wie in der Akte des Schwesterprojekts — deckungs-
    // gleich mit den M-IDs, die der Server im Briefing zitiert.
    const badge = document.createElement("span");
    badge.className = "line-badge";
    badge.textContent =
      `M${String(scenarioMsgCount).padStart(2, "0")} · ${ACTOR_LABEL[m.role]}`;
    el.prepend(badge);
  }
}

function tlMilestone(target) {
  pendingHandoverTarget = target;
  const div = document.createElement("div");
  div.className = "tl-milestone";
  div.append(`◆ Übergabe → ${target} — `);
  const link = document.createElement("button");
  link.type = "button";
  link.className = "handover-link";
  link.textContent = "jetzt übergeben ▸";
  link.addEventListener("click", requestHandover);
  div.appendChild(link);
  handoverBtn.hidden = false;
  handoverBtn.classList.add("pulse");
  setTimeout(() => handoverBtn.classList.remove("pulse"), 4000);
  return div;
}

function renderScenarioNav() {
  removeScenarioNav();
  const total = activeScenario.acts.length;
  const nav = document.createElement("div");
  nav.id = "scenario-nav";
  nav.className = "scenario-nav";
  nav.append(`Akt ${scenarioStep} von ${total} · `);
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "handover-link";
  if (scenarioStep < total) {
    btn.textContent = "Nächster Akt ▸";
    btn.addEventListener("click", advanceScenario);
  } else {
    btn.textContent = "↺ Von vorn";
    const scenario = activeScenario;
    btn.addEventListener("click", () => loadScenario(scenario));
  }
  nav.appendChild(btn);
  messagesEl.appendChild(nav);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function advanceScenario() {
  const act = activeScenario.acts[scenarioStep];
  const container = scenarioContainer();
  const station = document.createElement("div");
  station.className = "tl-station";
  station.textContent = `● Akt ${scenarioStep + 1} · ${act.title}`;
  container.appendChild(station);
  renderActMessages(act.messages);
  if (act.handoverTarget) container.appendChild(tlMilestone(act.handoverTarget));
  scenarioStep++;
  renderScenarioNav();
}

function loadScenario(scenario) {
  messagesEl.replaceChildren();
  history.length = 0;
  examplesEl.style.display = "none";
  activeScenario = scenario;
  scenarioStep = 0;
  scenarioMsgCount = 0;
  pendingHandoverTarget = null;
  addScenarioNote();
  const timeline = document.createElement("div");
  timeline.id = "scenario-timeline";
  timeline.className = "timeline";
  messagesEl.appendChild(timeline);
  advanceScenario();
}

const scenariosEl = document.getElementById("scenarios");
for (const scenario of SCENARIOS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "example";
  btn.textContent = scenario.label;
  btn.addEventListener("click", () => {
    // Nicht während eines laufenden Streams oder Handovers — loadScenario
    // leert Chat und History und würde die laufende Antwort korrumpieren.
    if (!handoverBtn.disabled && !sendBtn.disabled) loadScenario(scenario);
  });
  scenariosEl.appendChild(btn);
}

async function ask(question) {
  // Eigene Frage beendet den geführten Szenario-Modus; die M-Badges bleiben
  // korrekt, weil neue Nachrichten hinten angehängt werden.
  removeScenarioNav();
  activeScenario = null;
  pendingHandoverTarget = null;
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
      // Support-Zeilen aus Szenarien herausfiltern — der Chat-Endpoint
      // kennt nur user/assistant, der Handover-Endpoint alle drei Rollen.
      body: JSON.stringify({
        messages: history.filter((m) => m.role !== "agent").slice(-20),
      }),
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
