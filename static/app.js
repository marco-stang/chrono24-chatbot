const messagesEl = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const examplesEl = document.getElementById("examples");

const history = [];

const MOTION_OK = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
  summary.textContent = `🔍 Gefundene Hilfeseiten (${docs.length})`;
  const list = document.createElement("ul");
  for (const d of docs) {
    const li = document.createElement("li");
    const code = document.createElement("code");
    // Für Nicht-Techniker: ein Wert, ein Wort. Rerank-Logits über Sigmoid
    // in Prozent — RRF-Scores (ohne Rerank) bleiben als Rohwert.
    code.textContent = d.rerank == null
      ? `Relevanz ${d.score.toFixed(2)}`
      : `Relevanz ${Math.round(100 / (1 + Math.exp(-d.rerank)))} %`;
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
  summary.textContent = `Aussagen-Prüfung (${parts.join(" / ")})`;
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
  legend.textContent = "✅ wörtlich belegt · 🟡 sinngemäß · 🔴 keine Quelle";
  details.append(summary, list, legend);
  messagesEl.appendChild(details);
}

const handoverBtn = document.getElementById("handover");
const NOT_FOUND_TEXT = "Dazu finde ich nichts in den Chrono24-Hilfeseiten.";
const STATUS_LABEL = { ok: "✅ geprüft", rejected: "⛔ abgelehnt" };

function briefingRow(label, text, check, lines) {
  const row = document.createElement("div");
  row.className = "briefing-row";
  const icon = document.createElement("span");
  icon.className = "status-icon";
  icon.textContent = check ? STATUS_ICON[check.status] : "";
  const strong = document.createElement("strong");
  strong.textContent = label + ": ";
  // Icon in fester Spalte, Text daneben — Umbruchzeilen bleiben bündig.
  const body = document.createElement("div");
  body.className = "row-body";
  body.append(strong, text);
  row.append(icon, body);
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
    body.appendChild(details);
  }
  return row;
}

// Die Prüfung als Schauspiel: Zeilen erscheinen einzeln, jede läuft kurz als
// "wird geprüft …" mit Spinner, dann klappt das Ampel-Ergebnis ein. Klick auf
// die Karte (oder reduzierte Bewegung) überspringt die Inszenierung.
async function addBriefingCard(result, target = "Support") {
  const card = document.createElement("div");
  card.className = "briefing";
  const head = document.createElement("div");
  head.className = "briefing-head";
  head.textContent = `Übergabe-Briefing · für: ${target} · ${STATUS_LABEL[result.status]}`;
  card.appendChild(head);
  scenarioContainer().appendChild(card);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  if (result.status === "rejected") {
    const p = document.createElement("p");
    p.textContent =
      `Briefing nicht belegbar — der Validator hat ${result.failed_claims.length} ` +
      "Aussage(n) abgelehnt. Der Roh-Verlauf würde übergeben. " +
      "(Demo: es findet keine echte Weiterleitung statt.)";
    card.appendChild(p);
    return;
  }

  let skip = !MOTION_OK;
  card.addEventListener("click", () => { skip = true; });
  const pace = async (ms) => { if (!skip) await sleep(ms); };

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
  const mainRows = [
    briefingRow("Situation", b.situation.text, v[0], result.lines),
    briefingRow("Verlauf", b.history.text, v[1], result.lines),
    briefingRow("Stimmung",
      `${b.sentiment.label} — „${b.sentiment.quote}"`, v[2], result.lines),
    briefingRow("Offene Frage", b.open_question.text, v[3], result.lines),
  ];
  for (const row of mainRows) {
    const icon = row.querySelector(".status-icon");
    const belege = row.querySelector(".belege");
    const finalIcon = icon.textContent;
    icon.textContent = "";
    icon.classList.add("loading");
    if (belege) belege.hidden = true;
    card.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    await pace(650);
    icon.classList.remove("loading");
    icon.textContent = finalIcon;
    if (belege) belege.hidden = false;
  }

  // Einzelaussagen eingeklappt — die Summenzeile zeigt, DASS geprüft wird,
  // die Detailtiefe gibt's erst auf Klick.
  await pace(400);
  if (b.claims.length) {
    const counts = { PASS: 0, WEAK: 0, FAIL: 0 };
    for (const check of v.slice(4)) counts[check.status]++;
    const claimsBox = document.createElement("details");
    claimsBox.className = "claims";
    const claimsSummary = document.createElement("summary");
    const parts = Object.entries(counts)
      .filter(([, n]) => n > 0)
      .map(([status, n]) => `${n} ${STATUS_ICON[status]}`);
    const n = b.claims.length;
    claimsSummary.textContent =
      `${n} Aussage${n === 1 ? "" : "n"} automatisch gegengeprüft (${parts.join(" / ")})`;
    claimsBox.appendChild(claimsSummary);
    b.claims.forEach((claim, i) => {
      claimsBox.appendChild(briefingRow("Aussage", claim.text, v[4 + i], result.lines));
    });
    card.appendChild(claimsBox);
  }
  const legend = document.createElement("p");
  legend.className = "legend";
  legend.textContent =
    "✅ wörtlich belegt · 🟡 sinngemäß · 🔴 nicht belegt · (Demo: keine echte Weiterleitung)";
  card.appendChild(legend);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

let handoverInFlight = false;

async function requestHandover(trigger) {
  if (handoverInFlight) return;
  handoverInFlight = true;
  handoverBtn.disabled = true;
  const originalText = trigger ? trigger.textContent : null;
  if (trigger) {
    trigger.disabled = true;
    trigger.classList.add("loading");
    trigger.classList.remove("pulse");
    trigger.textContent = "Briefing wird erzeugt und geprüft …";
  }
  let succeeded = false;
  try {
    const response = await fetch("/api/handover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-20) }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      // detail ist bei 422 ein Array von Objekten — nur echte Strings anzeigen.
      // slowapi liefert 429 unter dem Key "error" statt "detail".
      const detail = typeof body.detail === "string" ? body.detail : null;
      const fallback = response.status === 429
        ? "Übergabe-Limit erreicht (3 pro Minute, 10 pro Tag) — bitte kurz warten."
        : "Übergabe gerade nicht möglich — bitte später erneut versuchen.";
      addMessage("bot", detail || fallback);
      return;
    }
    await addBriefingCard(await response.json(), pendingHandoverTarget || "Support");
    succeeded = true;
    if (activeScenario && pendingResolution) renderResolution();
  } catch {
    addMessage("bot", "Verbindungsfehler bei der Übergabe — bitte gleich nochmal versuchen.");
  } finally {
    handoverInFlight = false;
    handoverBtn.disabled = false;
    if (trigger) {
      trigger.classList.remove("loading");
      if (succeeded) {
        // Bleibt deaktiviert — verhindert Doppel-Briefings am selben Meilenstein.
        trigger.textContent = "✓ übergeben";
      } else {
        trigger.disabled = false;
        trigger.textContent = originalText;
      }
    }
  }
}

handoverBtn.addEventListener("click", () => requestHandover(handoverBtn));

function offerHandover() {
  const div = document.createElement("div");
  div.className = "handover-offer";
  div.append("Der Bot weiß hier nicht weiter — ");
  const link = document.createElement("button");
  link.type = "button";
  link.className = "handover-link";
  link.textContent = "an einen Menschen übergeben?";
  link.addEventListener("click", () => requestHandover(link));
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

// Echte Hilfeseiten, aus denen die gestellten Bot-Antworten paraphrasiert
// sind — die [n]-Marker der Szenarien verlinken darauf.
const SOURCE_PAGES = {
  faq: { title: "FAQ: Uhren kaufen",
    url: "https://www.chrono24.de/info/faqs.htm#chapter-2" },
  escrow: { title: "Käuferschutz & Treuhandservice",
    url: "https://www.chrono24.de/info/escrow.htm" },
  checkout: { title: "Der Treuhandservice auf Chrono24",
    url: "https://www.chrono24.de/info/c2c-checkout.htm" },
};

const SCENARIOS = [
  {
    label: "Uhr nicht angekommen",
    sub: "Händler antwortet nicht",
    acts: [
      { title: "Der Bot funktioniert", messages: [
        { role: "user", content: "Ich habe vor zwei Wochen eine Omega Speedmaster bei einem Händler gekauft. Wie lange dauert der Versand normalerweise?" },
        { role: "assistant", content: "Der Verkäufer versendet in der Regel innerhalb weniger Werktage, der Versand ist versichert [1]. Über den Sendungsstatus wirst du per E-Mail informiert [2].", sources: { 1: "faq", 2: "faq" } },
      ] },
      { title: "Es wird konkret — die FAQ reicht nicht mehr", messages: [
        { role: "user", content: "Es sind jetzt aber schon 14 Tage. Bestellnummer C24-88123. Der Händler antwortet nicht auf meine Nachrichten." },
        { role: "assistant", content: "Der Käuferschutz sichert deine Zahlung ab — das Geld liegt beim Treuhandservice, bis du die Uhr erhalten hast [1]. Bei Problemen kontaktiere das Support-Team [2].", sources: { 1: "escrow", 2: "faq" } },
        { role: "user", content: "Das habe ich alles gelesen. Ich will wissen, was jetzt mit MEINER Bestellung passiert." },
        { role: "assistant", content: "Auf einzelne Bestellungen habe ich keinen Zugriff — der Käuferschutz gilt, solange die Zahlung über Chrono24 abgewickelt wurde [1].", sources: { 1: "escrow" } },
      ] },
      { title: "Die Sackgasse", handoverTarget: "Tier-1-Support", messages: [
        { role: "user", content: "Kannst du prüfen, ob mein Geld noch beim Treuhandservice liegt, und den Händler mahnen?" },
        { role: "assistant", content: NOT_FOUND_TEXT },
      ] },
    ],
    resolution: [
      { role: "agent", content: "Ich habe Bestellung C24-88123 geprüft: Ihr Geld liegt sicher beim Treuhandservice. Den Händler habe ich heute schriftlich gemahnt — meldet er sich nicht binnen 48 Stunden, leiten wir die Rückabwicklung ein." },
      { role: "user", content: "Danke, das beruhigt mich sehr!" },
    ],
  },
  {
    label: "Zollfrage aus der Schweiz",
    sub: "Zoll bei Rückgabe",
    acts: [
      { title: "Der Bot funktioniert", messages: [
        { role: "user", content: "Ich kaufe aus der Schweiz. Ist der Versand dorthin möglich?" },
        { role: "assistant", content: "Internationaler Versand ist möglich und versichert [1]. Je nach Land können Zollgebühren anfallen [2].", sources: { 1: "faq", 2: "faq" } },
      ] },
      { title: "Die Rückgabe wird kompliziert", messages: [
        { role: "user", content: "Die Uhr passt mir nicht, ich will sie zurückgeben. Wer zahlt den Rückversand in die EU?" },
        { role: "assistant", content: "Innerhalb der 14-tägigen Widerrufsfrist kannst du den Kauf rückgängig machen [1]. Die Details zum Rückversand klärst du mit dem Verkäufer [2].", sources: { 1: "faq", 2: "faq" } },
        { role: "user", content: "Und die Schweizer Zollgebühren, die ich beim Import schon bezahlt habe — bekomme ich die zurück?" },
        { role: "assistant", content: "In den Hilfeseiten steht nur, dass Zollgebühren anfallen können [2] — zu deren Erstattung finde ich dort keine Angaben.", sources: { 2: "faq" } },
      ] },
      { title: "Die Sackgasse", handoverTarget: "Tier-1-Support", messages: [
        { role: "user", content: "Welches Formular brauche ich beim Schweizer Zoll für die Rückerstattung, und wen kontaktiere ich dort?" },
        { role: "assistant", content: NOT_FOUND_TEXT },
      ] },
    ],
    resolution: [
      { role: "agent", content: "Für die Rückerstattung der Schweizer Einfuhrabgaben stellen Sie beim Bundesamt für Zoll und Grenzsicherheit einen Rückerstattungsantrag — Link und Ausfüllhilfe habe ich Ihnen gerade per E-Mail geschickt, den Rückversand klärt der Verkäufer bereits." },
      { role: "user", content: "Perfekt, danke für die schnelle Hilfe!" },
    ],
  },
  {
    label: "Widersprüchliche Angaben",
    sub: "Zahlung verschwunden",
    acts: [
      { title: "Der Bot funktioniert", messages: [
        { role: "user", content: "Wie kann ich bei Chrono24 bezahlen?" },
        { role: "assistant", content: "Per Banküberweisung oder Kreditkarte über den Treuhandservice [1] — das Geld wird erst nach Erhalt der Uhr an den Verkäufer freigegeben [2].", sources: { 1: "checkout", 2: "escrow" } },
      ] },
      { title: "Die Angaben widersprechen sich", messages: [
        { role: "user", content: "Ich habe vor zwei Wochen per Überweisung bezahlt, aber der Status zeigt immer noch 'Zahlung ausstehend'." },
        { role: "assistant", content: "Banküberweisungen brauchen einige Werktage, bis sie beim Treuhandservice eingehen [1].", sources: { 1: "faq" } },
        { role: "user", content: "Moment — es war doch letzte Woche mit Kreditkarte, glaube ich. Auf jeden Fall ist das Geld weg." },
        { role: "assistant", content: "Kreditkartenzahlungen werden sofort verbucht [1]. Den konkreten Zahlungsstatus deiner Bestellung kann ich nicht einsehen.", sources: { 1: "faq" } },
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
    resolution: [
      { role: "agent", content: "Buchhaltung hier: Wir haben beide Zahlungen gefunden — Ihre Überweisung und die Kreditkartenzahlung Ihres Mannes. Die doppelte Zahlung erstatten wir innerhalb von 3 Werktagen zurück." },
      { role: "user", content: "Super, vielen Dank!" },
    ],
  },
];

// Durchklickbar in Akten. Links läuft pro Betreuer (Bot, Tier-1, Tier-2) eine
// eigene Swimlane: Punkt = dieser Betreuer antwortet gerade, ◆ = Übergabe —
// die alte Lane endet, die neue startet. Der Kunde hat keine Lane, seine
// Bubbles bleiben neutral.
let activeScenario = null;
let scenarioStep = 0;       // Anzahl bereits gezeigter Akte
let scenarioMsgCount = 0;   // fortlaufender M-Index über alle Akte
let pendingHandoverTarget = null;
let milestoneContent = null; // Inhaltszelle des letzten Meilensteins
let pendingResolution = null; // Schlussakt — erst nach Briefing am letzten ◆

const LANE_LABEL = { bot: "Bot", t1: "Tier-1", t2: "Tier-2" };
const TARGET_LANE = { "Tier-1-Support": "t1", "Tier-2-Support": "t2" };

let lanes = [];                  // Lanes dieses Szenarios, z.B. ["bot","t1"]
let laneStarted = new Set();     // Lanes, deren Linie schon läuft
let laneEnded = new Set();       // Lanes, deren Linie beendet ist
let ownerLane = "bot";           // wer den Fall gerade betreut

// Briefing-Karten landen beim Meilenstein (Demo) oder im Chatverlauf.
function scenarioContainer() {
  return milestoneContent || messagesEl;
}

// Linke Rail-Zelle einer Zeile: ein Span pro Lane. "transfer" zeichnet das
// Ende der alten und den ◆-Start der neuen Lane in derselben Zeile.
function railCell(kind, speakerLane, fromLane, toLane) {
  const rail = document.createElement("div");
  rail.className = "rail";
  for (const lane of lanes) {
    const span = document.createElement("span");
    span.className = `lane ${lane}`;
    if ((kind === "transfer" || kind === "resolve") && lane === fromLane) {
      span.classList.add("half-top");
    } else if (kind === "transfer" && lane === toLane) {
      span.classList.add("half-bottom", "start");
    } else if (laneStarted.has(lane) && !laneEnded.has(lane)) {
      span.classList.add("line");
    }
    if (lane === speakerLane) span.classList.add("dot");
    rail.appendChild(span);
  }
  return rail;
}

// Eine Zeitachsen-Zeile: Rail links, Inhalt rechts.
function tlRow(kind, speakerLane, fromLane, toLane) {
  const timeline = document.getElementById("scenario-timeline");
  const content = document.createElement("div");
  content.className = `tl-content ${kind}`;
  timeline.append(railCell(kind, speakerLane, fromLane, toLane), content);
  return content;
}

function laneChip(lane) {
  const chip = document.createElement("span");
  chip.className = `lane-chip ${lane}`;
  chip.textContent = LANE_LABEL[lane];
  return chip;
}

function addScenarioNote() {
  const div = document.createElement("div");
  div.className = "scenario-note";
  div.append("Gestellter Fall — klick dich durch. Am ◆ erzeugst du das Briefing live.");
  const chips = document.createElement("span");
  chips.className = "lane-chips";
  for (const lane of lanes) chips.appendChild(laneChip(lane));
  div.appendChild(chips);
  // Kino-Modus: Fall läuft von selbst ab, inklusive Briefing am ◆.
  const play = document.createElement("button");
  play.type = "button";
  play.className = "handover-link play-btn";
  play.textContent = "▶ Automatisch abspielen";
  play.addEventListener("click", () => {
    if (autoplay) {
      autoplay = false;
    } else {
      playScenario(activeScenario);
    }
  });
  div.appendChild(play);
  messagesEl.appendChild(div);
}

function removeScenarioNav() {
  const nav = document.getElementById("scenario-nav");
  if (nav) nav.remove();
}

// Lane des Sprechers: Bot fest, Support-Zeilen gehören dem aktuellen Betreuer.
function speakerLane(role) {
  if (role === "assistant") return "bot";
  if (role === "agent") return ownerLane;
  return null;
}

function renderOneMessage(m) {
  history.push({ role: m.role, content: m.content });
  const lane = speakerLane(m.role);
  const content = tlRow("message", lane);
  const el = addMessage(ROLE_CLASS[m.role], m.content);
  el.classList.add("appear");
  if (m.role === "agent") el.classList.add(ownerLane);
  content.appendChild(el);
  // Klickbare Quellen zu den [n]-Markern der gestellten Bot-Antworten.
  if (m.sources) {
    const line = document.createElement("div");
    line.className = "sources";
    line.append("Quellen: ");
    for (const [n, key] of Object.entries(m.sources)) {
      const page = SOURCE_PAGES[key];
      const a = document.createElement("a");
      a.href = page.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = `[${n}] ${page.title}`;
      line.appendChild(a);
    }
    content.appendChild(line);
  }
  scenarioMsgCount++;
  // Sichtbare Zeilen-ID — deckungsgleich mit den M-IDs, die der Server im
  // Briefing zitiert. Farbe folgt dem Sprecher.
  const badge = document.createElement("span");
  badge.className = `line-badge ${lane || "user"}`;
  badge.textContent =
    `M${String(scenarioMsgCount).padStart(2, "0")} · ${ACTOR_LABEL[m.role]}`;
  el.prepend(badge);
}

function renderActMessages(messages) {
  for (const m of messages) renderOneMessage(m);
}

function renderResolution() {
  const messages = pendingResolution;
  pendingResolution = null;
  const title = tlRow("act", null);
  title.textContent = "Auflösung";
  renderActMessages(messages);
  // Lane des letzten Betreuers endet — der Fall ist abgeschlossen.
  const content = tlRow("resolve", null, ownerLane);
  laneEnded.add(ownerLane);
  const done = document.createElement("div");
  done.className = "case-solved";
  done.textContent = "✓ Fall gelöst";
  content.appendChild(done);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function tlMilestone(target, resolution) {
  pendingHandoverTarget = target;
  pendingResolution = resolution || null;
  const toLane = TARGET_LANE[target] || "t1";
  const content = tlRow("transfer", null, ownerLane, toLane);
  laneEnded.add(ownerLane);
  laneStarted.add(toLane);
  ownerLane = toLane;
  const label = document.createElement("div");
  label.className = `milestone-label ${toLane}`;
  label.textContent = `◆ Übergabe an ${LANE_LABEL[toLane]}`;
  const link = document.createElement("button");
  link.type = "button";
  link.className = "handover-link pulse";
  link.textContent = "Briefing erzeugen und übergeben ▸";
  link.addEventListener("click", () => requestHandover(link));
  content.append(label, link);
  milestoneContent = content;
}

function renderScenarioNav() {
  removeScenarioNav();
  const total = activeScenario.acts.length;
  const nav = document.createElement("div");
  nav.id = "scenario-nav";
  nav.className = "scenario-nav";
  if (autoplay) {
    // Während des Kino-Modus keine klickbare Navigation — Stopp läuft über
    // den Abspielen-Button in der Szenario-Notiz.
    nav.append(`Akt ${scenarioStep} von ${total} · läuft ab …`);
    messagesEl.appendChild(nav);
    return;
  }
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
  const title = tlRow("act", null);
  title.textContent = `Akt ${scenarioStep + 1} · ${act.title}`;
  renderActMessages(act.messages);
  // Auflösung nur am letzten Meilenstein des Szenarios anbieten.
  const isLastAct = scenarioStep === activeScenario.acts.length - 1;
  if (act.handoverTarget) {
    tlMilestone(act.handoverTarget, isLastAct ? activeScenario.resolution : null);
  }
  scenarioStep++;
  renderScenarioNav();
}

// --- Kino-Modus: Akte laufen zeitversetzt ab, Briefings lösen am ◆ von
// selbst aus. Stopp (oder Tab-/Szenariowechsel) rendert den Rest des
// laufenden Akts sofort fertig und gibt die Steuerung zurück. ---
let autoplay = false;

async function advanceScenarioAnimated(scenario) {
  const stopped = () => !autoplay || activeScenario !== scenario;
  const act = scenario.acts[scenarioStep];
  const title = tlRow("act", null);
  title.textContent = `Akt ${scenarioStep + 1} · ${act.title}`;
  for (const m of act.messages) {
    if (activeScenario !== scenario) return;
    if (!stopped()) await sleep(MOTION_OK ? 900 : 400);
    if (activeScenario !== scenario) return;
    renderOneMessage(m);
  }
  const isLastAct = scenarioStep === scenario.acts.length - 1;
  if (act.handoverTarget) {
    tlMilestone(act.handoverTarget, isLastAct ? scenario.resolution : null);
    if (!stopped()) {
      await sleep(600);
      if (!stopped()) {
        const link = milestoneContent.querySelector(".handover-link");
        await requestHandover(link);
      }
    }
  }
  scenarioStep++;
  renderScenarioNav();
}

async function playScenario(scenario) {
  if (autoplay || handoverInFlight || sendBtn.disabled || !scenario) return;
  loadScenario(scenario, true);
  autoplay = true;
  const playBtn = document.querySelector(".play-btn");
  if (playBtn) playBtn.textContent = "■ Stopp";
  renderScenarioNav();
  while (autoplay && activeScenario === scenario
         && scenarioStep < scenario.acts.length) {
    await advanceScenarioAnimated(scenario);
  }
  autoplay = false;
  const btn = document.querySelector(".play-btn");
  if (btn) btn.textContent = "▶ Automatisch abspielen";
  if (activeScenario === scenario) renderScenarioNav();
}

function loadScenario(scenario, deferFirstAct = false) {
  messagesEl.replaceChildren();
  history.length = 0;
  examplesEl.style.display = "none";
  activeScenario = scenario;
  scenarioStep = 0;
  scenarioMsgCount = 0;
  pendingHandoverTarget = null;
  pendingResolution = null;
  milestoneContent = null;
  // Lanes dieses Szenarios: Bot plus alle Übergabe-Ziele in Reihenfolge.
  lanes = ["bot"];
  for (const act of scenario.acts) {
    const lane = TARGET_LANE[act.handoverTarget];
    if (lane && !lanes.includes(lane)) lanes.push(lane);
  }
  laneStarted = new Set(["bot"]);
  laneEnded = new Set();
  ownerLane = "bot";
  addScenarioNote();
  const timeline = document.createElement("div");
  timeline.id = "scenario-timeline";
  timeline.className = "timeline";
  timeline.style.setProperty("--lane-count", lanes.length);
  messagesEl.appendChild(timeline);
  if (!deferFirstAct) advanceScenario();
}

const scenariosEl = document.getElementById("scenarios");
for (const scenario of SCENARIOS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "example scenario-btn";
  const strong = document.createElement("strong");
  strong.textContent = scenario.label;
  const small = document.createElement("small");
  small.textContent = scenario.sub;
  btn.append(strong, small);
  btn.addEventListener("click", () => {
    // Nicht während eines laufenden Streams oder Handovers — loadScenario
    // leert Chat und History und würde die laufende Antwort korrumpieren.
    if (!handoverBtn.disabled && !sendBtn.disabled) loadScenario(scenario);
  });
  scenariosEl.appendChild(btn);
}

// --- Tabs: freier Chat vs. geführte Übergabe-Demo ---
const tabChat = document.getElementById("tab-chat");
const tabDemo = document.getElementById("tab-demo");
const tabIntro = document.getElementById("tab-intro");
const scenariosRow = document.getElementById("scenarios-row");

const TAB_INTRO = {
  chat: "Der Bot antwortet ausschließlich aus einem Snapshot der öffentlichen " +
    "Chrono24-Hilfeseiten (RAG: Stichwort- und Bedeutungssuche kombiniert, " +
    "dann neu sortiert). Jede Antwort nennt ihre Quellen — und eine Ampel " +
    "prüft Satz für Satz, ob das wirklich dort steht. Er erfindet nichts " +
    "dazu — heißt auch: Manchmal sagt er ehrlich „weiß ich nicht\".",
  demo: "Drei gestellte Fälle zeigen den Ernstfall: Der Bot weiß nicht " +
    "weiter und übergibt an Menschen — nicht als roher Chatverlauf, sondern " +
    "als automatisch erzeugtes Briefing. Jede Aussage darin wird live gegen " +
    "das Gespräch geprüft; nicht Belegbares wird abgelehnt.",
};

function switchTab(mode) {
  messagesEl.replaceChildren();
  history.length = 0;
  autoplay = false;
  activeScenario = null;
  pendingHandoverTarget = null;
  pendingResolution = null;
  milestoneContent = null;
  handoverBtn.hidden = true;
  tabIntro.textContent = TAB_INTRO[mode];
  const demo = mode === "demo";
  tabChat.classList.toggle("active", !demo);
  tabDemo.classList.toggle("active", demo);
  form.hidden = demo;
  examplesEl.hidden = demo;
  examplesEl.style.display = demo ? "none" : "flex";
  scenariosRow.hidden = !demo;
}

tabChat.addEventListener("click", () => switchTab("chat"));
tabDemo.addEventListener("click", () => switchTab("demo"));
switchTab("chat");

async function ask(question) {
  // Eigene Frage beendet den geführten Szenario-Modus; die M-Badges bleiben
  // korrekt, weil neue Nachrichten hinten angehängt werden.
  removeScenarioNav();
  autoplay = false;
  activeScenario = null;
  pendingHandoverTarget = null;
  pendingResolution = null;
  milestoneContent = null;
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
      // Neue Antwort = neuer Übergabe-Kontext: "✓ übergeben" zurücksetzen.
      handoverBtn.disabled = false;
      handoverBtn.textContent = "An Support übergeben";
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
