const $ = (id) => document.getElementById(id);
let key = "",
  sid = "",
  mode = "documents",
  busy = false,
  docs = [],
  workspaces = [],
  viewer = null,
  viewerLoading = false,
  image = null,
  turns = [];
const labels = {
  llm_tokens: "Model tokens",
  ocr_pages: "OCR pages",
  vlm_looks: "Vision looks",
  tool_calls: "Tool calls",
  seconds: "Time (seconds)",
};
try {
  key = sessionStorage.getItem("omni-key") || "";
} catch {}
function element(tag, text, className) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (className) e.className = className;
  return e;
}
function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function setBusy(value) {
  busy = value;
  document
    .querySelectorAll("[data-mode]")
    .forEach((button) => (button.disabled = value));
  $("connect").disabled = value;
  [
    "send",
    "files",
    "new-session",
    "sessions",
    "question",
    "chat-image",
    "rename-workspace",
    "upload-reading",
    "edit-limits",
  ].forEach((id) => ($(id).disabled = value || !sid));
  $("send").textContent = value ? "Working…" : "Send ↑";
}
async function api(path, opts = {}) {
  const headers = { Authorization: `Bearer ${key}`, ...opts.headers };
  if (opts.body && !(opts.body instanceof FormData))
    headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...opts, headers });
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(`Request failed (${response.status})`);
  }
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : result.error || `Request failed (${response.status})`,
    );
  return result;
}
function json(method, body) {
  return { method, body: JSON.stringify(body) };
}
function resetConversation() {
  turns = [];
  $("messages")
    .querySelectorAll(".message")
    .forEach((e) => e.remove());
  $("welcome").hidden = false;
  $("export").disabled = true;
  $("evidence").replaceChildren(
    element(
      "p",
      "Source excerpts appear here after your next document question.",
    ),
  );
}
async function refreshSessions() {
  const sessions = await api("/v1/sessions");
  workspaces = sessions;
  $("sessions").replaceChildren();
  sessions.forEach((s, i) => {
    const option = element(
      "option",
      `${s.name || `Workspace ${i + 1}`} · ${s.documents} file${s.documents === 1 ? "" : "s"}`,
    );
    option.value = s.session;
    $("sessions").append(option);
  });
  $("sessions").value = sid;
  $("session-count").textContent = sessions.length;
  rememberWorkspace();
  $("budget-label").textContent = "Current workspace";
}
function rememberWorkspace() {
  try {
    sessionStorage.setItem("omni-workspace", sid);
  } catch {}
}
async function restoreConversation() {
  const result = await api(`/v1/sessions/${sid}/history?mode=${mode}`);
  resetConversation();
  turns = result.messages;
  for (const turn of turns) {
    const article = message(
      turn.role,
      turn.content + (turn.attachment ? "\n[Image attached]" : ""),
      turn.result,
    );
    if (turn.result) {
      feedback(article, turn.result, sid);
      renderFields(article, turn.result, sid);
      renderEvidence(turn.result);
    }
  }
  $("export").disabled = !turns.length;
}
async function newWorkspace() {
  showError("");
  const result = await api(
    "/v1/sessions",
    json("POST", { preset: $("preset").value }),
  );
  sid = result.session;
  docs = [];
  image = null;
  $("image-name").textContent = "";
  $("chat-image").value = "";
  resetConversation();
  renderDocs();
  renderBudget(result.budget);
  $("budget-label").textContent = $("preset").selectedOptions[0].textContent;
  await refreshSessions();
  setBusy(false);
}
function renderDocs() {
  $("doc-count").textContent = docs.length;
  $("documents").replaceChildren();
  if (!docs.length)
    $("documents").append(
      element("p", "No files yet. Add a document to begin.", "subtle"),
    );
  docs.forEach((d) => {
    const row = element("div", null, "document");
    const selection = element("label");
    const checkbox = element("input");
    checkbox.type = "checkbox";
    checkbox.checked = d.selected !== false;
    checkbox.addEventListener("change", () => (d.selected = checkbox.checked));
    const info = element("span", d.name);
    info.append(
      element(
        "small",
        `${d.pages} page${d.pages === 1 ? "" : "s"} · ${d.unread_pages ? `${d.unread_pages} need OCR` : "Ready"}`,
      ),
    );
    selection.append(checkbox, info);
    const preview = element("button", "Read", "secondary");
    preview.setAttribute("aria-label", `Read ${d.name}`);
    preview.addEventListener("click", () => openPage(d.doc_id, 1));
    row.append(selection, preview);
    $("documents").append(row);
  });
}
async function refreshDocs() {
  if (!sid) return;
  const selected = new Map(docs.map((d) => [d.doc_id, d.selected]));
  docs = (await api(`/v1/sessions/${sid}/documents`)).map((d) => ({
    ...d,
    selected: selected.get(d.doc_id) !== false,
  }));
  renderDocs();
}
function renderBudget(budget, pending = []) {
  $("resources").replaceChildren();
  Object.entries(budget).forEach(([resource, b]) => {
    const row = element("div", null, "resource");
    const title = element("div");
    title.append(
      element("span", labels[resource] || resource),
      element("span", `${b.used} / ${b.limit}`),
    );
    const meter = element("progress");
    meter.max = Math.max(b.limit, 1);
    meter.value = Math.min(b.used, meter.max);
    meter.setAttribute("aria-label", labels[resource] || resource);
    row.append(title, meter);
    if (b.cap < b.limit)
      row.append(element("small", `Agent cap: ${b.cap}`, "subtle"));
    $("resources").append(row);
  });
  $("extensions").replaceChildren();
  pending.forEach((p) => {
    const row = element(
      "div",
      `${p.extra ?? "More"} more ${labels[p.resource || p.r] || p.resource || p.r || "resources"} requested`,
      "extension",
    );
    if (p.reason) row.append(element("p", p.reason, "subtle"));
    for (const [label, approve] of [
      ["Approve", true],
      ["Decline", false],
    ]) {
      const button = element("button", label, "secondary");
      button.addEventListener("click", async () => {
        try {
          await api(
            `/v1/sessions/${sid}/budget/resolve`,
            json("POST", { request_id: p.id, approve }),
          );
          await refreshBudget();
        } catch (e) {
          showError(e.message);
        }
      });
      row.append(button);
    }
    $("extensions").append(row);
  });
}
async function refreshBudget() {
  const result = await api(`/v1/sessions/${sid}/budget`);
  renderBudget(result.budget, result.pending_requests);
}
$("edit-limits").addEventListener("click", async () => {
  setBusy(true);
  showError("");
  try {
    const result = await api(`/v1/sessions/${sid}/budget`);
    $("limit-fields").replaceChildren();
    for (const [resource, title] of Object.entries(labels)) {
      const label = element("label", title);
      const input = element("input");
      input.type = "number";
      input.min = "0";
      input.step = resource === "seconds" ? "0.1" : "1";
      input.name = resource;
      input.required = true;
      input.value = result.budget[resource].limit;
      input.setAttribute("aria-label", `${title} limit`);
      label.append(
        input,
        element("small", `${result.budget[resource].used} used`, "subtle"),
      );
      $("limit-fields").append(label);
    }
    $("limits-error").hidden = true;
    $("limits-dialog").showModal();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});
$("cancel-limits").addEventListener("click", () => $("limits-dialog").close());
$("limits-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const limits = Object.fromEntries(
    [...$("limit-fields").querySelectorAll("input")].map((input) => [
      input.name,
      Number(input.value),
    ]),
  );
  if (
    Object.entries(limits).some(
      ([r, v]) =>
        !Number.isFinite(v) ||
        v < 0 ||
        (r !== "seconds" && !Number.isSafeInteger(v)),
    )
  ) {
    $("limits-error").textContent =
      "Use nonnegative whole numbers for resource counts and a finite time limit.";
    $("limits-error").hidden = false;
    return;
  }
  setBusy(true);
  $("save-limits").disabled = true;
  $("cancel-limits").disabled = true;
  try {
    const result = await api(
      `/v1/sessions/${sid}/budget/limits`,
      json("POST", { limits }),
    );
    renderBudget(result.budget, result.pending_requests);
    $("limits-dialog").close();
  } catch (error) {
    $("limits-error").textContent = error.message;
    $("limits-error").hidden = false;
  } finally {
    $("save-limits").disabled = false;
    $("cancel-limits").disabled = false;
    setBusy(false);
  }
});
function message(role, text, result) {
  $("welcome").hidden = true;
  const article = element("article", null, `message ${role}`);
  const meta = element("div", null, "meta");
  meta.append(element("strong", role === "user" ? "You" : "Omni"));
  if (result?.verified !== undefined)
    meta.append(
      element(
        "span",
        result.verified ? "✓ Evidence matched" : "Review needed",
        result.verified ? "verified" : "warning",
      ),
    );
  article.append(meta, element("div", text, "body"));
  if (result?.warning || result?.budget_note)
    article.append(
      element("p", result.warning || result.budget_note, "warning"),
    );
  if (result?.understood_as)
    article.append(
      element("small", `Understood as: ${result.understood_as}`, "subtle"),
    );
  $("messages").append(article);
  $("messages").scrollTop = $("messages").scrollHeight;
  return article;
}
function renderEvidence(result) {
  $("evidence").replaceChildren();
  const evidence =
    result.evidence ||
    result.fields
      ?.filter((f) => f.value)
      .map((f) => ({
        name: f.doc,
        doc: f.doc_id,
        page: f.page,
        text: `${f.field}: ${f.value}`,
      })) ||
    [];
  if (!evidence.length)
    $("evidence").append(
      element(
        "p",
        "No source excerpt was returned. This answer may need your review.",
      ),
    );
  evidence.forEach((e) => {
    const card = element("article", null, "citation");
    card.append(
      element(
        "strong",
        `${e.name || "Document"}${e.page ? ` · Page ${e.page}` : ""}`,
      ),
      element("p", e.text || ""),
    );
    const matching = docs.filter((d) => d.name === e.name);
    const docId = e.doc || (matching.length === 1 ? matching[0].doc_id : null);
    if (docId && e.page) {
      const read = element("button", "Read page text", "secondary");
      read.addEventListener("click", () => openPage(docId, e.page));
      card.append(read);
    }
    $("evidence").append(card);
  });
}
function renderFields(article, result, sessionId) {
  if (result.mode !== "extract" || !Array.isArray(result.fields)) return;
  article.querySelector(".body").textContent =
    "Review the extracted fields and their sources below.";
  const section = element("section", null, "extracted-fields");
  const head = element("div", null, "field-table-heading");
  head.append(
    element(
      "strong",
      `${result.fields.length} extracted field${result.fields.length === 1 ? "" : "s"}`,
    ),
  );
  const download = element("button", "Export all fields CSV", "secondary");
  download.addEventListener("click", async () => {
    download.disabled = true;
    showError("");
    try {
      const response = await fetch(
        `/v1/sessions/${sessionId}/extractions/${encodeURIComponent(result.id)}.csv`,
        { headers: { Authorization: `Bearer ${key}` } },
      );
      if (!response.ok)
        throw new Error(
          response.status === 404
            ? "This extraction is no longer in the retained workspace history."
            : `Export failed (${response.status})`,
        );
      const url = URL.createObjectURL(await response.blob());
      const link = element("a");
      link.href = url;
      link.download = "omni-extracted-fields.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      showError(error.message);
    } finally {
      download.disabled = false;
    }
  });
  head.append(download);
  section.append(head);
  const filters = element("div", null, "field-filters");
  const search = element("input");
  search.type = "search";
  search.placeholder = "Find a field, value, or document";
  search.setAttribute("aria-label", "Search extracted fields");
  const status = element("select");
  status.setAttribute("aria-label", "Field evidence status");
  for (const [value, label] of [["all", "All evidence statuses"], ["matched", "Evidence matched"], ["review", "Review needed"]]) {
    const option = element("option", label);
    option.value = value;
    status.append(option);
  }
  const clear = element("button", "Clear field filters", "secondary");
  filters.append(search, status, clear);
  section.append(filters);
  const count = element("p", null, "subtle");
  count.setAttribute("role", "status");
  section.append(count);
  const scroller = element("div", null, "field-table-scroll");
  const table = element("table");
  table.setAttribute("aria-label", "Extracted fields");
  const thead = element("thead"),
    header = element("tr");
  for (const title of ["Field", "Value", "Source", "Status"]) {
    const th = element("th", title);
    th.scope = "col";
    header.append(th);
  }
  thead.append(header);
  table.append(thead);
  const body = element("tbody");
  const fieldRows = [];
  for (const field of result.fields) {
    const row = element("tr");
    fieldRows.push({row, field, text: [field.field, field.value ?? "Not found", field.doc ?? "", field.page ?? ""].join(" ").toLowerCase()});
    row.append(
      element("td", field.field),
      element("td", field.value ?? "Not found"),
    );
    const source = element("td");
    if (field.doc) {
      const name = `${field.doc}${field.page ? ` · Page ${field.page}` : ""}`;
      if (field.doc_id && field.page) {
        const button = element("button", name, "source-link");
        button.addEventListener("click", () =>
          openPage(field.doc_id, field.page),
        );
        source.append(button);
      } else source.textContent = name;
    } else source.textContent = "No source matched";
    row.append(
      source,
      element(
        "td",
        field.verified ? "Evidence matched" : "Review needed",
        field.verified ? "verified" : "warning",
      ),
    );
    body.append(row);
  }
  table.append(body);
  scroller.append(table);
  section.append(scroller);
  const empty = element("p", "No fields match these filters.", "subtle");
  section.append(empty);
  function filterFields() {
    const query = search.value.trim().toLowerCase();
    let shown = 0;
    for (const item of fieldRows) {
      const match = item.text.includes(query) && (status.value === "all" || (status.value === "matched" ? !!item.field.verified : !item.field.verified));
      item.row.hidden = !match;
      if (match) shown++;
    }
    count.textContent = `${shown} of ${fieldRows.length} fields shown. CSV exports all fields.`;
    empty.hidden = shown > 0 || !fieldRows.length;
  }
  search.addEventListener("input", filterFields);
  status.addEventListener("change", filterFields);
  clear.addEventListener("click", () => {
    search.value = "";
    status.value = "all";
    filterFields();
    search.focus();
  });
  filterFields();
  section.append(
    element(
      "p",
      "Swipe sideways to see every column.",
      "field-table-hint subtle",
    ),
  );
  if (!result.fields.length)
    section.append(element("p", "No field rows were returned.", "subtle"));
  article.append(section);
}
function viewerError(text) {
  $("page-error").textContent = text;
  $("page-error").hidden = !text;
}
function renderPageText() {
  const text = viewer?.text || "No readable text is cached for this page yet.";
  const query = $("page-search").value.trim();
  $("page-text").replaceChildren();
  if (query.length < 2) {
    $("page-text").textContent = text;
    return;
  }
  let position = 0,
    match;
  while (
    (match = text.toLowerCase().indexOf(query.toLowerCase(), position)) !== -1
  ) {
    $("page-text").append(
      document.createTextNode(text.slice(position, match)),
      element("mark", text.slice(match, match + query.length)),
    );
    position = match + query.length;
  }
  $("page-text").append(document.createTextNode(text.slice(position)));
}
function viewerControls() {
  $("previous-page").disabled = viewerLoading || !viewer || viewer.page <= 1;
  $("next-page").disabled =
    viewerLoading || !viewer || viewer.page >= viewer.pages;
  $("page-search-form").querySelector("button").disabled = viewerLoading;
}
async function loadPage(number) {
  if (!viewer || viewerLoading) return;
  const current = viewer;
  viewerLoading = true;
  viewerControls();
  viewerError("");
  try {
    const result = await api(
      `/v1/sessions/${current.session}/documents/${current.doc}/pages/${number}`,
    );
    if (viewer !== current) return;
    Object.assign(viewer, result);
    $("page-title").textContent = result.name;
    $("page-position").textContent = `Page ${result.page} of ${result.pages}`;
    $("page-source").textContent = result.needs_ocr
      ? "This page still needs OCR. Search covers cached text only; ask a document question to read it within your budget."
      : result.source === "ocr"
        ? "OCR text · Check uncertain readings against your original file."
        : "Extracted document text · Text preview, without original page formatting.";
    renderPageText();
  } catch (error) {
    if (viewer === current) viewerError(error.message);
  } finally {
    if (viewer === current) {
      viewerLoading = false;
      viewerControls();
    }
  }
}
function openPage(doc, number) {
  if (busy) return;
  viewer = { session: sid, doc, page: number, pages: number, text: "" };
  viewerLoading = false;
  $("page-title").textContent = "Loading document…";
  $("page-source").textContent = "";
  $("page-position").textContent = "";
  $("page-text").textContent = "";
  $("page-search").value = "";
  $("page-matches").replaceChildren();
  $("page-dialog").showModal();
  loadPage(number);
}
$("close-page").addEventListener("click", () => $("page-dialog").close());
$("page-dialog").addEventListener("close", () => {
  viewer = null;
  viewerLoading = false;
});
$("previous-page").addEventListener("click", () => loadPage(viewer.page - 1));
$("next-page").addEventListener("click", () => loadPage(viewer.page + 1));
$("page-search").addEventListener("input", () => {
  $("page-matches").replaceChildren();
  renderPageText();
});
$("page-search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!viewer || viewerLoading) return;
  const current = viewer,
    query = $("page-search").value.trim();
  if (query.length < 2) {
    viewerError("Search needs at least two characters.");
    return;
  }
  viewerLoading = true;
  viewerControls();
  viewerError("");
  try {
    const result = await api(
      `/v1/sessions/${current.session}/documents/${current.doc}/search?q=${encodeURIComponent(query)}`,
    );
    if (viewer !== current || $("page-search").value.trim() !== query) return;
    $("page-matches").replaceChildren(
      element(
        "p",
        `${result.total_matching_pages} matching page${result.total_matching_pages === 1 ? "" : "s"}${result.unread_pages ? ` · ${result.unread_pages} unread page(s) not fully searchable` : ""}`,
      ),
    );
    if (result.total_matching_pages > result.matches.length)
      $("page-matches").append(
        element("p", "Showing the first 50 matching pages."),
      );
    for (const match of result.matches) {
      const button = element(
        "button",
        `Page ${match.page} · ${match.excerpt}`,
        "secondary",
      );
      button.addEventListener("click", () => loadPage(match.page));
      $("page-matches").append(button);
    }
  } catch (error) {
    if (viewer === current) viewerError(error.message);
  } finally {
    if (viewer === current) {
      viewerLoading = false;
      viewerControls();
    }
  }
});
function feedback(article, result, sessionId) {
  if (!result.id || result.mode === "extract") return;
  const row = element("div", null, "feedback");
  const good = element("button", "Helpful", "secondary");
  const correct = element("button", "Correct answer", "secondary");
  const editor = element("form", null, "correction");
  editor.hidden = true;
  const input = element("textarea");
  input.rows = 2;
  input.required = true;
  input.placeholder = "Enter the corrected answer";
  input.setAttribute("aria-label", "Corrected answer");
  const save = element("button", "Save correction", "secondary");
  save.type = "submit";
  editor.append(input, save);
  async function send(verdict, correction) {
    try {
      const outcome = await api(
        `/v1/sessions/${sessionId}/feedback`,
        json("POST", { id: result.id, verdict, correction }),
      );
      if (!outcome.ok)
        throw new Error(outcome.error || "Feedback could not be saved");
      good.textContent = "Feedback saved";
      good.disabled = true;
      correct.disabled = true;
      editor.hidden = true;
    } catch (e) {
      showError(e.message);
    }
  }
  good.addEventListener("click", () => send("good"));
  correct.addEventListener("click", () => {
    editor.hidden = !editor.hidden;
    input.focus();
  });
  editor.addEventListener("submit", (e) => {
    e.preventDefault();
    send("bad", input.value);
  });
  row.append(good, correct);
  article.append(row, editor);
}
async function upload(files) {
  if (!sid || busy) return;
  setBusy(true);
  showError("");
  try {
    for (const file of files) {
      const body = new FormData();
      body.append("file", file);
      await api(
        `/v1/sessions/${sid}/documents?read=${$("upload-reading").value}`,
        {
          method: "POST",
          body,
        },
      );
    }
    await refreshDocs();
    await refreshSessions();
  } catch (e) {
    showError(e.message);
  } finally {
    setBusy(false);
    $("files").value = "";
  }
}
$("files").addEventListener("change", (e) => upload(e.target.files));
for (const event of ["dragenter", "dragover"])
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("dragging");
  });
$("dropzone").addEventListener("dragleave", () =>
  $("dropzone").classList.remove("dragging"),
);
$("dropzone").addEventListener("drop", (e) => {
  e.preventDefault();
  $("dropzone").classList.remove("dragging");
  upload(e.dataTransfer.files);
});
$("new-session").addEventListener("click", async () => {
  setBusy(true);
  try {
    await newWorkspace();
  } catch (e) {
    showError(e.message);
  } finally {
    setBusy(false);
  }
});
$("sessions").addEventListener("change", async (e) => {
  sid = e.target.value;
  image = null;
  $("chat-image").value = "";
  $("image-name").textContent = "";
  resetConversation();
  setBusy(true);
  try {
    await refreshDocs();
    await refreshBudget();
    await restoreConversation();
    rememberWorkspace();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});
$("rename-workspace").addEventListener("click", () => {
  $("workspace-name").value =
    workspaces.find((s) => s.session === sid)?.name || "";
  $("name-error").hidden = true;
  $("name-dialog").showModal();
  $("workspace-name").focus();
});
$("cancel-name").addEventListener("click", () => $("name-dialog").close());
$("name-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("workspace-name").value.trim();
  if (!name) {
    $("name-error").textContent = "Enter a workspace name.";
    $("name-error").hidden = false;
    return;
  }
  setBusy(true);
  try {
    await api(`/v1/sessions/${sid}`, json("PATCH", { name }));
    await refreshSessions();
    $("name-dialog").close();
  } catch (error) {
    $("name-error").textContent = error.message;
    $("name-error").hidden = false;
  } finally {
    setBusy(false);
  }
});
document.querySelectorAll("[data-mode]").forEach((button) =>
  button.addEventListener("click", async () => {
    if (busy) return;
    mode = button.dataset.mode;
    document.querySelector(".resource-panel").hidden = mode === "chat";
    document
      .querySelectorAll("[data-mode]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    $("extract-option").hidden = mode === "chat";
    $("image-option").hidden = mode !== "chat";
    $("mode-note").textContent =
      mode === "chat"
        ? "Text and image conversations"
        : "Answers grounded in your files";
    $("question").placeholder =
      mode === "chat"
        ? "Ask a question or attach an image…"
        : "Ask anything about your documents…";
    image = null;
    $("chat-image").value = "";
    $("image-name").textContent = "";
    if (sid) {
      setBusy(true);
      try {
        await restoreConversation();
      } catch (error) {
        resetConversation();
        showError(error.message);
      } finally {
        setBusy(false);
      }
    } else resetConversation();
  }),
);
document.querySelectorAll("[data-question]").forEach((button) =>
  button.addEventListener("click", () => {
    $("question").value = button.dataset.question;
    $("extract").checked = !!button.dataset.extract;
    $("question").focus();
  }),
);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
$("chat-image").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    showError("Choose an image smaller than 10 MB.");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    image = reader.result;
    $("image-name").textContent = file.name;
  };
  reader.readAsDataURL(file);
});
$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!sid || busy || !question) return;
  const selected = docs
    .filter((d) => d.selected !== false)
    .map((d) => d.doc_id);
  if (mode === "documents" && !selected.length) {
    showError("Upload and select a document first, or switch to Vision chat.");
    return;
  }
  setBusy(true);
  showError("");
  message("user", question);
  turns.push({ role: "user", content: question });
  $("question").value = "";
  try {
    let result;
    if (mode === "documents") {
      result = await api(
        `/v1/sessions/${sid}/ask`,
        json("POST", {
          question,
          doc_ids: selected,
          mode: $("extract").checked ? "extract" : "auto",
        }),
      );
      const text =
        result.answer ||
        result.fields
          ?.map(
            (f) =>
              `${f.field}: ${f.value ?? "Not found"}${f.verified ? "" : " (unverified)"}`,
          )
          .join("\n") ||
        "No fields found.";
      const article = message("assistant", text, result);
      feedback(article, result, sid);
      renderFields(article, result, sid);
      renderEvidence(result);
      turns.push({ role: "assistant", content: text, result });
      await refreshBudget();
      await refreshDocs();
    } else {
      const content = image
        ? [
            { type: "text", text: question },
            { type: "image_url", image_url: { url: image } },
          ]
        : question;
      result = await api(
        "/v1/chat/completions",
        json("POST", {
          messages: [{ role: "user", content }],
          session_id: sid,
          max_tokens: 512,
        }),
      );
      const text =
        result.choices?.[0]?.message?.content || "No answer returned.";
      message("assistant", text);
      turns.push({ role: "assistant", content: text });
      image = null;
      $("image-name").textContent = "";
      $("chat-image").value = "";
    }
    $("export").disabled = false;
  } catch (error) {
    showError(error.message);
    message(
      "assistant",
      "The request could not finish. Your question is above; try again after checking the connection.",
    );
  } finally {
    setBusy(false);
    $("question").focus();
  }
});
$("export").addEventListener("click", () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify({ session: sid, messages: turns }, null, 2)], {
      type: "application/json",
    }),
  );
  const a = element("a");
  a.href = url;
  a.download = "omni-conversation.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("theme").addEventListener("click", () => {
  const theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("omni-theme", theme);
  } catch {}
});
$("connect").addEventListener("click", () => {
  $("auth-dialog").showModal();
  $("api-key").focus();
});
$("cancel-auth").addEventListener("click", () => $("auth-dialog").close());
$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const previous = key;
  key = $("api-key").value.trim();
  $("auth-error").hidden = true;
  try {
    const sessions = await api("/v1/sessions");
    if (sessions.length) {
      sid = sessions[sessions.length - 1].session;
      await refreshSessions();
      await refreshDocs();
      await refreshBudget();
      await restoreConversation();
      setBusy(false);
    } else {
      sid = "";
      await newWorkspace();
    }
    try {
      sessionStorage.setItem("omni-key", key);
    } catch {}
    $("api-key").value = "";
    $("auth-dialog").close();
    $("connect").textContent = "Connected";
  } catch (error) {
    key = previous;
    $("auth-error").textContent = error.message;
    $("auth-error").hidden = false;
  }
});
async function health() {
  try {
    const response = await fetch("/health");
    const result = await response.json();
    $("health").textContent = result.model_server
      ? "Text model online"
      : "Text model offline";
    $("health-dot").classList.toggle("ready", !!result.model_server);
  } catch {
    $("health").textContent = "Server unavailable";
  }
}
health();
setInterval(health, 30000);
setInterval(() => {
  if (sid && !busy && docs.some((d) => d.unread_pages))
    refreshDocs().catch(() => {});
}, 4000);
if (key) {
  api("/v1/sessions")
    .then(async (sessions) => {
      if (sessions.length) {
        let saved = "";
        try {
          saved = sessionStorage.getItem("omni-workspace");
        } catch {}
        sid =
          sessions.find((s) => s.session === saved)?.session ||
          sessions[sessions.length - 1].session;
        await refreshSessions();
        await refreshDocs();
        await refreshBudget();
        await restoreConversation();
        setBusy(false);
      } else await newWorkspace();
      $("connect").textContent = "Connected";
    })
    .catch(() => {
      key = "";
      $("auth-dialog").showModal();
    });
}
