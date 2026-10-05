const $ = (id) => document.getElementById(id);
let key = "",
  sid = "",
  mode = "documents",
  busy = false,
  docs = [],
  workspaces = [],
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
  $("connect").disabled = value;
  [
    "send",
    "files",
    "new-session",
    "sessions",
    "question",
    "chat-image",
    "rename-workspace",
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
    const row = element("label", null, "document");
    const checkbox = element("input");
    checkbox.type = "checkbox";
    checkbox.checked = d.selected !== false;
    checkbox.addEventListener("change", () => (d.selected = checkbox.checked));
    const info = element("span", d.name);
    info.append(
      element(
        "small",
        `${d.pages} page${d.pages === 1 ? "" : "s"} · ${d.unread_pages ? "Reading…" : "Ready"}`,
      ),
    );
    row.append(checkbox, info);
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
    $("resources").append(row);
  });
  $("extensions").replaceChildren();
  pending.forEach((p) => {
    const row = element(
      "div",
      `More ${labels[p.resource || p.r] || p.resource || p.r || "resources"} requested`,
      "extension",
    );
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
    $("evidence").append(card);
  });
}
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
      await api(`/v1/sessions/${sid}/documents?read=background`, {
        method: "POST",
        body,
      });
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
