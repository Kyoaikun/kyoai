
import json
import os
import re
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, simpledialog

from openai import OpenAI

import local_agent as la


class KyoaiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kyoai Local")
        self.root.geometry("900x700")

        self.client = OpenAI(
            base_url=os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key="local",
        )
        self.model = os.environ.get("LLAMA_MODEL", "qwen2.5-14b-q5km")

        self.user_root_base = os.environ.get("LLAMA_USER_ROOT", r"F:\kimi\agent\users")
        self.user_name = os.environ.get("LLAMA_USER_NAME", "default")
        self.user_root = Path(self.user_root_base) / self.user_name
        la.set_user_paths(str(self.user_root))

        self.sessions = self._load_sessions()
        self.current_session = self.sessions.get("current", "default")
        self.messages = self._init_system_messages()
        self.messages.extend(self._session_messages())
        self.last_assistant_text = ""
        self.last_user_text = ""

        self.auto_save_pattern = os.environ.get("LLAMA_AUTO_SAVE_PATTERN", "0") == "1"
        self.pattern_min_score = la.PATTERN_MIN_SCORE
        self.auto_tune_patterns = True
        self.pattern_target_rate = float(os.environ.get("LLAMA_PATTERN_TARGET_RATE", "0.2"))
        self.pattern_tune_window = int(os.environ.get("LLAMA_PATTERN_TUNE_WINDOW", "50"))
        self._score_history = []

        self.gui_memory = os.environ.get("LLAMA_GUI_MEMORY", "1") == "1"
        self.echo_tool_output = os.environ.get("LLAMA_GUI_ECHO_TOOL_OUTPUT", "0") == "1"
        self.project_root = os.environ.get("LLAMA_PROJECT_ROOT", r"F:\kimi\projects")
        self.base_rag_dirs = list(la.RAG_DIRS)
        self.enable_planner = os.environ.get("LLAMA_GUI_PLANNER", "1") == "1"
        self.safe_mode = os.environ.get("LLAMA_SAFE_MODE", "0") == "1"

        self.profiles_path = Path(os.environ.get("LLAMA_PROFILES_PATH", r"F:\kimi\agent\profiles.json"))
        self.profiles = self._load_profiles()
        self.current_profile = self.profiles[0]["name"] if self.profiles else "default"

        self.settings = self._load_settings()
        self._apply_settings()

        self._build_ui()
        self._render_session_history()
        self._apply_project_scope(self.current_session)

    def _init_system_messages(self):
        system = (
            "You are a local assistant. You can access tools to operate on the local machine. "
            "When the user asks you to create or modify files, do it directly using tools instead of just describing steps. "
            "If a task requires reading or writing files or running commands, you MUST call the appropriate tools; otherwise answer directly without tools. "
            "Be concise. Never run destructive commands. Respond in English only and do not use any other language. "
            "Never claim to have used tools, skills, or edited files unless you actually did. "
            "Do not invent tool outputs, diffs, or actions. Only edit files when the user explicitly asks. "
            "If asked for the last N lines of a file, read only the tail and output only those N lines. "
            "After any file write or edit, verify by listing the directory and reading back the file."
        )
        if self.current_session.startswith("project:"):
            pname = self.current_session.split(":", 1)[1].strip()
            project_path = os.path.join(self.project_root, pname)
            system = system + f" Active project path: {project_path}."
        memory = la.load_memory()
        if memory:
            system = system + " Persistent memory:\n" + memory
        if la.PATTERN_PATH.exists():
            try:
                patterns = la.PATTERN_PATH.read_text(encoding="utf-8", errors="ignore")
                if patterns.strip():
                    system = system + "\nPattern library:\n" + patterns[-2000:]
            except Exception:
                pass
        return [{"role": "system", "content": system}]

    def _build_ui(self):
        self.root.configure(bg="#0f1115")

        header = tk.Frame(self.root, bg="#0f1115")
        header.pack(fill="x", padx=16, pady=(12, 6))

        title = tk.Label(
            header,
            text="Kyoai Local",
            fg="#e6e6e6",
            bg="#0f1115",
            font=("Segoe UI Semibold", 16),
        )
        title.pack(side="left")

        self.score_label = tk.Label(
            header,
            text=self._score_label_text(),
            fg="#9aa0a6",
            bg="#0f1115",
            font=("Segoe UI", 10),
        )
        self.score_label.pack(side="right", padx=(0, 12))

        self.status = tk.Label(
            header,
            text="idle",
            fg="#9aa0a6",
            bg="#0f1115",
            font=("Segoe UI", 10),
        )
        self.status.pack(side="right")

        session_bar = tk.Frame(self.root, bg="#0f1115")
        session_bar.pack(fill="x", padx=16, pady=(0, 6))

        tk.Label(
            session_bar,
            text="Session:",
            fg="#9aa0a6",
            bg="#0f1115",
            font=("Segoe UI", 10),
        ).pack(side="left")

        self.session_var = tk.StringVar(value=self.current_session)
        self.session_box = ttk.Combobox(
            session_bar,
            textvariable=self.session_var,
            values=self._list_sessions(),
            state="readonly",
            width=28,
        )
        self.session_box.pack(side="left", padx=(8, 8))
        self.session_box.bind("<<ComboboxSelected>>", self._on_switch_session)

        self.new_session_entry = tk.Entry(
            session_bar,
            bg="#1a1d23",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Segoe UI", 10),
            relief="flat",
            width=22,
        )
        self.new_session_entry.pack(side="left", padx=(0, 6), ipady=4)

        self.new_btn = ttk.Button(session_bar, text="New", command=self._on_new_session)
        self.new_btn.pack(side="left", padx=(0, 6))

        self.profile_var = tk.StringVar(value=self.current_profile)
        self.profile_box = ttk.Combobox(
            session_bar,
            textvariable=self.profile_var,
            values=[p["name"] for p in self.profiles] or ["default"],
            state="readonly",
            width=14,
        )
        self.profile_box.pack(side="left", padx=(6, 0))
        self.profile_box.bind("<<ComboboxSelected>>", self._on_profile_change)

        self.user_btn = ttk.Button(session_bar, text="User", command=self._on_set_user)
        self.user_btn.pack(side="left", padx=(6, 0))

        self.delete_btn = ttk.Button(session_bar, text="Delete", command=self._on_delete_session)
        self.delete_btn.pack(side="left", padx=(6, 0))

        # compact "More" menu to keep a single row
        self.auto_var = tk.BooleanVar(value=self.auto_save_pattern)
        self.echo_var = tk.BooleanVar(value=self.echo_tool_output)
        self.safe_var = tk.BooleanVar(value=self.safe_mode)

        self.more_btn = tk.Menubutton(session_bar, text="More", bg="#0f1115", fg="#e6e6e6", relief="flat")
        self.more_btn.pack(side="left", padx=(6, 0))
        more_menu = tk.Menu(self.more_btn, tearoff=0)
        more_menu.add_checkbutton(label="Auto Save", variable=self.auto_var, command=self._on_toggle_auto_save)
        more_menu.add_checkbutton(label="Echo Tools", variable=self.echo_var, command=self._on_toggle_echo)
        more_menu.add_checkbutton(label="Safe Mode", variable=self.safe_var, command=self._on_toggle_safe_mode)
        more_menu.add_separator()
        more_menu.add_command(label="Save Pattern", command=self._on_save_pattern)
        more_menu.add_command(label="Patterns", command=self._on_view_patterns)
        more_menu.add_command(label="Logs", command=self._on_view_logs)
        more_menu.add_command(label="Debug", command=self._on_view_debug)
        more_menu.add_command(label="Trace", command=self._on_view_trace)
        more_menu.add_separator()
        more_menu.add_command(label="Project", command=self._on_new_project)
        more_menu.add_command(label="Index", command=self._index_current_rag)
        more_menu.add_separator()
        more_menu.add_command(label=self._min_score_menu_text(), command=lambda: None)
        self._min_score_menu_index = more_menu.index("end")
        self.more_btn["menu"] = more_menu

        self.chat = tk.Text(
            self.root,
            wrap="word",
            bg="#111318",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Consolas", 11),
            padx=12,
            pady=12,
            bd=0,
            highlightthickness=0,
        )
        self.chat.pack(fill="both", expand=True, padx=16, pady=8)
        self.chat.tag_configure("user", foreground="#9fd3ff")
        self.chat.tag_configure("assistant", foreground="#e6e6e6")
        self.chat.tag_configure("error", foreground="#ff6b6b")

        input_frame = tk.Frame(self.root, bg="#0f1115")
        input_frame.pack(fill="x", padx=16, pady=(0, 12))

        self.entry = tk.Entry(
            input_frame,
            bg="#1a1d23",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Segoe UI", 11),
            relief="flat",
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=8)
        self.entry.bind("<Return>", self._on_send)

        self.send_btn = ttk.Button(input_frame, text="Send", command=self._on_send)
        self.send_btn.pack(side="right")

        self.root.bind("<Control-p>", self._open_command_palette)
    def _append(self, text, tag):
        def do():
            self.chat.configure(state="normal")
            self.chat.insert("end", text + "\n\n", tag)
            self.chat.configure(state="disabled")
            self.chat.see("end")
        self.root.after(0, do)

    def _append_error_banner(self, text):
        self._append("ERROR: " + text, "error")
        self._log_error(text)

    def _error_log_path(self):
        return la.SESSIONS_PATH.parent / "gui_errors.log"

    def _log_error(self, text):
        try:
            path = self._error_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            with path.open("a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {text}\n")
        except Exception:
            pass

    def _set_status(self, text):
        self.root.after(0, lambda: self.status.configure(text=text))

    def _set_send_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.root.after(0, lambda: self.send_btn.configure(state=state))

    def _apply_project_scope(self, name):
        if name and name.startswith("project:"):
            pname = name.split(":", 1)[1].strip()
            project_path = os.path.join(self.project_root, pname)
            dirs = [project_path] + self.base_rag_dirs
            la.set_rag_dirs(dirs)
            if not os.path.isdir(project_path):
                self._append(f"Project path not found: {project_path}", "error")
        else:
            la.set_rag_dirs(self.base_rag_dirs)

    def _set_user(self, name):
        self.user_name = name
        self.user_root = Path(self.user_root_base) / name
        la.set_user_paths(str(self.user_root))
        self.settings = self._load_settings()
        self._apply_settings()
        self.sessions = self._load_sessions()
        self.current_session = self.sessions.get("current", "default")
        self.messages = self._init_system_messages()
        self.messages.extend(self._session_messages())
        self._apply_project_scope(self.current_session)
        self._refresh_sessions()
        self._render_session_history()
        self._append(f"User set to: {name}", "assistant")

    def _on_send(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, "end")
        self._append("You: " + user_text, "user")
        self.last_user_text = user_text
        self.messages.append({"role": "user", "content": user_text})
        self._set_status("thinking...")
        self._set_send_enabled(False)

        thread = threading.Thread(target=self._run_model, daemon=True)
        thread.start()

    def _run_model(self):
        try:
            try:
                self.messages = self._trim_messages(self.messages, max_keep=12)
                needs_tools = self._needs_tools(self.last_user_text)
                tool_choice = "required" if needs_tools else "auto"
                self._append(f"Tool policy: {tool_choice}", "assistant")
                plan_msg = None
                if self.enable_planner and self._needs_plan(self.last_user_text):
                    plan_msg = {
                        "role": "system",
                        "content": "If the task is complex or multi-step, first write a brief plan (2-6 bullets), then execute.",
                    }
                    self.messages.append(plan_msg)
                skill_msg = None
                skill_prompt = self._select_skill_prompt(self.last_user_text)
                if skill_prompt:
                    skill_msg = {"role": "system", "content": skill_prompt}
                    self._append(f"Skill: {skill_prompt.split(':', 1)[0].replace('Use the ', '').strip()}", "assistant")
                    self.messages.append(skill_msg)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=la.tools,
                    tool_choice=tool_choice,
                    temperature=0.2,
                    timeout=300,
                )
                msg = response.choices[0].message
                if needs_tools and not msg.tool_calls:
                    self._append("Retrying tool call enforcement...", "assistant")
                    self.messages.append({
                        "role": "system",
                        "content": "Tool enforcement: you MUST call tools to complete this request. Return only tool calls.",
                    })
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=la.tools,
                        tool_choice="required",
                        temperature=0.2,
                        timeout=300,
                    )
                    msg = response.choices[0].message
            except Exception as e:
                self._append_error_banner(str(e))
                self._set_status("error")
                self._set_send_enabled(True)
                return

            tool_trace = []
            while msg.tool_calls:
                self.messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    try:
                        self._append(f"Running tool: {name} {args}", "assistant")
                        self._set_status(f"tool: {name}")
                        if name == "list_dir":
                            result = la.list_dir(**args)
                        elif name == "read_file":
                            result = la.read_file(**args)
                        elif name == "read_tail":
                            result = la.read_tail(**args)
                        elif name == "write_file":
                            if self.safe_mode:
                                result = "Blocked by Safe Mode"
                            elif not self._user_explicit_write_request(self.last_user_text):
                                result = "Blocked: explicit write request required (say edit/write file)."
                                self._append_error_banner("Write blocked: user did not explicitly request a file edit.")
                            else:
                                result = la.write_file(**args)
                        elif name == "run_powershell":
                            if self.safe_mode:
                                result = "Blocked by Safe Mode"
                            else:
                                result = la.run_powershell(**args)
                        elif name == "fetch_url":
                            if self.safe_mode:
                                result = "Blocked by Safe Mode"
                            else:
                                result = la.fetch_url(**args)
                        elif name == "fetch_url_rendered":
                            if self.safe_mode:
                                result = "Blocked by Safe Mode"
                            else:
                                result = la.fetch_url_rendered(**args)
                        elif name == "search_text":
                            result = la.search_text(**args)
                        elif name == "rag_index":
                            result = la.rag_index(**args)
                        else:
                            result = "Unknown tool"
                    except Exception as e:
                        result = f"Tool error: {e}"

                    if isinstance(result, str) and len(result) > 4000:
                        result = result[:4000] + "\n...[truncated]..."

                    tool_trace.append({
                        "tool": name,
                        "args": args,
                        "result_preview": result[:500] if isinstance(result, str) else str(result)[:500],
                    })
                    if self.echo_tool_output:
                        preview = result if isinstance(result, str) else str(result)
                        if len(preview) > 2000:
                            preview = preview[:2000] + "\n...[truncated]..."
                        self._append(f"Tool output ({name}):\n{preview}", "assistant")
                    self._append(f"Tool done: {name}", "assistant")
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": result,
                        }
                    )

                try:
                    self.messages = self._trim_messages(self.messages, max_keep=12)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=la.tools,
                        tool_choice="auto",
                        temperature=0.2,
                        timeout=300,
                    )
                    msg = response.choices[0].message
                except Exception as e:
                    self._append_error_banner(str(e))
                    self._set_status("error")
                    self._set_send_enabled(True)
                    return

            assistant_text = msg.content or ""
            if self._detect_fabricated_tool_log(assistant_text):
                self._append_error_banner("Fabricated tool/skill log detected in assistant response.")
            if needs_tools and self._looks_non_english(assistant_text):
                self.messages.append({
                    "role": "system",
                    "content": "Respond in English only. Do not use any other language.",
                })
                self.messages.append({
                    "role": "user",
                    "content": "Provide the final answer in English only. Keep it concise.",
                })
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=la.tools,
                        tool_choice="auto",
                        temperature=0.2,
                        timeout=300,
                    )
                    msg = response.choices[0].message
                    assistant_text = msg.content or assistant_text
                except Exception:
                    pass
            self._append("Kyoai: " + assistant_text, "assistant")
            self.messages.append({"role": "assistant", "content": assistant_text})
            self.last_assistant_text = assistant_text
            self._set_status("idle")
            self._set_send_enabled(True)
            if plan_msg is not None and plan_msg in self.messages:
                try:
                    self.messages.remove(plan_msg)
                except ValueError:
                    pass
            if skill_msg is not None and skill_msg in self.messages:
                try:
                    self.messages.remove(skill_msg)
                except ValueError:
                    pass
            self._save_session_messages()
            self._append_history_record(self.last_user_text, assistant_text, tool_trace)
            if self.auto_save_pattern:
                self._auto_tune_threshold(self.last_user_text, tool_trace, assistant_text)
                la._append_pattern_entry_structured(
                    user_input=self.last_user_text,
                    tool_trace=tool_trace,
                    assistant_text=assistant_text,
                    source="gui",
                    require_best=True,
                    require_tool=self._should_require_tool(self.last_user_text, assistant_text),
                )
            if self.gui_memory:
                self._update_memory_summary()
        except Exception as e:
            self._append_error_banner(str(e))
            self._set_status("error")
            self._set_send_enabled(True)
            return
    def _load_sessions(self):
        if not la.SESSIONS_PATH.exists():
            return {"current": "default", "sessions": {"default": []}}
        try:
            data = json.loads(la.SESSIONS_PATH.read_text(encoding="utf-8"))
            if "current" not in data or "sessions" not in data:
                return {"current": "default", "sessions": {"default": []}}
            if data["current"] not in data["sessions"]:
                data["sessions"][data["current"]] = []
            return data
        except Exception:
            return {"current": "default", "sessions": {"default": []}}

    def _save_sessions(self):
        la.SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        la.SESSIONS_PATH.write_text(json.dumps(self.sessions, ensure_ascii=False), encoding="utf-8")

    def _list_sessions(self):
        return sorted(self.sessions.get("sessions", {}).keys())

    def _session_messages(self):
        return self.sessions.get("sessions", {}).get(self.current_session, [])

    def _save_session_messages(self):
        safe = self._serialize_messages(self.messages[1:])[-50:]
        self.sessions["sessions"][self.current_session] = safe
        self.sessions["current"] = self.current_session
        self._save_sessions()

    def _refresh_sessions(self):
        self.session_box["values"] = self._list_sessions()
        self.session_var.set(self.current_session)

    def _on_switch_session(self, event=None):
        name = self.session_var.get().strip()
        if not name:
            return
        if name not in self.sessions.get("sessions", {}):
            return
        self.current_session = name
        self.messages = self._init_system_messages()
        self.messages.extend(self._session_messages())
        self._apply_project_scope(self.current_session)
        self._render_session_history()
        self._append(f"Switched to session: {name}", "assistant")

    def _on_new_session(self):
        name = self.new_session_entry.get().strip()
        if not name:
            return
        if name in self.sessions.get("sessions", {}):
            self._append(f"Session already exists: {name}", "error")
            return
        self.sessions["sessions"][name] = []
        self.current_session = name
        self.sessions["current"] = name
        self._save_sessions()
        self._refresh_sessions()
        self._render_session_history()
        self.new_session_entry.delete(0, "end")
        self._append(f"New session: {name}", "assistant")

    def _on_delete_session(self):
        name = self.current_session
        if name == "default":
            self._append("Cannot delete default session.", "error")
            return
        if name not in self.sessions.get("sessions", {}):
            return
        del self.sessions["sessions"][name]
        self.current_session = "default"
        self.sessions["current"] = "default"
        self._save_sessions()
        self._refresh_sessions()
        self._render_session_history()
        self._append(f"Deleted session: {name}", "assistant")

    def _on_new_project(self):
        name = simpledialog.askstring("Project", "Enter project name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        pname = f"project: {name}"
        if pname not in self.sessions.get("sessions", {}):
            self.sessions["sessions"][pname] = []
        self.current_session = pname
        self.sessions["current"] = pname
        self._save_sessions()
        self._refresh_sessions()
        self._render_session_history()
        self._apply_project_scope(self.current_session)
        self._append(f"Project session: {name}", "assistant")

    def _on_set_user(self):
        name = simpledialog.askstring("User", "Enter user name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        self._set_user(name)

    def _on_profile_change(self, event=None):
        name = self.profile_var.get().strip()
        if not name:
            return
        self.current_profile = name
        self._apply_profile(name)
        self._save_settings()
        self._append(f"Profile set to: {name}", "assistant")

    def _on_toggle_auto_save(self):
        self.auto_save_pattern = bool(self.auto_var.get())
        self._save_settings()
        self._append(f"Auto Save: {self.auto_save_pattern}", "assistant")

    def _on_toggle_echo(self):
        self.echo_tool_output = bool(self.echo_var.get())
        self._save_settings()
        self._append(f"Echo Tools: {self.echo_tool_output}", "assistant")

    def _on_toggle_safe_mode(self):
        self.safe_mode = bool(self.safe_var.get())
        self._save_settings()
        self._append(f"Safe Mode: {self.safe_mode}", "assistant")

    def _toggle_safe_mode_cmd(self):
        self.safe_mode = not self.safe_mode
        self.safe_var.set(self.safe_mode)
        self._save_settings()
        self._append(f"Safe Mode: {self.safe_mode}", "assistant")

    def _toggle_echo_cmd(self):
        self.echo_tool_output = not self.echo_tool_output
        self.echo_var.set(self.echo_tool_output)
        self._save_settings()
        self._append(f"Echo Tools: {self.echo_tool_output}", "assistant")

    def _on_save_pattern(self):
        text = self.last_assistant_text.strip()
        if not text:
            self._append("No assistant output to save.", "error")
            return
        la._append_pattern_entry_structured(
            user_input=self.last_user_text,
            tool_trace=[],
            assistant_text=text,
            source="manual",
            require_best=False,
            require_tool=False,
        )
        self._append("Saved pattern to library.", "assistant")

    def _on_view_patterns(self):
        path = la.PATTERN_PATH
        self._open_text_viewer("Patterns", path)

    def _on_view_logs(self):
        top = tk.Toplevel(self.root)
        top.title("Logs")
        top.geometry("720x520")
        top.configure(bg="#0f1115")

        text = tk.Text(
            top,
            wrap="word",
            bg="#111318",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Consolas", 10),
            bd=0,
            highlightthickness=0,
        )
        text.pack(fill="both", expand=True, padx=12, pady=12)

        def render():
            chunks = []
            err_path = self._error_log_path()
            if err_path.exists():
                try:
                    chunks.append("=== gui_errors.log ===\n" + err_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
                except Exception:
                    pass
            server_log = la.SESSIONS_PATH.parent / "server.log"
            if server_log.exists():
                try:
                    chunks.append("=== server.log ===\n" + server_log.read_text(encoding="utf-8", errors="ignore")[-8000:])
                except Exception:
                    pass
            content = "\n\n".join(chunks).strip() or "No logs."
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("end", content)
            text.configure(state="disabled")

        render()

    def _on_view_debug(self):
        info = [
            f"Model: {self.model}",
            f"Base URL: {self.client.base_url if hasattr(self.client, 'base_url') else 'local'}",
            f"User: {self.user_name}",
            f"Profile: {self.current_profile}",
            f"Safe Mode: {self.safe_mode}",
            f"Echo Tools: {self.echo_tool_output}",
            f"Auto Save: {self.auto_save_pattern}",
            f"Project Root: {self.project_root}",
            f"RAG Dirs: {', '.join(la.RAG_DIRS)}",
        ]
        self._open_text_viewer("Debug", "\n".join(info))

    def _on_view_trace(self):
        top = tk.Toplevel(self.root)
        top.title("Trace")
        top.geometry("820x560")
        top.configure(bg="#0f1115")

        query = tk.StringVar(value="")
        qentry = tk.Entry(
            top,
            bg="#1a1d23",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Segoe UI", 10),
            relief="flat",
            textvariable=query,
        )
        qentry.pack(fill="x", padx=12, pady=(12, 6), ipady=4)

        text = tk.Text(
            top,
            wrap="word",
            bg="#111318",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Consolas", 10),
            bd=0,
            highlightthickness=0,
        )
        text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        def render():
            q = query.get().strip().lower()
            content = ""
            if la.HISTORY_PATH.exists():
                try:
                    data = la.HISTORY_PATH.read_text(encoding="utf-8", errors="ignore")
                    lines = data.splitlines()[-200:]
                    blocks = []
                    for line in lines:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        tools = rec.get("tools") or []
                        if not tools:
                            continue
                        header = f"[{rec.get('ts','')}] USER: {rec.get('user','')}"
                        tool_lines = []
                        for t in tools:
                            name = t.get("tool", "unknown")
                            args = t.get("args", {})
                            preview = t.get("result_preview", "")
                            tool_lines.append(f"- {name} {args}\n  {preview}")
                        block = header + "\n" + "\n".join(tool_lines)
                        if q and q not in block.lower():
                            continue
                        blocks.append(block)
                    content = "\n\n".join(blocks).strip() or "No tool traces."
                except Exception:
                    content = "No tool traces."
            if len(content) > 200_000:
                content = content[-200_000:]
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("end", content)
            text.configure(state="disabled")

        refresh = ttk.Button(top, text="Refresh", command=render)
        refresh.pack(side="left", padx=12, pady=(0, 12))

        qentry.bind("<Return>", lambda e: render())
        render()

    def _open_text_viewer(self, title, source):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("720x520")
        top.configure(bg="#0f1115")

        text = tk.Text(
            top,
            wrap="word",
            bg="#111318",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            font=("Consolas", 10),
            bd=0,
            highlightthickness=0,
        )
        text.pack(fill="both", expand=True, padx=12, pady=12)
        content = ""
        if isinstance(source, (str, Path)) and Path(str(source)).exists():
            try:
                content = Path(str(source)).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""
        else:
            content = str(source)
        if len(content) > 200_000:
            content = content[-200_000:]
        text.insert("end", content)
        text.configure(state="disabled")

    def _index_current_rag(self):
        def do():
            self._set_status("indexing...")
            res = la.rag_index()
            self._append(f"RAG index: {res}", "assistant")
            self._set_status("idle")
        threading.Thread(target=do, daemon=True).start()

    def _open_command_palette(self, event=None):
        win = tk.Toplevel(self.root)
        win.title("Command Palette")
        win.geometry("520x360")
        win.configure(bg="#0f1115")

        cmd_map = [
            ("Start Server", self._start_server),
            ("Start Server (Profile)", self._start_server_profile),
            ("Stop Server", self._stop_server),
            ("Clear Current Session", self._clear_current_session),
            ("Export Current Session", self._export_session),
            ("Toggle Safe Mode", self._toggle_safe_mode_cmd),
            ("Toggle Echo Tools", self._toggle_echo_cmd),
            ("Index Current RAG Scope", self._index_current_rag),
        ]

        listbox = tk.Listbox(
            win,
            bg="#111318",
            fg="#e6e6e6",
            selectbackground="#2a2f3a",
            font=("Consolas", 11),
            bd=0,
            highlightthickness=0,
        )
        listbox.pack(fill="both", expand=True, padx=12, pady=12)
        for label, _ in cmd_map:
            listbox.insert("end", label)

        def run_selected(event=None):
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            cmd_map[idx][1]()
            win.destroy()

        listbox.bind("<Return>", run_selected)
        listbox.bind("<Double-Button-1>", run_selected)

    def _clear_current_session(self):
        self.sessions["sessions"][self.current_session] = []
        self._save_sessions()
        self._render_session_history()
        self._append("Cleared current session.", "assistant")

    def _export_session(self):
        path = simpledialog.askstring("Export Session", "Enter output path:")
        if not path:
            return
        path = path.strip()
        if not path:
            return
        lines = []
        for m in self._session_messages():
            role = m.get("role", "unknown")
            content = m.get("content", "")
            lines.append(f"{role.upper()}:\n{content}\n")
        content = "\n".join(lines).strip() + "\n"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
            self._append(f"Exported session to: {path}", "assistant")
        except Exception as e:
            self._append_error_banner(str(e))

    def _start_server(self):
        try:
            subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", r"F:\kimi\agent\start_server.ps1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._append("Started server.", "assistant")
        except Exception as e:
            self._append_error_banner(str(e))

    def _start_server_profile(self):
        prof = self._get_profile(self.current_profile)
        script = None
        if prof:
            script = prof.get("start_script")
        if not script:
            self._append_error_banner("No start_script for profile.")
            return
        try:
            subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._append(f"Started server: {self.current_profile}", "assistant")
        except Exception as e:
            self._append_error_banner(str(e))

    def _stop_server(self):
        if self.safe_mode:
            self._append("Blocked by Safe Mode", "error")
            return
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process"],
                capture_output=True,
                text=True,
            )
            self._append("Stopped server (if running).", "assistant")
        except Exception as e:
            self._append_error_banner(str(e))
    def _trim_messages(self, msgs, max_keep=12):
        if len(msgs) <= max_keep:
            return msgs
        return [msgs[0]] + msgs[-(max_keep - 1):]

    def _needs_tools(self, text):
        if not text:
            return False
        t = text.lower()
        if re.search(r"\b(read|write|edit|create|delete|list|show|open|tail|run|execute|powershell|cmd|shell|directory|folder|file)\b", t):
            return True
        if re.search(r"[a-zA-Z]:\\", text):
            return True
        if "path" in t:
            return True
        return False

    def _user_explicit_write_request(self, text):
        if not text:
            return False
        t = text.lower()
        if t.strip() in {"yes", "y", "ok", "okay", "go ahead", "do it", "proceed", "confirm", "sure"}:
            return True
        # Allow common intent verbs even if the user doesn't mention a file explicitly.
        explicit = re.search(r"\b(edit|write|modify|change|update|create|add|delete|remove|fix|adjust|tweak|move)\b", t)
        if explicit:
            return True
        if "file" in t or "files" in t or "path" in t:
            return True
        if re.search(r"[a-zA-Z]:\\", text):
            return True
        if re.search(r"\.(py|txt|md|json|ps1|bat|cpp|c|h|js|ts|html|css|yaml|yml|toml|ini)\b", t):
            return True
        return False

    def _detect_fabricated_tool_log(self, text):
        if not text:
            return False
        markers = [
            "Tool policy:",
            "Skill:",
            "Running tool:",
            "Tool done:",
            "Tool output",
        ]
        return any(m in text for m in markers)

    def _looks_non_english(self, text):
        if not text:
            return False
        if re.search(r"[^\x00-\x7F]", text):
            return True
        return False

    def _needs_plan(self, text):
        if not text:
            return False
        t = text.lower()
        if any(k in t for k in ["plan", "steps", "step-by-step", "checklist", "roadmap"]):
            return True
        if re.search(r"\b(1\.|1\)|2\.|2\)|3\.|3\))", t):
            return True
        if any(k in t for k in ["first", "then", "after that", "next", "finally"]):
            return True
        if "\n" in text and text.count("\n") >= 2:
            return True
        return False

    def _select_skill_prompt(self, text):
        if not text:
            return ""
        t = text.lower()
        if any(k in t for k in ["bug", "error", "trace", "exception", "stack trace", "crash", "failed"]):
            return "Use the Bug Fix skill: identify root cause, apply minimal fix, verify."
        if any(k in t for k in ["edit", "modify", "change", "update", "refactor", "add code", "implement"]):
            return "Use the Code Edit skill: minimal change, verify by reading back."
        if any(k in t for k in ["scan", "search", "where is", "find", "locate", "repo", "project structure"]):
            return "Use the Repo Scan skill: list, search, read relevant sections, summarize."
        if any(k in t for k in ["test", "build", "run", "compile", "benchmark"]):
            return "Use the Test Run skill: confirm command, run, report key output."
        return ""

    def _render_session_history(self):
        def do():
            self.chat.configure(state="normal")
            self.chat.delete("1.0", "end")
            for m in self._session_messages():
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    self.chat.insert("end", "You: " + content + "\n\n", "user")
                elif role == "assistant":
                    self.chat.insert("end", "Kyoai: " + content + "\n\n", "assistant")
                elif role == "error":
                    self.chat.insert("end", content + "\n\n", "error")
            self.chat.configure(state="disabled")
            self.chat.see("end")
        self.root.after(0, do)

    def _append_history_record(self, user_text, assistant_text, tool_trace):
        try:
            la.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "user": user_text,
                "assistant": assistant_text,
                "tools": tool_trace,
            }
            with la.HISTORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _msg_role(self, msg):
        if isinstance(msg, dict):
            return msg.get("role")
        return getattr(msg, "role", None)

    def _msg_content(self, msg):
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "") or ""

    def _serialize_messages(self, msgs):
        kept = []
        for m in msgs:
            role = self._msg_role(m)
            if role not in {"user", "assistant", "error"}:
                continue
            content = self._msg_content(m)
            kept.append({"role": role, "content": content})
        return kept

    def _update_memory_summary(self):
        try:
            mem_input = "\n".join(
                self._msg_content(m)
                for m in self.messages[-10:]
                if isinstance(self._msg_content(m), str)
            )
            if len(mem_input) > 3000:
                mem_input = mem_input[-3000:]
            mem_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize key user preferences, ongoing tasks, and important facts in <=200 words. Be concise.",
                    },
                    {"role": "user", "content": mem_input},
                ],
                temperature=0.1,
            )
            mem_text = (mem_resp.choices[0].message.content or "").strip()
            if mem_text:
                la.save_memory(mem_text)
        except Exception:
            pass

    def _auto_tune_threshold(self, user_text, tool_trace, assistant_text):
        if not self.auto_tune_patterns:
            return
        score, _ = la._pattern_score(user_text, tool_trace, assistant_text)
        self._score_history.append(score)
        if len(self._score_history) > self.pattern_tune_window:
            self._score_history = self._score_history[-self.pattern_tune_window:]
        if len(self._score_history) < 8:
            return
        scores = sorted(self._score_history)
        target = self.pattern_target_rate
        idx = int((1.0 - target) * len(scores)) - 1
        idx = max(0, min(len(scores) - 1, idx))
        threshold = float(scores[idx])
        self.pattern_min_score = threshold
        la.PATTERN_MIN_SCORE = threshold
        self._update_score_label()
        self._save_settings()

    def _should_require_tool(self, user_text: str, assistant_text: str) -> bool:
        text = f"{user_text}\n{assistant_text}".lower()
        if any(k in text for k in ["read", "write", "edit", "file", "folder", "directory", "path", "run", "execute"]):
            return True
        return False

    def _score_label_text(self):
        if self.auto_tune_patterns:
            return "min score: auto"
        return f"min score: {self.pattern_min_score:.2f}"

    def _min_score_menu_text(self):
        if self.auto_tune_patterns:
            return "Min Score: auto"
        return f"Min Score: {self.pattern_min_score:.2f}"

    def _update_score_label(self):
        if hasattr(self, "score_label"):
            self.score_label.configure(text=self._score_label_text())
        if hasattr(self, "more_btn"):
            try:
                menu = self.more_btn["menu"]
                menu.entryconfig(self._min_score_menu_index, label=self._min_score_menu_text())
            except Exception:
                pass

    def _get_profile(self, name):
        for p in self.profiles:
            if p.get("name") == name:
                return p
        return None

    def _apply_profile(self, name):
        prof = self._get_profile(name)
        if not prof:
            return
        base_url = prof.get("base_url")
        model = prof.get("model")
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key="local")
        if model:
            self.model = model

    def _load_profiles(self):
        if not self.profiles_path.exists():
            return []
        try:
            data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _settings_path(self):
        return self.user_root / "gui_settings.json"

    def _load_settings(self):
        path = self._settings_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self):
        data = {
            "auto_save": self.auto_save_pattern,
            "min_score": self.pattern_min_score,
            "auto_tune": self.auto_tune_patterns,
            "target_rate": self.pattern_target_rate,
            "tune_window": self.pattern_tune_window,
            "echo_tools": self.echo_tool_output,
            "safe_mode": self.safe_mode,
            "profile": self.current_profile,
        }
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _apply_settings(self):
        s = self.settings or {}
        if "auto_save" in s:
            self.auto_save_pattern = bool(s.get("auto_save"))
        if "min_score" in s:
            try:
                self.pattern_min_score = float(s.get("min_score"))
                la.PATTERN_MIN_SCORE = self.pattern_min_score
            except Exception:
                pass
        if "auto_tune" in s:
            self.auto_tune_patterns = bool(s.get("auto_tune"))
        if "target_rate" in s:
            try:
                self.pattern_target_rate = float(s.get("target_rate"))
            except Exception:
                pass
        if "tune_window" in s:
            try:
                self.pattern_tune_window = int(s.get("tune_window"))
            except Exception:
                pass
        if "echo_tools" in s:
            self.echo_tool_output = bool(s.get("echo_tools"))
        if "safe_mode" in s:
            self.safe_mode = bool(s.get("safe_mode"))
        if "profile" in s:
            self.current_profile = s.get("profile") or self.current_profile
            self._apply_profile(self.current_profile)

    def _open_text(self, path):
        try:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def main():
    root = tk.Tk()
    app = KyoaiGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
