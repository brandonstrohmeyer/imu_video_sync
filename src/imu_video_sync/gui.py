from __future__ import annotations

import contextlib
import queue
import re
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

from . import __version__
from . import analysis
from . import update_check
from .theme import choose_ttkbootstrap_theme

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
    from tkinter import ttk as ttk_std
    try:
        import ttkbootstrap as ttk
        _USING_TTKBOOTSTRAP = True
    except Exception:
        ttk = ttk_std
        _USING_TTKBOOTSTRAP = False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        _HAS_DND = True
    except Exception:
        DND_FILES = None
        TkinterDnD = None
        _HAS_DND = False
except Exception as exc:  # pragma: no cover - only triggered when Tk is missing
    raise RuntimeError(
        "Tkinter is not available. Install a Python build that includes Tk (Tcl/Tk runtime)."
    ) from exc


class _QueueWriter:
    def __init__(self, out_queue: queue.Queue) -> None:
        self._queue = out_queue
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def write(self, data: str) -> None:
        if not data:
            return
        cleaned = self._ansi_re.sub("", data)
        cleaned = cleaned.expandtabs(8)
        if cleaned:
            self._queue.put(cleaned)

    def flush(self) -> None:
        return None


class _GuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"IMU Video Sync {__version__}")
        self.root.geometry("784x688")
        self.root.minsize(720, 576)

        self.video_var = tk.StringVar()
        self.log_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.error_var = tk.StringVar(value="")
        self._theme_var = tk.StringVar(value="system")

        self.confidence_var = tk.StringVar(value="--")
        self.correlation_var = tk.StringVar(value="--")
        self.psr_var = tk.StringVar(value="--")
        self.signal_var = tk.StringVar(value="--")

        self.seconds_var = tk.StringVar(value="--")
        self.frames_var = tk.StringVar(value="--")
        self.timecode_var = tk.StringVar(value="--")
        self.project_var = tk.StringVar(value="--")

        self._queue: queue.Queue = queue.Queue()
        self._done_sentinel = object()
        self._running = False
        self._update_inflight = False
        self._update_url: str | None = None

        self._last_result: analysis.SyncResult | None = None
        self._last_error: Exception | None = None

        self._plot_available = False
        self._plot_canvas = None
        self._plot_axes = None
        self._plot_fig = None
        self._plot_placeholder = None
        self._plot_widget: ttk.Widget | None = None
        self._plot_min_height = 110
        self._corr_min_height = self._plot_min_height + 70
        self._theme_mode = "system"
        self._log_collapsed = True
        self._copy_buttons: list[ttk.Button] = []
        self._card_min_width = 160
        self._card_max_width = 260
        self._cards_wrap: ttk.Frame | None = None
        self._cards_frame: ttk.Frame | None = None

        self._init_fonts()
        self._build_menu()
        self._build_ui()
        self._set_window_icon()
        self.root.after(75, self._poll_output)
        self._schedule_update_check()
        self._sync_theme_mode()

    def _init_fonts(self) -> None:
        try:
            import tkinter.font as tkfont

            base = tkfont.nametofont("TkDefaultFont")
            self._font_label = base.copy()
            self._font_label.configure(size=max(9, int(base.cget("size"))))
            self._font_value = base.copy()
            self._font_value.configure(size=max(16, int(base.cget("size")) + 6), weight="bold")
            self._font_small = base.copy()
            self._font_small.configure(size=max(8, int(base.cget("size")) - 1))
            self._font_bold = base.copy()
            self._font_bold.configure(weight="bold")
        except Exception:
            self._font_label = None
            self._font_value = None
            self._font_small = None
            self._font_bold = None

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        view_menu = tk.Menu(menubar, tearoff=0)
        appearance_menu = tk.Menu(view_menu, tearoff=0)
        appearance_menu.add_radiobutton(
            label="System (default)",
            value="system",
            variable=self._theme_var,
            command=self._apply_theme_selection,
        )
        appearance_menu.add_radiobutton(
            label="Light",
            value="light",
            variable=self._theme_var,
            command=self._apply_theme_selection,
        )
        appearance_menu.add_radiobutton(
            label="Dark",
            value="dark",
            variable=self._theme_var,
            command=self._apply_theme_selection,
        )
        view_menu.add_cascade(label="Appearance", menu=appearance_menu)
        menubar.add_cascade(label="View", menu=view_menu)
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About IMU Video Sync...", command=self._show_about)
        about_menu.add_separator()
        about_menu.add_command(
            label="Check for updates", command=lambda: self._start_update_check(manual=True)
        )
        menubar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menubar)

    def _apply_theme_selection(self) -> None:
        if not _USING_TTKBOOTSTRAP:
            messagebox.showinfo(
                "Appearance",
                "Theme switching requires ttkbootstrap. The default theme will be used.",
            )
            return
        choice = self._theme_var.get()
        self._theme_mode = choice
        if choice == "light":
            theme = "lumen"
        elif choice == "dark":
            theme = "darkly"
        else:
            theme = choose_ttkbootstrap_theme()
        try:
            ttk.Style().theme_use(theme)
        except Exception:
            pass
        self._sync_theme_mode()
        self._update_plot_theme()

    def _sync_theme_mode(self) -> None:
        choice = self._theme_var.get()
        if choice in ("light", "dark"):
            self._theme_mode = choice
            return
        if not _USING_TTKBOOTSTRAP:
            self._theme_mode = "light"
            return
        try:
            active = ttk.Style().theme_use()
        except Exception:
            active = ""
        self._theme_mode = "dark" if active == "darkly" else "light"


    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=(12, 12, 12, 6))
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        row = 0
        self._build_file_panel(container, row)
        row += 1
        self._build_results_panel(container, row)
        row += 1
        self._build_analysis_tabs(container, row)
        row += 1
        self._build_update_bar(self.root, row=1)

        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        self.root.after(0, self._on_cards_wrap_resize)

    def _build_file_panel(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Labelframe(parent, text="File Inputs", padding=12)
        frame.grid(row=row, column=0, sticky="we", pady=(0, 10))

        ttk.Label(frame, text="Video File").grid(row=0, column=0, sticky="w")
        video_entry = ttk.Entry(frame, textvariable=self.video_var)
        video_entry.grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frame, text="Browse...", command=self._browse_video).grid(
            row=1, column=1, sticky="we"
        )

        ttk.Label(frame, text="Telemetry Log").grid(row=2, column=0, sticky="w", pady=(8, 0))
        log_entry = ttk.Entry(frame, textvariable=self.log_var)
        log_entry.grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frame, text="Browse...", command=self._browse_log).grid(
            row=3, column=1, sticky="we"
        )

        if _HAS_DND:
            self._register_drop_target(video_entry, self.video_var, (".mp4", ".mov", ".mkv"))
            self._register_drop_target(log_entry, self.log_var, (".csv", ".log", ".txt"))

        self.run_button = ttk.Button(frame, text="Analyze & Sync", command=self._start_run)
        self.run_button.grid(row=4, column=0, sticky="w", pady=(10, 0))

        self.status_label = ttk.Label(frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=1, sticky="e", pady=(10, 0))

        frame.columnconfigure(0, weight=1)

    def _build_results_panel(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Labelframe(parent, text="Alignment Results", padding=12)
        frame.grid(row=row, column=0, sticky="we", pady=(0, 10))

        self.error_label = tk.Label(
            frame,
            textvariable=self.error_var,
            fg="#B00020",
            justify="left",
            anchor="w",
        )
        self.error_label.grid(row=0, column=0, sticky="we", pady=(0, 6))
        self.error_label.grid_remove()

        cards_wrap = ttk.Frame(frame)
        cards_wrap.grid(row=1, column=0, sticky="we")
        cards_wrap.columnconfigure(0, weight=1)
        cards_wrap.columnconfigure(1, weight=0)
        cards_wrap.columnconfigure(2, weight=1)
        cards_wrap.rowconfigure(0, weight=1)

        cards = ttk.Frame(cards_wrap)
        cards.grid(row=0, column=1, sticky="we")
        cards.grid_propagate(False)
        cards.columnconfigure(0, weight=1, uniform="result_cards")
        cards.columnconfigure(1, weight=1, uniform="result_cards")
        cards.columnconfigure(2, weight=1, uniform="result_cards")
        cards.columnconfigure(3, weight=1, uniform="result_cards")
        cards.rowconfigure(0, weight=1)
        self._cards_wrap = cards_wrap
        self._cards_frame = cards
        cards_wrap.bind("<Configure>", self._on_cards_wrap_resize)

        self._copy_buttons = []

        self._seconds_card = self._build_offset_card(
            cards,
            title="Seconds Offset",
            unit="seconds",
            value_var=self.seconds_var,
            row=0,
            column=0,
        )
        self._frames_card = self._build_offset_card(
            cards,
            title="Frame Offset",
            unit="frames",
            value_var=self.frames_var,
            row=0,
            column=1,
        )
        self._timecode_card = self._build_offset_card(
            cards,
            title="Timecode Offset",
            unit="hh:mm:ss:ff",
            value_var=self.timecode_var,
            row=0,
            column=2,
        )
        self.project_title_var = tk.StringVar(value="Video Offset")
        self._project_card = self._build_offset_card(
            cards,
            title=self.project_title_var,
            unit="timeline position",
            value_var=self.project_var,
            row=0,
            column=3,
        )

        frame.columnconfigure(0, weight=1)

    def _build_offset_card(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        unit: str,
        value_var: tk.StringVar,
        row: int,
        column: int,
        sublabel_var: tk.StringVar | None = None,
    ) -> ttk.Frame:
        card = ttk.Frame(parent, padding=(6, 6), relief="ridge")
        card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        card.columnconfigure(0, weight=1)

        if isinstance(title, tk.StringVar):
            title_label = ttk.Label(card, textvariable=title, font=self._font_label)
        else:
            title_label = ttk.Label(card, text=title, font=self._font_label)
        title_label.grid(row=0, column=0, sticky="w")

        value_label = ttk.Label(card, textvariable=value_var, font=self._font_value)
        value_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        if sublabel_var is not None:
            sub_label = ttk.Label(card, textvariable=sublabel_var, font=self._font_small)
            sub_label.grid(row=2, column=0, sticky="w", pady=(2, 0))
            spacer_row = 3
            button_row = 4
        else:
            spacer_row = 2
            button_row = 3

        spacer = ttk.Frame(card)
        spacer.grid(row=spacer_row, column=0, sticky="nsew")
        card.rowconfigure(spacer_row, weight=1)

        copy_btn = ttk.Button(
            card,
            text="Copy",
            command=lambda v=value_var: self._copy_value(v.get()),
            padding=(6, 1),
        )
        copy_btn.grid(row=button_row, column=0, sticky="we", pady=(4, 0))
        self._copy_buttons.append(copy_btn)

        return card

    def _build_analysis_tabs(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1, minsize=self._corr_min_height)

        notebook = ttk.Notebook(frame)
        notebook.grid(row=0, column=0, sticky="nsew")

        self._corr_view = ttk.Frame(notebook)
        self._corr_view.columnconfigure(0, weight=1)
        self._corr_view.rowconfigure(0, weight=1, minsize=self._corr_min_height)

        self._signal_view = ttk.Frame(notebook)
        self._signal_view.columnconfigure(0, weight=1)
        self._signal_view.rowconfigure(0, weight=1)

        self._log_view = ttk.Frame(notebook)
        self._log_view.columnconfigure(0, weight=1)
        self._log_view.rowconfigure(0, weight=1)

        notebook.add(self._corr_view, text="Correlation")
        notebook.add(self._signal_view, text="Signal Candidates")
        notebook.add(self._log_view, text="Logs")
        notebook.select(self._corr_view)

        self._build_correlation_panel(self._corr_view, row=0)
        self._build_signal_panel(self._signal_view, row=0)
        self._build_log_panel(self._log_view, row=0, tabbed=True)

    def _build_correlation_panel(self, parent: ttk.Frame, row: int, column: int = 0) -> None:
        frame = ttk.Frame(parent, padding=(8, 2, 8, 6))
        frame.grid(row=row, column=column, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        diag = ttk.Frame(frame)
        diag.grid(row=0, column=0, sticky="we", pady=(0, 6))
        diag.columnconfigure(0, weight=0)
        diag.columnconfigure(1, weight=0)
        diag.columnconfigure(2, weight=0)
        diag.columnconfigure(3, weight=0)
        diag.columnconfigure(4, weight=0)
        diag.columnconfigure(5, weight=1)

        ttk.Label(diag, text="Confidence:").grid(row=0, column=0, sticky="w")
        self.confidence_value = ttk.Label(diag, textvariable=self.confidence_var)
        self.confidence_value.grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(diag, text="Correlation:").grid(row=0, column=2, sticky="w")
        ttk.Label(diag, textvariable=self.correlation_var).grid(
            row=0, column=3, sticky="w", padx=(4, 12)
        )

        ttk.Label(diag, text="PSR:").grid(row=0, column=4, sticky="w")
        ttk.Label(diag, textvariable=self.psr_var).grid(
            row=0, column=5, sticky="w", padx=(4, 12)
        )

        ttk.Label(diag, text="Signal:").grid(row=0, column=6, sticky="w")
        ttk.Label(diag, textvariable=self.signal_var).grid(
            row=0, column=7, sticky="w", padx=(4, 0)
        )

        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.configure(height=self._plot_min_height)
        plot_frame.grid_propagate(False)

        self._init_plot(plot_frame)

    def _init_plot(self, parent: ttk.Frame) -> None:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(6, 4), dpi=100)
            fig.patch.set_alpha(0.0)
            fig.patch.set_facecolor("none")
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor("none")
            fig.subplots_adjust(left=0.02, right=0.995, top=0.995, bottom=0.12)

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, sticky="nsew")

            self._plot_available = True
            self._plot_canvas = canvas
            self._plot_axes = ax1
            self._plot_fig = fig
            self._plot_widget = canvas_widget
            canvas_widget.grid_remove()
        except Exception:
            self._plot_available = False
            self._plot_placeholder = ttk.Label(
                parent,
                text="Plotting requires matplotlib. Install it to view the aligned signals plot.",
                anchor="center",
                justify="center",
            )
            self._plot_placeholder.grid(row=0, column=0, sticky="nsew")

    def _build_signal_panel(self, parent: ttk.Frame, row: int, column: int = 0) -> None:
        frame = ttk.Frame(parent, padding=(8, 8, 8, 6))
        frame.grid(row=row, column=column, sticky="nsew", pady=(10, 0))
        frame.columnconfigure(0, weight=1)

        columns = ("signal", "correlation", "psr", "score")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=4)
        tree.heading("signal", text="Signal")
        tree.heading("correlation", text="Correlation")
        tree.heading("psr", text="PSR")
        tree.heading("score", text="Score")
        tree.column("signal", width=160, anchor="w")
        tree.column("correlation", width=100, anchor="center")
        tree.column("psr", width=80, anchor="center")
        tree.column("score", width=80, anchor="center")

        if self._font_bold is not None:
            tree.tag_configure("selected", font=self._font_bold)

        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        frame.rowconfigure(0, weight=1)

        self.signal_tree = tree

    def _build_log_panel(self, parent: ttk.Frame, row: int, tabbed: bool = False) -> None:
        frame_class = ttk.Frame if tabbed else ttk.Labelframe
        frame = frame_class(parent, padding=(8, 8, 8, 8))
        if not tabbed:
            frame.configure(text="Log Output")
        frame.grid(row=row, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="we")
        header.columnconfigure(0, weight=1)

        if tabbed:
            self.log_toggle_button = None
        else:
            self.log_toggle_button = ttk.Button(
                header, text="Expand", command=self._toggle_log
            )
            self.log_toggle_button.grid(row=0, column=0, sticky="e")

        self.log_body = ttk.Frame(frame)
        self.log_body.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_body.columnconfigure(0, weight=1)
        self.log_body.rowconfigure(0, weight=1)

        font_name = "Consolas" if sys.platform.startswith("win") else "TkFixedFont"
        self.output = scrolledtext.ScrolledText(
            self.log_body,
            height=10,
            wrap="none",
            state="disabled",
            font=(font_name, 10),
        )
        self.output.grid(row=0, column=0, sticky="nsew")
        hbar = ttk.Scrollbar(self.log_body, orient="horizontal", command=self.output.xview)
        hbar.grid(row=1, column=0, sticky="we")
        self.output.configure(xscrollcommand=hbar.set)

        try:
            import tkinter.font as tkfont

            font = tkfont.nametofont("TkFixedFont")
            tab_px = font.measure(" " * 8)
            self.output.configure(tabs=(tab_px,))
        except Exception:
            pass

        if tabbed:
            self._set_log_collapsed(False)
        else:
            self._set_log_collapsed(True)

    def _build_update_bar(self, parent: ttk.Frame, row: int) -> None:
        self._update_bar = ttk.Frame(parent, padding=(12, 2, 12, 6))
        self._update_label = ttk.Label(self._update_bar, text="", justify="left", anchor="w")
        self._update_label.grid(row=0, column=0, sticky="w")
        self._update_button = ttk.Button(
            self._update_bar, text="Download", command=self._open_update_url, padding=(6, 2)
        )
        self._update_button.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self._update_bar.columnconfigure(0, weight=1)
        self._update_bar.grid(row=row, column=0, sticky="we")
        self._update_bar.grid_remove()
        self._update_bar.bind("<Configure>", self._refresh_update_wrap)


    def _toggle_log(self) -> None:
        self._set_log_collapsed(not self._log_collapsed)

    def _set_log_collapsed(self, collapsed: bool) -> None:
        self._log_collapsed = collapsed
        if collapsed:
            if self.log_body.winfo_ismapped():
                self.log_body.grid_remove()
            if self.log_toggle_button is not None:
                self.log_toggle_button.configure(text="Expand")
        else:
            if not self.log_body.winfo_ismapped():
                self.log_body.grid()
            if self.log_toggle_button is not None:
                self.log_toggle_button.configure(text="Collapse")

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video (MP4)",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
        )
        if path:
            self.video_var.set(path)

    def _browse_log(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Log (CSV)",
            filetypes=[("CSV Log", "*.csv"), ("All Files", "*.*")],
        )
        if path:
            self.log_var.set(path)

    def _set_window_icon(self) -> None:
        candidates = []
        if getattr(sys, "frozen", False):
            base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            candidates.append(base / "assets" / "icon" / "IMUVideoSync.png")
            candidates.append(base / "assets" / "icon" / "IMUVideoSync.ico")
            candidates.append(Path(sys.executable).with_name("IMUVideoSync.png"))
            candidates.append(Path(sys.executable).with_name("IMUVideoSync.ico"))
        else:
            base = Path(__file__).resolve().parents[2]
            candidates.append(base / "assets" / "icon" / "IMUVideoSync.png")
            candidates.append(base / "assets" / "icon" / "IMUVideoSync.ico")

        png_path = next((p for p in candidates if p.suffix.lower() == ".png" and p.exists()), None)
        ico_path = next((p for p in candidates if p.suffix.lower() == ".ico" and p.exists()), None)

        try:
            if ico_path and sys.platform.startswith("win"):
                self.root.iconbitmap(str(ico_path))
            elif png_path:
                icon = tk.PhotoImage(file=str(png_path))
                self.root.iconphoto(True, icon)
                self.root._icon_ref = icon  # Prevent garbage collection.
        except Exception:
            pass

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.run_button.configure(state="disabled" if running else "normal")
        self.status_var.set("Running..." if running else "Ready")

    def _show_about(self) -> None:
        message = "\n".join(
            [
                "IMUVideoSync by Brandon Strohmeyer",
                f"Version: {__version__}",
            ]
        )
        messagebox.showinfo("About IMU Video Sync", message)

    def _show_update_bar(self, message: str, url: str) -> None:
        self._update_url = url
        self._update_label.configure(text=message)
        if not self._update_bar.winfo_ismapped():
            self._update_bar.grid()
        self._refresh_update_wrap()

    def _hide_update_bar(self) -> None:
        self._update_url = None
        if self._update_bar.winfo_ismapped():
            self._update_bar.grid_remove()

    def _refresh_update_wrap(self, event: tk.Event | None = None) -> None:
        if not self._update_bar.winfo_ismapped():
            return
        self.root.update_idletasks()
        bar_width = self._update_bar.winfo_width() or self.root.winfo_width()
        button_width = self._update_button.winfo_reqwidth()
        wrap = max(200, bar_width - button_width - 32)
        self._update_label.configure(wraplength=wrap)

    def _open_update_url(self) -> None:
        if self._update_url:
            webbrowser.open(self._update_url)

    def _schedule_update_check(self) -> None:
        if update_check.is_disabled():
            return
        self.root.after(300, lambda: self._start_update_check(manual=False))

    def _start_update_check(self, *, manual: bool) -> None:
        if update_check.is_disabled():
            if manual:
                messagebox.showinfo(
                    "Update Check Disabled",
                    "Update checks are disabled via IMU_VIDEO_SYNC_DISABLE_UPDATE_CHECK.",
                )
            return
        if self._update_inflight:
            if manual:
                messagebox.showinfo("Update Check", "An update check is already running.")
            return

        self._update_inflight = True

        def worker() -> None:
            result = update_check.check_for_updates(include_prereleases=True, timeout_s=2.5)
            self.root.after(0, lambda: self._handle_update_result(result, manual))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_update_result(
        self, result: update_check.UpdateResult | None, manual: bool
    ) -> None:
        self._update_inflight = False
        if result is None:
            if manual:
                messagebox.showinfo("Update Check", "Unable to check for updates right now.")
            return
        if result.update_available:
            banner = (
                f"Update available: {result.latest_version} "
                f"(current {result.current_version})"
            )
            self._show_update_bar(banner, result.release_url)
            if manual:
                message = (
                    f"A new version is available: {result.latest_version}\n"
                    f"You are running {result.current_version}.\n\n"
                    "Open the download page--"
                )
                if messagebox.askyesno("Update Available", message):
                    webbrowser.open(result.release_url)
            return
        if manual:
            self._hide_update_bar()
            messagebox.showinfo(
                "Up to Date", f"You're up to date ({result.current_version})."
            )

    def _append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _validate_paths(self) -> tuple[Path, Path] | None:
        video_str = self.video_var.get().strip().strip("\"'")
        log_str = self.log_var.get().strip().strip("\"'")

        if not video_str or not log_str:
            self._set_error(
                "Please select both a video and a log file.",
                [],
            )
            return None
        video = Path(video_str)
        log = Path(log_str)
        if not video.exists():
            self._set_error("Selected video file was not found.", [str(video)])
            return None
        if not log.exists():
            self._set_error("Selected log file was not found.", [str(log)])
            return None
        return video, log

    def _register_drop_target(
        self,
        widget: ttk.Widget,
        target_var: tk.StringVar,
        allowed_exts: tuple[str, ...] | None = None,
    ) -> None:
        try:
            widget.drop_target_register(DND_FILES)  # type: ignore[union-attr]
            widget.dnd_bind(
                "<<Drop>>",
                lambda event, var=target_var, exts=allowed_exts: self._handle_drop(
                    event, var, exts
                ),
            )
        except Exception:
            return

    def _handle_drop(
        self,
        event: tk.Event,
        target_var: tk.StringVar,
        allowed_exts: tuple[str, ...] | None,
    ) -> None:
        data = getattr(event, "data", "")
        if not data:
            return
        try:
            paths = self.root.tk.splitlist(data)
        except Exception:
            paths = [data]
        cleaned: list[str] = []
        for item in paths:
            item = item.strip().strip("\"'")
            if item.startswith("{") and item.endswith("}"):
                item = item[1:-1]
            if item:
                cleaned.append(item)
        if not cleaned:
            return
        if allowed_exts:
            for item in cleaned:
                if Path(item).suffix.lower() in allowed_exts:
                    target_var.set(item)
                    return
        target_var.set(cleaned[0])

    def _start_run(self) -> None:
        if self._running:
            return
        resolved = self._validate_paths()
        if resolved is None:
            return
        video, log = resolved

        self._clear_error()
        self._clear_results()
        self._set_running(True)
        self._append_output("Starting...\n")

        def worker() -> None:
            writer = _QueueWriter(self._queue)
            self._last_result = None
            self._last_error = None

            def emit(line: str) -> None:
                if not line.endswith("\n"):
                    line = f"{line}\n"
                writer.write(line)

            options = analysis.SyncOptions()
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    result = analysis.run_sync(video, log, options=options, emit=emit)
                self._last_result = result
            except Exception as exc:
                self._last_error = exc
                writer.write("\nUnexpected error:\n")
                writer.write(traceback.format_exc())
            finally:
                self._queue.put(self._done_sentinel)

        threading.Thread(target=worker, daemon=True).start()

    def _poll_output(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                if item is self._done_sentinel:
                    self._set_running(False)
                    if self._last_result is not None:
                        self._apply_result(self._last_result)
                        if self._last_result.post_summary_warnings:
                            for warning in self._last_result.post_summary_warnings:
                                self._append_output(f"{warning}\n")
                    else:
                        self._handle_error(self._last_error)
                else:
                    self._append_output(str(item))
        except queue.Empty:
            pass
        self.root.after(75, self._poll_output)

    def _apply_result(self, result: analysis.SyncResult) -> None:
        self.confidence_var.set(result.diagnostics.confidence_label)
        self.correlation_var.set(f"{result.diagnostics.correlation_peak:.3f}")
        self.psr_var.set(f"{result.diagnostics.psr:.3f}")
        self.signal_var.set(str(result.diagnostics.signal))
        self._apply_confidence_style(result.diagnostics.confidence_label)

        self.seconds_var.set(result.offsets.lag_seconds_str)
        self.frames_var.set(result.offsets.lag_frames or "n/a")
        self.timecode_var.set(result.offsets.timecode_offset or "n/a")
        self.project_var.set(result.offsets.project_position)
        self.project_title_var.set(
            "Video Offset" if result.offsets.is_video_offset else "Data Offset"
        )

        self._set_copy_state(True)

        self._populate_signals(result)
        self._update_plot(result)

    def _apply_confidence_style(self, rating: str) -> None:
        if _USING_TTKBOOTSTRAP:
            style = "success" if rating == "High" else "warning" if rating == "Medium" else "danger"
            try:
                self.confidence_value.configure(bootstyle=style)
            except Exception:
                pass
        else:
            color = "#2E7D32" if rating == "High" else "#B26A00" if rating == "Medium" else "#B00020"
            try:
                self.confidence_value.configure(foreground=color)
            except Exception:
                pass

    def _populate_signals(self, result: analysis.SyncResult) -> None:
        self.signal_tree.delete(*self.signal_tree.get_children())
        for idx, cand in enumerate(result.candidates):
            tags = ("selected",) if idx == result.selected_index else ()
            self.signal_tree.insert(
                "",
                "end",
                values=(
                    cand.signal,
                    f"{cand.peak:.3f}",
                    f"{cand.psr:.3f}",
                    f"{cand.score:.3f}",
                ),
                tags=tags,
            )

    def _update_plot(self, result: analysis.SyncResult) -> None:
        if not self._plot_available or self._plot_canvas is None or self._plot_axes is None:
            return
        ax1 = self._plot_axes
        ax1.clear()
        ax1.axis("on")
        if self._plot_widget is not None and not self._plot_widget.winfo_ismapped():
            self._plot_widget.grid()

        log_time = result.plot.time_s
        video_time = result.plot.video_time_s
        log_origin = float(log_time[0]) if log_time.size else 0.0
        time_rel = log_time - log_origin
        video_time_aligned = video_time + float(result.plot.lag_seconds)
        video_time_rel = video_time_aligned - log_origin
        log_y = result.plot.log_y
        video_y = result.plot.video_y
        try:
            import numpy as np

            log_arr = np.asarray(log_y, dtype=float)
            video_arr = np.asarray(video_y, dtype=float)

            def smooth_series(values: np.ndarray, window: int = 9) -> np.ndarray:
                if values.size == 0 or window <= 1:
                    return values
                kernel = np.ones(window, dtype=float) / float(window)
                return np.convolve(values, kernel, mode="same")

            combined = np.concatenate([np.abs(log_arr), np.abs(video_arr)])
            scale = float(np.percentile(combined, 90)) if combined.size else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            log_y = np.arcsinh(log_arr / scale)
            video_y = np.arcsinh(video_arr / scale)

            log_y = smooth_series(log_y, window=9)
            video_y = smooth_series(video_y, window=9)
        except Exception:
            log_y = result.plot.log_y
            video_y = result.plot.video_y

        ax1.plot(
            time_rel,
            log_y,
            label="Telemetry",
            linewidth=0.8,
            color="#F2994A",
        )
        ax1.plot(
            video_time_rel,
            video_y,
            label="Video IMU",
            linewidth=0.8,
            color="#2D9CDB",
            alpha=0.85,
        )
        ax1.set_xlabel("Time (s)", labelpad=6, loc="right")
        ax1.set_yticks([])
        ax1.set_ylabel("")
        ax1.margins(x=0, y=0.15)
        ax1.legend(
            loc="lower right",
            ncol=2,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
        self._update_plot_theme()
        if self._plot_fig is not None:
            self._plot_fig.subplots_adjust(left=0.02, right=0.995, top=0.995, bottom=0.12)

        try:
            self._plot_canvas.draw()
        except Exception:
            pass

    def _update_plot_theme(self) -> None:
        if not self._plot_available or self._plot_axes is None:
            return
        is_dark = self._theme_mode == "dark"
        text_color = "#f5f5f5" if is_dark else "#1a1a1a"
        ax1 = self._plot_axes
        try:
            ax1.tick_params(colors=text_color)
            ax1.xaxis.label.set_color(text_color)
            ax1.yaxis.label.set_color(text_color)
            for spine in ax1.spines.values():
                spine.set_color(text_color)
            ax1.spines["left"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            legend = ax1.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(text_color)
        except Exception:
            return

    def _handle_error(self, exc: Exception | None) -> None:
        if exc is None:
            self._set_error("Analysis failed. See log output for details.", [])
            return
        summary, suggestions = self._friendly_error(exc)
        self._set_error(summary, suggestions)

    def _friendly_error(self, exc: Exception) -> tuple[str, list[str]]:
        message = str(exc)
        lower = message.lower()
        if "no compatible signals" in lower:
            return (
                "No compatible signals were found between the selected video and log.",
                [
                    "Verify the log contains motion-related channels.",
                    "Verify the video contains IMU telemetry.",
                    "Check that both files belong to the same session.",
                ],
            )
        if "video not found" in lower:
            return ("The selected video file could not be found.", [message])
        if "log not found" in lower:
            return ("The selected log file could not be found.", [message])
        if "auto-window selection failed" in lower:
            return (
                "No suitable analysis window was found for these files.",
                [
                    "Try a different session or shorter clip.",
                    "Verify that both files contain movement.",
                ],
            )
        return ("Analysis failed. See log output for details.", [])

    def _set_error(self, summary: str, suggestions: list[str]) -> None:
        lines = [summary]
        for item in suggestions:
            lines.append(f"- {item}")
        self.error_var.set("\n".join(lines))
        if not self.error_label.winfo_ismapped():
            self.error_label.grid()

    def _clear_error(self) -> None:
        self.error_var.set("")
        if self.error_label.winfo_ismapped():
            self.error_label.grid_remove()

    def _clear_results(self) -> None:
        self.confidence_var.set("--")
        self.correlation_var.set("--")
        self.psr_var.set("--")
        self.signal_var.set("--")
        self.seconds_var.set("--")
        self.frames_var.set("--")
        self.timecode_var.set("--")
        self.project_var.set("--")
        self.project_title_var.set("Video Offset")
        self._set_copy_state(False)
        self.signal_tree.delete(*self.signal_tree.get_children())
        if self._plot_available and self._plot_axes is not None:
            ax1 = self._plot_axes
            ax1.clear()
            ax1.axis("off")
            if self._plot_widget is not None and self._plot_widget.winfo_ismapped():
                self._plot_widget.grid_remove()
            try:
                self._plot_canvas.draw()
            except Exception:
                pass

    def _set_copy_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in self._copy_buttons:
            btn.configure(state=state)

    def _on_cards_wrap_resize(self, event: tk.Event | None = None) -> None:
        if self._cards_wrap is None or self._cards_frame is None:
            return
        width = self._cards_wrap.winfo_width()
        if width <= 1:
            width = self._card_min_width * 4
        per_card = width / 4.0
        per_card = max(self._card_min_width, min(per_card, self._card_max_width))
        total = int(per_card * 4)
        req_height = self._cards_frame.winfo_reqheight()
        if req_height <= 1:
            req_height = 120
        self._cards_frame.configure(width=total, height=req_height)
        self._cards_wrap.columnconfigure(1, minsize=total)
        for idx in range(4):
            self._cards_frame.columnconfigure(
                idx, weight=1, uniform="result_cards", minsize=int(per_card)
            )

    def _copy_value(self, value: str) -> None:
        if not value or value.strip() in ("--", "n/a", "n/a (fps unknown)"):
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        if not self._running:
            self.status_var.set("Copied")
            self.root.after(1500, lambda: self.status_var.set("Ready"))



def main() -> None:
    if _HAS_DND:
        root = TkinterDnD.Tk()  # type: ignore[assignment,call-arg]
        if _USING_TTKBOOTSTRAP:
            try:
                ttk.Style().theme_use(choose_ttkbootstrap_theme())
            except Exception:
                pass
    elif _USING_TTKBOOTSTRAP:
        root = ttk.Window(themename=choose_ttkbootstrap_theme())
    else:
        root = tk.Tk()
    _GuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

