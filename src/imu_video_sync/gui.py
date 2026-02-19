from __future__ import annotations

import contextlib
import queue
import re
import sys
import threading
import traceback
from pathlib import Path

from . import __version__
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
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
        self.root.geometry("800x520")

        self.video_var = tk.StringVar()
        self.log_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self._queue: queue.Queue = queue.Queue()
        self._done_sentinel = object()
        self._running = False

        self._build_ui()
        self._set_window_icon()
        self.root.after(75, self._poll_output)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Video (MP4)").grid(row=0, column=0, sticky="w")
        video_entry = tk.Entry(frame, textvariable=self.video_var)
        video_entry.grid(row=1, column=0, sticky="we", padx=(0, 8))
        tk.Button(frame, text="Browse...", command=self._browse_video).grid(
            row=1, column=1, sticky="we"
        )

        tk.Label(frame, text="Log (CSV)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        log_entry = tk.Entry(frame, textvariable=self.log_var)
        log_entry.grid(row=3, column=0, sticky="we", padx=(0, 8))
        tk.Button(frame, text="Browse...", command=self._browse_log).grid(
            row=3, column=1, sticky="we"
        )

        self.run_button = tk.Button(frame, text="Generate Offset", command=self._start_run)
        self.run_button.grid(row=4, column=0, sticky="w", pady=(12, 0))

        tk.Label(frame, textvariable=self.status_var).grid(row=4, column=1, sticky="e")

        font_name = "Consolas" if sys.platform.startswith("win") else "TkFixedFont"
        self.output = scrolledtext.ScrolledText(
            frame, height=18, wrap="none", state="disabled", font=(font_name, 10)
        )
        self.output.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        hbar = tk.Scrollbar(frame, orient="horizontal", command=self.output.xview)
        hbar.grid(row=6, column=0, columnspan=2, sticky="we")
        self.output.configure(xscrollcommand=hbar.set)
        try:
            import tkinter.font as tkfont

            font = tkfont.nametofont("TkFixedFont")
            tab_px = font.measure(" " * 8)
            self.output.configure(tabs=(tab_px,))
        except Exception:
            pass

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

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

    def _append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _validate_paths(self) -> tuple[Path, Path] | None:
        video_str = self.video_var.get().strip()
        log_str = self.log_var.get().strip()

        if not video_str or not log_str:
            messagebox.showerror("Missing inputs", "Please select both a video and a log file.")
            return None
        video = Path(video_str)
        log = Path(log_str)
        if not video.exists():
            messagebox.showerror("Video not found", f"Video not found:\n{video}")
            return None
        if not log.exists():
            messagebox.showerror("Log not found", f"Log not found:\n{log}")
            return None
        return video, log

    def _start_run(self) -> None:
        if self._running:
            return
        resolved = self._validate_paths()
        if resolved is None:
            return
        video, log = resolved

        self._set_running(True)
        self._append_output("Starting...\n")

        def worker() -> None:
            from .cli import main as cli_main

            writer = _QueueWriter(self._queue)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    cli_main(["--video", str(video), "--log", str(log)])
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    writer.write(f"\nExited with code {exc.code}\n")
            except Exception:
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
                else:
                    self._append_output(str(item))
        except queue.Empty:
            pass
        self.root.after(75, self._poll_output)


def main() -> None:
    root = tk.Tk()
    _GuiApp(root)
    root.mainloop()
