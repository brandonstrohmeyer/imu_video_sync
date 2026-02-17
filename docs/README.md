# Docs Index

These notes explain how `imu-video-sync` works internally at a level a 2nd year CS student should be comfortable with.

- `overview.md`: high-level architecture, what is compared, and why it works.
- `data_sources.md`: how GoPro MP4 and AiM CSV are parsed.
- `signal_processing.md`: resampling, filtering, normalization.
- `correlation.md`: lag estimation, FFT usage, window scanning, and confidence metrics.
- `cli_flow.md`: CLI behavior, defaults, and how arguments change behavior.
