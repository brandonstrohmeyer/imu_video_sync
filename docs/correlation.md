# Correlation and Lag Estimation

The goal is to find the time shift that best aligns the two preprocessed signals.

## Cross-correlation
Cross-correlation measures similarity as one signal slides over the other. The peak of the correlation curve indicates the best alignment.

We compute correlation over a lag range (default +/-600 seconds).

## FFT in this project
We use FFT to compute cross-correlation efficiently. Conceptually it works like this:
1. Transform both signals to the frequency domain with FFT.
2. Multiply one FFT by the conjugate of the other.
3. Inverse FFT back to time domain.

That inverse FFT output is the correlation curve. The FFT itself is not the score, it is just a fast way to compute the curve.

## Window scanning and consensus
The tool scans many fixed-length windows in the log data (default window is 6 minutes). Each window is compared against the full video signal within the allowed lag range. This is repeated every `--window-step` seconds (default 20 seconds).

From those windows, the tool keeps the top 40 percent by score (minimum 5 windows). It then computes a weighted median of their lag estimates to produce the final lag.

## Window count estimate
If the log duration is `T`, window length is `W`, and step is `S`, then the number of windows is approximately:
`floor((T - W) / S) + 1`

Example: 20 minutes of data, 6 minute window, 20 second step:
`(1200 - 360) / 20 + 1 = 43` windows.

## Lag sign convention
`lag_seconds > 0` means video telemetry occurs later than log data (log leads). To align the log to video time, add `+lag_seconds` to log timestamps.

## Confidence metrics
- Correlation peak: the max correlation value (higher is better).
- PSR (peak-to-sidelobe ratio): main peak divided by the next-best local peak (higher is better).
- Stability (stddev s): how much the lag varies across top windows (lower is better).
- Drift (s/s): trend in lag over time (close to 0 is best).
- Confidence rating: a High/Medium/Low label plus a 0-100 score derived from peak, PSR, and stability.

Notes:
- Stability may be reported as `nan` if the aligned video window does not fully fit in the video timeline (for example, when the window equals the clip length).
