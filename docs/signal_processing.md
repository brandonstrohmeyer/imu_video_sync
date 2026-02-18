# Signal Processing

Once both signals are loaded, we need them on the same time grid and in a comparable format.

## Resampling
- The two sources have different sample rates.
- We resample both signals to a fixed rate (default 50 Hz).
- This is done by linear interpolation on a uniform time grid.
- We only keep the overlapping time range to avoid extrapolation.

## Filtering
We use a band-pass filter (default 0.2-8 Hz).
The high-pass removes slow drift and bias.
The low-pass removes high-frequency noise.

This keeps the signal focused on human-scale vehicle motion.

## Normalization
After filtering we subtract the mean and divide by the standard deviation (z-score). This makes different sensors comparable even if their units differ.

## Signal choices
- `gyroMag = sqrt(gx^2 + gy^2 + gz^2)` is the default because it is orientation-agnostic.
- Other signals (`yawRate`, `latAcc`, etc.) can be used if they exist in the data.

## FFT in this project
We do not compare signals in the frequency domain. We compare them in the time domain. FFT is only used as a speed trick to compute cross-correlation faster.
