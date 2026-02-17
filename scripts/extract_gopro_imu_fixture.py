from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from imu_video_sync.gopro_extract import extract_gopro_imu


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GoPro IMU metadata into a compressed test fixture.",
    )
    parser.add_argument("--mp4", required=True, help="Path to GoPro MP4 file")
    parser.add_argument(
        "--out",
        required=True,
        help="Output .npz path for the fixture (e.g., tests/fixtures/gopro_imu_full.npz)",
    )
    args = parser.parse_args()

    mp4_path = Path(args.mp4)
    if not mp4_path.exists():
        raise SystemExit(f"MP4 not found: {mp4_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imu = extract_gopro_imu(str(mp4_path))

    np.savez_compressed(
        out_path,
        gyro_time_s=imu.gyro_time_s,
        gyro=imu.gyro,
        accel_time_s=imu.accel_time_s,
        accel=imu.accel,
        backend=imu.backend,
        source=str(mp4_path),
    )

    gyro_shape = None if imu.gyro is None else imu.gyro.shape
    accel_shape = None if imu.accel is None else imu.accel.shape
    print(f"Wrote fixture: {out_path}")
    print(f"Gyro shape: {gyro_shape}")
    print(f"Accel shape: {accel_shape}")


if __name__ == "__main__":
    main()
