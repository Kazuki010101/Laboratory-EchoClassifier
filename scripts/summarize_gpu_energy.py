#!/usr/bin/env python3

import argparse
import csv
import json
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


MONITOR_RUNNING = True


def stop_monitoring(signum, frame):
    del signum, frame

    global MONITOR_RUNNING
    MONITOR_RUNNING = False


def parse_number(value):
    value = value.strip()

    if not value or value.lower() in {
        "n/a",
        "[n/a]",
        "nan",
        "not supported",
    }:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def query_gpu(gpu_index):
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu="
        "index,power.draw,utilization.gpu,"
        "memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    line = result.stdout.strip().splitlines()[0]
    values = next(csv.reader([line]))

    if len(values) != 5:
        raise RuntimeError(
            f"Unexpected nvidia-smi output: {line}"
        )

    return {
        "gpu_index": int(values[0].strip()),
        "power_w": parse_number(values[1]),
        "utilization_percent": parse_number(values[2]),
        "memory_used_mib": parse_number(values[3]),
        "temperature_c": parse_number(values[4]),
    }


def monitor(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal.signal(signal.SIGTERM, stop_monitoring)
    signal.signal(signal.SIGINT, stop_monitoring)

    interval_seconds = args.interval_ms / 1000.0

    fieldnames = [
        "epoch_seconds",
        "utc_time",
        "gpu_index",
        "power_w",
        "utilization_percent",
        "memory_used_mib",
        "temperature_c",
    ]

    consecutive_errors = 0

    with output_path.open(
        "w",
        newline="",
        buffering=1,
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        while MONITOR_RUNNING:
            loop_start = time.monotonic()

            try:
                metrics = query_gpu(args.gpu_index)
                now = time.time()

                writer.writerow(
                    {
                        "epoch_seconds": f"{now:.9f}",
                        "utc_time": datetime.fromtimestamp(
                            now,
                            tz=timezone.utc,
                        ).isoformat(),
                        **metrics,
                    }
                )

                output_file.flush()
                consecutive_errors = 0

            except Exception as error:
                consecutive_errors += 1
                print(
                    f"[GPU MONITOR WARNING] {error}",
                    flush=True,
                )

                if consecutive_errors >= 10:
                    raise RuntimeError(
                        "GPU monitoring failed 10 times "
                        "consecutively."
                    ) from error

            elapsed = time.monotonic() - loop_start
            remaining = interval_seconds - elapsed

            if remaining > 0:
                time.sleep(remaining)

    print(
        f"[GPU MONITOR] Saved: {output_path}",
        flush=True,
    )


def read_rows(input_path, start_epoch, end_epoch):
    rows = []

    with Path(input_path).open(newline="") as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            epoch_seconds = parse_number(
                row.get("epoch_seconds", "")
            )
            power_w = parse_number(
                row.get("power_w", "")
            )

            if epoch_seconds is None or power_w is None:
                continue

            if (
                start_epoch is not None
                and epoch_seconds < start_epoch
            ):
                continue

            if (
                end_epoch is not None
                and epoch_seconds > end_epoch
            ):
                continue

            rows.append(
                {
                    "epoch_seconds": epoch_seconds,
                    "utc_time": row.get("utc_time"),
                    "gpu_index": int(
                        float(row.get("gpu_index", 0))
                    ),
                    "power_w": power_w,
                    "utilization_percent": parse_number(
                        row.get(
                            "utilization_percent",
                            "",
                        )
                    ),
                    "memory_used_mib": parse_number(
                        row.get(
                            "memory_used_mib",
                            "",
                        )
                    ),
                    "temperature_c": parse_number(
                        row.get(
                            "temperature_c",
                            "",
                        )
                    ),
                }
            )

    rows.sort(
        key=lambda row: row["epoch_seconds"]
    )

    return rows


def numeric_values(rows, key):
    return [
        row[key]
        for row in rows
        if row[key] is not None
    ]


def summarize(args):
    rows = read_rows(
        input_path=args.input,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
    )

    if len(rows) < 2:
        raise RuntimeError(
            "At least two valid GPU power samples "
            "are required."
        )

    energy_joules = 0.0
    intervals = []

    for previous, current in zip(
        rows[:-1],
        rows[1:],
    ):
        duration = (
            current["epoch_seconds"]
            - previous["epoch_seconds"]
        )

        if duration <= 0:
            continue

        average_interval_power = (
            previous["power_w"]
            + current["power_w"]
        ) / 2.0

        energy_joules += (
            average_interval_power * duration
        )
        intervals.append(duration)

    monitored_duration = (
        rows[-1]["epoch_seconds"]
        - rows[0]["epoch_seconds"]
    )

    energy_wh = energy_joules / 3600.0
    energy_kwh = energy_wh / 1000.0

    power_values = numeric_values(
        rows,
        "power_w",
    )
    utilization_values = numeric_values(
        rows,
        "utilization_percent",
    )
    memory_values = numeric_values(
        rows,
        "memory_used_mib",
    )
    temperature_values = numeric_values(
        rows,
        "temperature_c",
    )

    time_weighted_average_power = (
        energy_joules / monitored_duration
        if monitored_duration > 0
        else None
    )

    wall_duration = None

    if (
        args.start_epoch is not None
        and args.end_epoch is not None
    ):
        wall_duration = (
            args.end_epoch - args.start_epoch
        )

    summary = {
        "gpu_index": rows[0]["gpu_index"],
        "power_samples": len(rows),
        "monitor_start_utc": rows[0]["utc_time"],
        "monitor_end_utc": rows[-1]["utc_time"],
        "monitored_duration_seconds": (
            monitored_duration
        ),
        "wall_duration_seconds": wall_duration,
        "sampling_interval_seconds_median": (
            statistics.median(intervals)
            if intervals
            else None
        ),
        "energy_joules": energy_joules,
        "energy_wh": energy_wh,
        "energy_kwh": energy_kwh,
        "average_power_w": (
            time_weighted_average_power
        ),
        "maximum_power_w": max(power_values),
        "minimum_power_w": min(power_values),
        "average_gpu_utilization_percent": (
            statistics.fmean(utilization_values)
            if utilization_values
            else None
        ),
        "maximum_gpu_utilization_percent": (
            max(utilization_values)
            if utilization_values
            else None
        ),
        "average_memory_used_mib": (
            statistics.fmean(memory_values)
            if memory_values
            else None
        ),
        "maximum_memory_used_mib": (
            max(memory_values)
            if memory_values
            else None
        ),
        "average_temperature_c": (
            statistics.fmean(temperature_values)
            if temperature_values
            else None
        ),
        "maximum_temperature_c": (
            max(temperature_values)
            if temperature_values
            else None
        ),
        "epochs": args.epochs,
        "energy_per_epoch_wh": (
            energy_wh / args.epochs
            if args.epochs
            and args.epochs > 0
            else None
        ),
        "training_exit_code": args.exit_code,
        "measurement_method": (
            "nvidia-smi power.draw sampled and "
            "integrated with the trapezoidal rule"
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
        )

    print("========================================")
    print("GPU ENERGY SUMMARY")
    print("========================================")
    print(
        f"Samples:       "
        f"{summary['power_samples']}"
    )
    print(
        f"Duration:      "
        f"{summary['monitored_duration_seconds']:.2f} s"
    )
    print(
        f"Average power: "
        f"{summary['average_power_w']:.2f} W"
    )
    print(
        f"Maximum power: "
        f"{summary['maximum_power_w']:.2f} W"
    )
    print(
        f"Energy:        "
        f"{summary['energy_wh']:.6f} Wh"
    )
    print(
        f"Energy:        "
        f"{summary['energy_kwh']:.9f} kWh"
    )

    if summary["energy_per_epoch_wh"] is not None:
        print(
            f"Per epoch:     "
            f"{summary['energy_per_epoch_wh']:.6f} Wh"
        )

    print(
        f"GPU util avg:  "
        f"{summary['average_gpu_utilization_percent']:.2f}%"
    )
    print(
        f"Memory max:    "
        f"{summary['maximum_memory_used_mib']:.2f} MiB"
    )
    print(
        f"Saved:         {output_path}"
    )
    print("========================================")


def build_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    monitor_parser = subparsers.add_parser(
        "monitor"
    )
    monitor_parser.add_argument(
        "--output",
        required=True,
    )
    monitor_parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
    )
    monitor_parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
    )
    monitor_parser.set_defaults(
        function=monitor
    )

    summarize_parser = subparsers.add_parser(
        "summarize"
    )
    summarize_parser.add_argument(
        "--input",
        required=True,
    )
    summarize_parser.add_argument(
        "--output",
        required=True,
    )
    summarize_parser.add_argument(
        "--start-epoch",
        type=float,
        default=None,
    )
    summarize_parser.add_argument(
        "--end-epoch",
        type=float,
        default=None,
    )
    summarize_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    summarize_parser.add_argument(
        "--exit-code",
        type=int,
        default=None,
    )
    summarize_parser.set_defaults(
        function=summarize
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()