"""Unit-explicit survival summaries for reusable artifact inputs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from statomix.analytics.datatypes.survival import SingleClassSurv
from statomix.core.artifacts import canonical_json, digest_json, safe_relative
from statomix.storage.artifacts import artifact_lock
from statomix.storage.hashing import sha256_file

from .artifact_inputs import load_binding


def create_artifact_survival_summary(analyzer, bundle):
    reference, state, _, binding = load_binding(analyzer, bundle)
    config_path = bundle["config"]["path"]
    destination = config_path / "surv"
    manifest_path = destination / "report_manifest.json"
    signature = digest_json(binding)
    with artifact_lock(config_path):
        if destination.exists():
            if not manifest_path.exists():
                raise FileExistsError(
                    "Incomplete or legacy survival report exists; choose a new Analyzer configuration."
                )
            report = json.loads(manifest_path.read_text())
            report_id = report.pop("report_id", None)
            if report_id != digest_json(report):
                raise ValueError("Survival report manifest was modified.")
            if report["binding_sha256"] != signature:
                raise ValueError(
                    "Cached survival summary has different inputs/options."
                )
            for record in report["files"]:
                if (
                    sha256_file(path=safe_relative(destination, record["path"]))
                    != record["sha256"]
                ):
                    raise ValueError(
                        "A saved survival report file is missing or modified."
                    )
            return destination / "descriptives.xlsx"
        stage = Path(mkdtemp(prefix=".surv-stage-", dir=config_path))
        try:
            plot_dir = stage / "km_plots"
            plot_dir.mkdir()
            rows, plot_records = [], []
            for label, pair in state.pairs.pairs.items():
                duration = pair.time_profile.col_name
                event = pair.event_profile.col_name
                frame = state.df[[duration, event]].rename(
                    columns={duration: "time", event: "event"}
                )
                evaluation = binding["survival_evaluation"][label]
                survival = SingleClassSurv(surv_label=label, surv_df=frame)
                # Hash-derived filenames remain safe for arbitrary endpoint labels.
                filename = digest_json(label)[:24] + ".png"
                maximum = float(frame["time"].max())
                ticks = _survival_axis_ticks(
                    maximum=maximum,
                    unit_name=evaluation["unit"]["name"],
                )
                survival.plot_km_curve(
                    title=label,
                    xlabel=f"Time ({evaluation['unit']['name']})",
                    save_path=plot_dir / filename,
                    x_axis_range=ticks,
                    plot=False,
                    plot_grid=False,
                )
                for point in evaluation["time_points"]:
                    survival.get_survival_probability(time_point=point)
                    survival.get_rmst(restricted_time=point)
                row = pd.json_normalize(survival.descriptives).iloc[0].to_dict()
                row.update(surv_label=label, duration_unit=evaluation["unit"]["name"])
                rows.append(row)
                plot_records.append(
                    {
                        "endpoint": label,
                        "kind": "kaplan_meier",
                        "path": f"km_plots/{filename}",
                        "x_axis_unit": evaluation["unit"]["name"],
                        "x_axis_ticks": ticks,
                    }
                )
            frame = (
                pd.DataFrame(rows).set_index("surv_label")
                if rows
                else pd.DataFrame(index=pd.Index([], name="surv_label"))
            )
            frame.to_excel(stage / "descriptives.xlsx")
            files = [
                {"path": p.relative_to(stage).as_posix(), "sha256": sha256_file(path=p)}
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            ]
            report = {
                "schema_version": 1,
                "binding_sha256": signature,
                "input_artifact_id": reference.artifact_id,
                "files": files,
                "plots": plot_records,
                "status": "completed" if rows else "not_applicable",
            }
            report["report_id"] = digest_json(report)
            (stage / "report_manifest.json").write_text(
                canonical_json(report), encoding="utf-8"
            )
            os.rename(stage, destination)
        except BaseException:
            shutil.rmtree(stage)
            raise
    return destination / "descriptives.xlsx"


def _survival_axis_ticks(
    *,
    maximum: float,
    unit_name: str,
) -> list[int] | list[float]:
    """Choose display ticks without modifying survival times.

    Month-based axes use 12-month intervals within the observed range.
    Other units retain the existing six-tick display behaviour.
    """
    maximum = float(maximum)

    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError(
            "The maximum survival duration must be finite and nonnegative."
        )

    normalized_unit = unit_name.strip().casefold()

    if normalized_unit in {"month", "months"}:
        last_tick = 12 * int(maximum // 12)
        return list(range(0, last_tick + 1, 12))

    if maximum == 0:
        return [0.0]

    return np.linspace(0, maximum, 6).tolist()
