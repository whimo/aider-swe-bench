#!/usr/bin/env python

import json
import sys
from pathlib import Path
from typing import Optional, List

from dump import dump  # noqa: F401
from swebench.harness.run_evaluation import main as run_evaluation
from tests import remove_patches_to_tests
from utils import load_predictions

AIDER_RESOLVED = {
    "sympy__sympy-23117",
    "django__django-13315",
    "mwaskom__seaborn-3010",
    "django__django-14382",
    "django__django-15789",
    "django__django-14999",
    "django__django-14915",
    "django__django-16139",
    "django__django-16255",
    "scikit-learn__scikit-learn-13584",
    "django__django-11583",
    "sympy__sympy-14774",
    "django__django-13768",
    "django__django-12286",
    "django__django-13658",
    "django__django-14752",
    "django__django-16527",
    "django__django-16379",
    "pytest-dev__pytest-11143",
    "psf__requests-863",
    "django__django-11422",
    "django__django-13447",
    "sympy__sympy-24213",
    "sympy__sympy-13647",
    "scikit-learn__scikit-learn-10297",
    "django__django-14016",
    "django__django-16041",
    "sympy__sympy-13031",
    "sympy__sympy-17655",
    "sympy__sympy-24152",
    "django__django-11179",
    "matplotlib__matplotlib-23562",
    "scikit-learn__scikit-learn-15535",
    "scikit-learn__scikit-learn-13241",
    "sympy__sympy-20212",
    "psf__requests-2317",
    "pytest-dev__pytest-7373",
    "scikit-learn__scikit-learn-13496",
    "django__django-12453",
    "django__django-16046",
    "scikit-learn__scikit-learn-11281",
    "pydata__xarray-5131",
    "sympy__sympy-18621",
    "pytest-dev__pytest-7432",
    "django__django-12983",
    "django__django-17051",
    "matplotlib__matplotlib-23964",
    "sympy__sympy-21055",
    "sympy__sympy-15678",
    "pytest-dev__pytest-7490",
    "django__django-15814",
    "sympy__sympy-13480",
    "scikit-learn__scikit-learn-13779",
    "django__django-13158",
    "pytest-dev__pytest-5227",
    "django__django-13401",
    "psf__requests-2674",
    "django__django-11099",
    "sympy__sympy-13471",
    "scikit-learn__scikit-learn-14894",
    "matplotlib__matplotlib-26020",
    "django__django-13933",
    "sympy__sympy-22714",
    "django__django-12708",
    "scikit-learn__scikit-learn-13439",
    "django__django-14855",
    "django__django-11133",
    "django__django-13590",
    "pytest-dev__pytest-5692",
    "django__django-12125",
    "scikit-learn__scikit-learn-25570",
    "matplotlib__matplotlib-23913",
    "sympy__sympy-18532",
    "sphinx-doc__sphinx-8713",
    "sphinx-doc__sphinx-8721",
    "django__django-11039",
    "django__django-13710",
    "django__django-11049",
    "django__django-14608",
}


def update_pred_json(predictions, report):
    all_instances = set(report.get("generated", []))
    all_instances.update(set(report.get("no_generation", [])))

    for instance_id, pred in predictions.items():
        was_resolved = instance_id in report["resolved"]
        if "resolved" in pred and pred["resolved"] == was_resolved:
            continue

        assert instance_id in all_instances, instance_id

        pred["resolved"] = was_resolved
        save = dict(pred)
        del save["json_fname"]
        Path(pred["json_fname"]).write_text(json.dumps(save, indent=4))

    return predictions


def preds_to_jsonl(dname, predictions):
    dname = Path(dname)

    predictions_jsonl = str(dname / "all_preds.jsonl")
    dump(predictions_jsonl)
    model_name_or_path = list(predictions.values())[0]["model_name_or_path"]
    with open(predictions_jsonl, "w") as fh:
        for inst, pred in predictions.items():
            assert model_name_or_path == pred["model_name_or_path"]
            minimal_pred = dict(
                model_name_or_path=model_name_or_path,
                model_patch=remove_patches_to_tests(pred["model_patch"]),
                instance_id=pred["instance_id"],
            )
            fh.write(json.dumps(minimal_pred) + "\n")
    return predictions_jsonl


def run_evals_on_dir(
    predictions_dir: str,
    dataset_name: str,
    split: str,
    instance_ids: Optional[List] = None,
    max_workers: int = 4,
    use_gold: bool = False,
    run_id_suffix: Optional[str] = None,
):
    predictions_dir = Path(predictions_dir)

    predictions = load_predictions([predictions_dir])
    instance_ids = instance_ids or list(predictions.keys())
    run_id = str(predictions_dir).replace("/", "_")

    if run_id_suffix:
        run_id += f"_{run_id_suffix}"

    if use_gold:
        predictions_jsonl = "gold"
        run_id += "_gold"
    else:
        predictions_jsonl = preds_to_jsonl(predictions_dir, predictions)
    dump(predictions_jsonl)

    report_file = run_evaluation(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=predictions_jsonl,
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="instance",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=1800,
    )

    report = json.load(open(report_file, "r"))
    return report


def main(
    predictions_dir: str,
    dataset_name: str,
    split: str,
    instance_ids: Optional[List] = None,
    max_workers: int = 4,
    use_gold: bool = False,
    run_id_suffix: Optional[str] = None,
):
    # Run with a set of prediction directories, in order of priority.
    # Plausible solution found in the earliest directory will be selected.
    report = run_evals_on_dir(
        predictions_dir=predictions_dir,
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        max_workers=max_workers,
        use_gold=use_gold,
        run_id_suffix=run_id_suffix,
    )

    resolved_instances = report["resolved_ids"]
    dump(sorted(resolved_instances))

    applied_minus_resolved = set(report["submitted_ids"]) - set(resolved_instances)

    aider_resolved_we_didnt = applied_minus_resolved.intersection(AIDER_RESOLVED)
    dump(sorted(aider_resolved_we_didnt))

    we_resolved_aider_didnt = set(resolved_instances) - AIDER_RESOLVED
    dump(sorted(we_resolved_aider_didnt))

    percent_of_total = report["resolved_instances"] * 100 / report["total_instances"]
    print(f"{percent_of_total= :.1f}%")

    percent_of_sumbitted = report["resolved_instances"] * 100 / report["completed_instances"]
    print(f"{percent_of_sumbitted= :.1f}%")

    plus_one_percent = (
        (report["resolved_instances"] + 1) * 100 / (report["completed_instances"] + 1)
    )
    print(f"{plus_one_percent= :.1f}%")

    return 0


if __name__ == "__main__":
    status = main(
        predictions_dir="/home/ubuntu/predictions/p5--gpt-4o",
        dataset_name="swebench-lite",
        split="test",
        max_workers=8,
        run_id_suffix="rep1",
        # instance_ids=["matplotlib__matplotlib-23562"],
        # use_gold=True,
    )
    sys.exit(status)
