#!/usr/bin/env python

import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from swebench.metrics.report import get_model_report

from dump import dump  # noqa: F401
from tests import remove_patches_to_tests, run_tests
from utils import (
    LITE_DATASET_FNAME,
    choose_predictions,
    get_dataset,
    get_devin_instance_ids,
    load_predictions,
    old,
)

JUST_DEVIN_570 = False
DATASET_FNAME = LITE_DATASET_FNAME
using_dataset = "lite"


NUM_EVAL_PROCS = 5


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


def run_evals(swe_bench_tasks, log_dir, predictions_jsonl):
    base = os.getcwd()

    run_evals_cmd = f"""
python {base}/aider-swe-bench/SWE-bench-docker/run_evaluation.py
    --log_dir {base}/{log_dir}
    --swe_bench_tasks {base}/{swe_bench_tasks}
    --skip_existing
    --predictions_path {predictions_jsonl}
    --num_processes {NUM_EVAL_PROCS}
"""
    run_evals_cmd = " ".join([line.strip() for line in run_evals_cmd.split() if line.strip()])
    dump(run_evals_cmd)

    subprocess.run(run_evals_cmd.split(), check=True)


def get_report(swe_bench_tasks, log_dir, predictions_jsonl, model_name_or_path):
    if True:
        # try:
        base = os.getcwd()
        print(
            model_name_or_path,
            predictions_jsonl,
            swe_bench_tasks,
            log_dir,
        )
        report = get_model_report(
            model_name_or_path,
            predictions_jsonl,
            f"{base}/{swe_bench_tasks}",
            log_dir,
            verbose=True,
        )
    # except KeyError:
    #     report = dict()

    # for k, v in report.items():
    #    print(f"- {k}: {len(v)}")

    # dump(report)

    resolved_instances = set(report["resolved"])
    dump(sorted(resolved_instances))

    generated = set(report["generated"])
    applied = set(report["applied"])
    generated_minus_applied = generated - applied
    dump(len(generated_minus_applied))
    generated_minus_applied = " ".join(iid + "*" for iid in sorted(generated_minus_applied))
    dump(generated_minus_applied)

    applied_minus_resolved = applied - resolved_instances

    dump(sorted(applied_minus_resolved))

    aider_resolved_we_didnt = applied_minus_resolved.intersection(AIDER_RESOLVED)
    dump(sorted(aider_resolved_we_didnt))

    we_resolved_aider_didnt = resolved_instances - AIDER_RESOLVED
    dump(sorted(we_resolved_aider_didnt))

    with_logs = set(report["with_logs"])
    with_logs_minus_applied = with_logs - applied
    dump(len(with_logs_minus_applied))
    dump(with_logs_minus_applied)

    no_apply = set(report["no_apply"])
    dump(len(no_apply))
    no_apply = " ".join(iid + "*" for iid in sorted(no_apply))
    dump(no_apply)

    return report


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


def run_evals_on_dname(dname):
    dname = Path(dname)

    predictions = load_predictions([dname], devin_only=(using_dataset == "devin"))

    predictions_jsonl = preds_to_jsonl(dname, predictions)
    dump(predictions_jsonl)

    log_dir = Path("logs") / dname.name
    log_dir.mkdir(exist_ok=True, parents=True)
    dump(log_dir)

    any_need_evals = any("resolved" not in pred for pred in predictions.values())
    any_need_evals = True
    if any_need_evals:
        run_evals(DATASET_FNAME, str(log_dir), predictions_jsonl)

        model_name_or_path = list(predictions.values())[0]["model_name_or_path"]
        report = get_report(DATASET_FNAME, log_dir, predictions_jsonl, model_name_or_path)
        predictions = update_pred_json(predictions, report)

    return predictions_jsonl, log_dir


def combine_jsonl_logs(predictions, model_name_or_path):
    logs = Path("logs")
    log_dir = logs / model_name_or_path
    old(log_dir)

    log_dir.mkdir(exist_ok=True)
    dump(log_dir)

    preds_dir = Path("predictions") / model_name_or_path

    predictions_jsonl = preds_to_jsonl(preds_dir, predictions)
    for inst, pred in predictions.items():
        from_fname = logs / pred["dname"]
        # dump(from_fname, inst)
        from_fname = list(from_fname.glob(f"{inst}.*.log"))
        assert len(from_fname) <= 1, from_fname
        if not len(from_fname):
            print("Missing", pred["dname"], inst)
            continue
        from_fname = from_fname[0]
        # dump(from_fname)

        to_fname = log_dir / f"{inst}.{model_name_or_path}.eval.log"
        # dump(from_fname, to_fname)
        shutil.copyfile(from_fname, to_fname)

    return predictions_jsonl, log_dir


def main(path):
    # Run with a set of prediction directories, in order of priority.
    # Plausible solution found in the earliest directory will be selected.
    dnames = [path]
    # dnames = sys.argv[1:]

    # Make sure evals have been completed on all instances in all supplied
    # predictions dirs.
    for dname in dnames:
        dump(dname)
        run_evals_on_dname(dname)

    # Directory to make under predictions/ and logs/ to store the
    # plausible predictions which were selected.
    # Outputs a clean `all_preds.jsonl`, `results.json`, `logs/`
    # and copies over all markdown chat transcripts.
    model_name_or_path = "lite-multi"

    preds_dir = Path("predictions") / model_name_or_path
    old(preds_dir)
    preds_dir.mkdir(exist_ok=True)

    # Choose the 1st plausible pred or use the fallback logic for least bad pred
    predictions = choose_predictions(
        dnames, model_name_or_path, copy_md=True, devin_only=(using_dataset == "devin")
    )
    if not predictions:
        print("No predictions")
        return

    dump(len(predictions))

    predictions_jsonl, log_dir = combine_jsonl_logs(predictions, model_name_or_path)
    report = get_report(DATASET_FNAME, log_dir, predictions_jsonl, model_name_or_path)
    results_json = Path("predictions") / model_name_or_path / "results.json"
    results_json.write_text(json.dumps(report, indent=4))

    # Show the key stats on how many instances are resolved, etc
    counts = defaultdict(int, [(k, len(v)) for k, v in report.items()])
    dump(counts)

    total = counts["generated"] + counts["no_generation"]
    dump(total)
    missing_logs = total - counts["with_logs"]
    dump(missing_logs)

    if total:
        percent = counts["resolved"] * 100 / total
        print(f"{percent= :.1f}%")

        plus_one_percent = (counts["resolved"] + 1) * 100 / (total + 1)
        print(f"{plus_one_percent= :.1f}%")

    print()

    # NEED TO BE RUN?
    need_to_be_run = missing_logs - counts["no_generation"]
    if need_to_be_run:
        dump(need_to_be_run)

        should_count = total - need_to_be_run
        dump(should_count)

        percent_of_should = counts["resolved"] * 100 / should_count
        print(f"{percent_of_should=:.1f}")

    # COSTS
    costs = []
    for data in predictions.values():
        cost = data.get("cost")
        if cost is not None and cost > 0:
            costs.append(cost)

    if len(costs):
        #
        # Cost estimates are unreliable!
        #
        recent = costs[-5:]
        recent = [f"{c:.2f}" for c in recent]
        print("recent costs:", ", ".join(recent))
        avg_cost = sum(costs) / len(costs)
        print(f"avg_cost: ${avg_cost:.2f}/instance")

        spent = sum(costs)
        print(f"spent: ${spent:.2f}")

        # If configured to assume the Devin 570 need to be processed
        if using_dataset == "devin":
            num_instances = len(get_devin_instance_ids())
        elif using_dataset == "lite":
            num_instances = 300
        else:
            num_instances = len(json.load(open(DATASET_FNAME)))

        expected_cost = num_instances * avg_cost
        print(f"expected_cost: ${expected_cost:.2f}")

        print()

    # added gold files?

    total_plausible = 0
    resolved_plausible = 0

    total_with_added = 0
    total_with_gold_attr = 0
    total_added_gold = 0
    gold_resolved = 0

    added_timeline = ""
    repomap_timeline = ""
    timeline = ""
    for instance_id, data in predictions.items():
        gold_files = set(data.get("gold_files", []))
        added_files = set([r.replace("\\", "/") for r in data.get("added_files", [])])

        resolved = data.get("resolved")
        added_gold = (added_files.intersection(gold_files) == gold_files) and gold_files

        plausible = (
            data["model_patch"]
            and data["edit_outcome"]
            and data["lint_outcome"]
            and data["test_outcome"]
        )
        if plausible:
            total_plausible += 1
            if resolved:
                resolved_plausible += 1

        if added_files:
            total_with_added += 1
            added_timeline += str(len(added_files))
        else:
            added_timeline += "_"

        if gold_files:
            total_with_gold_attr += 1
        if added_gold:
            total_added_gold += 1

        if not gold_files and not resolved:
            timeline += "."
        elif added_gold and resolved:
            timeline += "R"
            gold_resolved += 1
        elif added_gold and not resolved:
            timeline += "g"
        elif not added_gold and not resolved:
            timeline += "_"
        elif not added_gold and resolved:
            timeline += "!"
            # print(data['instance_id'])

        if data.get("initial_map_has_gold_file") or data.get("map_has_gold_file"):
            repomap_timeline += "M"
        else:
            repomap_timeline += "_"

    pct_maps_with_gold_file = len(repomap_timeline.replace("_", "")) / len(repomap_timeline) * 100
    dump(pct_maps_with_gold_file)

    dump(total_with_gold_attr)
    dump(total_added_gold)
    if total_with_gold_attr:
        pct_added = total_added_gold / total_with_gold_attr * 100
        print(f"pct_added_gold: {pct_added:.1f}%")

    if total_added_gold:
        pct_added_gold_resolved = gold_resolved / total_added_gold * 100
        print(f"pct_added_gold_resolved: {pct_added_gold_resolved:.1f}%")

        print()

    dump(total_with_added)
    pct_with_added = total_with_added / total * 100
    dump(pct_with_added)
    print()
    # print(timeline)
    # print(added_timeline)
    # print(repomap_timeline)

    dump(total_plausible)
    dump(resolved_plausible)
    if total_plausible:
        pct_resolved_plausible = 100 * resolved_plausible / total_plausible
        dump(pct_resolved_plausible)

    pct_plausible = total_plausible / total * 100
    dump(pct_plausible)

    # stats_on_tests_before_and_after(report, predictions.values())


def stats_on_tests_before_and_after(report, predictions):
    num = 0
    num_before_pass = 0
    num_pass_to_fail = 0

    dataset = get_dataset()

    random.shuffle(predictions)

    outcomes = defaultdict(int)
    for pred in predictions:
        instance_id = pred["instance_id"]

        # if instance_id not in has_patch_not_resolved:
        #    continue

        num += 1

        entry = dataset[instance_id]
        before_passed, _ = run_tests(entry)
        if not before_passed:
            continue

        after_passed, _ = run_tests(entry, model_patch=pred["model_patch"])

        resolved = instance_id in report["resolved"]
        dump(before_passed, after_passed, resolved)
        outcome = (before_passed, after_passed, resolved)
        outcomes[outcome] += 1
        dump(sorted(outcomes.items()))

        if before_passed:
            num_before_pass += 1
        if before_passed and not after_passed:
            num_pass_to_fail += 1

        print()
        dump(num)
        dump(num_before_pass)
        dump(num_pass_to_fail)

        pct_before_pass = num_before_pass / num * 100
        dump(pct_before_pass)
        pct_pass_to_fail = num_pass_to_fail / num_before_pass * 100
        dump(pct_pass_to_fail)

        print()


if __name__ == "__main__":
    status = main(path="/home/ubuntu/predictions/testpart1--gpt-4o")
    sys.exit(status)
