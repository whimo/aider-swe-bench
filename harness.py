#!/usr/bin/env python

import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from joblib import Parallel, delayed

from dump import dump
from entry_point import entry_point
from tests import run_tests
from utils import get_devin_instance_ids, get_plausible, load_predictions, pick_winner
from utils import get_full_dataset  # noqa: F401
from utils import get_lite_dataset  # noqa: F401

REPOS_DNAME = Path("repos")
CHAT_LOGS_DNAME = Path("chat-logs")
PREDS_DNAME = Path("predictions")


def diff_versus_commit(git_dname, commit):
    """
    Take a diff of `git_dname` current contents versus the `commit`.
    """

    diff_cmd = f"git -C {git_dname} diff {commit}"
    diff_output = subprocess.check_output(diff_cmd.split()).decode()
    return diff_output


def files_in_patch(patch):
    """
    Extract the list of modified files from a unified diff patch string.
    """
    files = []
    for line in patch.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fname = line.split("/", 1)[1]
            if fname not in files:
                files.append(fname)
    return files


def checkout_repo(git_tempdir, entry, how="https"):
    """
    Clone the SWE Bench entry's git `repo` into `dname` at the `base_commit`.
    Make a tempdir if no `dname` provided.
    """
    if how == "https":
        github_url = "https://github.com/"
        repo_url = github_url + entry["repo"]
    else:  # ssh
        github_url = "git@github.com:"
        repo_url = github_url + entry["repo"] + ".git"

    commit = entry["base_commit"]

    print(repo_url, commit)

    checkout_repo_url_commit(git_tempdir, repo_url, commit)


def checkout_repo_url_commit(repo_dname, url, commit):
    """
    Clone the git `url` into `dname` at `commit`.
    Check a local cache of the bare repo to avoid pulling from github every time.
    """

    # Extract repo name from URL
    repo_name = url.split("/")[-1].split(".")[0]
    repo_name += ".git"

    # dump(repo_name)
    REPOS_DNAME.mkdir(exist_ok=True)
    bare_repo = REPOS_DNAME / repo_name

    if not bare_repo.exists():
        cmd = f"git clone --bare {url} {bare_repo}"
        subprocess.run(cmd.split(), check=True)

    cmd = f"git clone {bare_repo} {repo_dname}"
    subprocess.run(cmd.split(), check=True)

    cmd = f"git -c advice.detachedHead=false -C {repo_dname} checkout {commit}"
    subprocess.run(cmd.split(), check=True)


def show_problems(dataset):
    """
    Print out all the instance_id and problem_descriptions.
    """
    for inst, entry in dataset.items():
        problem = entry["problem_statement"].splitlines()[0]
        print(f"{inst}: {problem}")


def run_pre_existing_tests(entry, git_dname):
    """Given the current contents of the `git_dname`, run the tests that
    were present in the entry's `repo` at the time of the
    `base_commit` or which have been added into the repo since.  This
    checks if the code in the `git_dname` has broken pre-existing
    tests or is failing any newly added tests.

    It does NOT attempt to run the tests in the `test_patch` which
    are used to evaluate whether the `model_patch` has resolved the
    `problem_statement`.

    Returns None if all the tests passed. Returns the text of the
    test run output if any failed.
    """

    model_patch = diff_versus_commit(git_dname, entry["base_commit"])
    passed, output = run_tests(
        entry,
        model_patch=model_patch,
        use_test_patch=False,
    )
    # We were UNABLE to run tests
    if passed is None:
        return

    if passed:
        return

    # Just keep the output after the (no-op) test patch applied,
    # which is the actual output from the tests that were run.
    output = output.split(">>>>> Applied Patch (test)")[-1]

    return output


def process_one_instance(entry, num_tries, models, temperature, model_name_or_path, out_dname):
    """Process one `entry` from SWE Bench using the LLM `models` at the
    given `temperature`.  Set `model_name_or_path` in the result json.
    Store the result json and the chat log into `out_dname`.
    """

    instance_id = entry["instance_id"]
    base_commit = entry["base_commit"]

    print("=" * 60)
    dump(instance_id)
    print("=" * 60)
    problem_statement = entry["problem_statement"]

    print(problem_statement)

    ###
    # DO NOT assist aider by telling it which files need to be modified!
    oracle = False
    gold_files = files_in_patch(entry["patch"])
    if oracle:
        oracle_files = gold_files
    else:
        oracle_files = None
    ###

    chat_history_file = out_dname / (instance_id + ".md")

    # Clean up chat history from previous aborted run
    if chat_history_file.exists():
        chat_history_file.unlink()

    results = []
    cost = 0
    winner = None
    success = False
    attempt = 0

    # Do NUM_TRIES tries for each of the models, until we find a *plausible* solution
    for attempt in range(1, num_tries + 1):
        for model_family, model in models:
            set_model_as_motleycrew_default(model_family, model)
            dump(attempt, model)

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as git_tempdir:
                dump(git_tempdir)
                checkout_repo(git_tempdir, entry)

                # Prepare the test command which will run the pre-existing tests
                test_cmd = lambda: run_pre_existing_tests(entry, git_tempdir)  # noqa: E731

                def result_writer(output: dict):
                    output["gold_files"] = gold_files
                    output["gold_patch"] = entry["patch"]
                    output["instance_id"] = instance_id
                    print(output)
                    if not output["entity"][1] in set(gold_files):
                        print("Oops!")

                    out_fname = out_dname / (instance_id + "_file.json")
                    out_fname.write_text(json.dumps(output, indent=4))

                run_result = entry_point(
                    problem_statement=problem_statement,
                    repo_path=git_tempdir,
                    existing_test_runner=test_cmd,
                    chat_history_file=chat_history_file,
                    llm_name=model,
                )

                if run_result is None:
                    success = False
                    continue
                else:
                    success = True

                added_files = run_result["files"]
                tests_passed = run_result["result"]

                dump(instance_id)
                dump(gold_files)
                dump(added_files)

                # TODO: Keep track of API costs
                # Get the diff between the current state and the original commit
                print(">>>>>>>> Start diff_versus_commit <<<<<<<<")
                model_patch = diff_versus_commit(git_tempdir, base_commit)
                print(">>>>>>>> Finished diff_versus_commit <<<<<<<<")
                dump(model_patch)

            # Record the results for the logs
            result = dict(
                # Required args for running eval tests
                instance_id=instance_id,
                model_name_or_path=model_name_or_path,
                model_patch=model_patch,
                # For computing stats
                model=model,
                temperature=temperature,
                added_files=added_files,
                gold_files=gold_files,
                edited_files=files_in_patch(model_patch),
                edit_outcome=success,
                lint_outcome=success,
                test_outcome=success,
            )
            result["try"] = attempt  # `try` is a python keyword
            results.append(result)

            print(">>>>>>>> Result <<<<<<<<")
            dump(result)

            # Did we get a successful edit, lint and test? If so, we found a plausible solution!
            if model_patch and success:
                winner = result
                break

        # also break out of the attempts loop
        if winner:
            break

    # If there's no clear winner, look for the most viable result we got...
    if not winner:
        winner = pick_winner(results)

    if not winner:
        result = dict(
            # Required args for running eval tests
            instance_id=instance_id,
            model_name_or_path=model_name_or_path,
            model_patch=None,
        )

    dump(winner)
    if not winner:
        return

    print("\n\nFinal diff:\n")
    print(winner["model_patch"])

    # Avoid circular reference when we save to json
    winner = dict(winner)

    winner.update(
        dict(
            tries=attempt,
            all_results=results,  # Record all the results for later analysis
            cost=cost,  # total cost across all results
        )
    )

    out_fname = out_dname / (instance_id + ".json")
    out_fname.write_text(json.dumps(winner, indent=4))


def select_random_portion(lst, fraction, seed):
    # Set the seed for reproducibility
    random.seed(seed)

    # Calculate the number of elements to select
    k = math.ceil(len(lst) * fraction)

    # Randomly select k elements from the list
    selected_elements = random.sample(lst, k)

    return selected_elements


def process_instances(
    prefix,
    dataset,
    models,
    num_tries,
    temperature,
    threads,
    prior_dnames,
    instances,
    just_devin_570,
    dataset_portion,
    random_seed,
):
    """
    prefix - Prefix used in front of the dirname in predictions/.
    dataset - The subset of the SWE Bench dataset to process.
    models - List of models to use to try and find plausible solutions.
    num_tries - Number of attempts to make using each model.
    temperature - Temp to use during chat completions.
    threads - How many problems to attempt concurrently.
    prior_dnames - Names of predictions/ dirnames from previous runs.
                   If they contain a plausible solution for an instance,
                   don't continue looking.
    """
    models_slug = "--".join(model.replace("/", "-") for model_family, model in models)
    model_name_or_path = "aider--" + models_slug
    models_slug = prefix + "--" + models_slug

    dump(models)
    dump(temperature)

    dataset_instances = select_random_portion(list(dataset.keys()), dataset_portion, random_seed)
    dump(dataset_instances)

    out_dname = PREDS_DNAME / models_slug
    if not out_dname.exists():
        out_dname.mkdir(parents=True)

    dump(out_dname)

    # If we are restarting this run, figure out which instances are already done.
    done_preds = load_predictions([out_dname], just_devin_570)
    done_instances = set(done_preds.keys())
    dump(len(done_instances))

    dump(prior_dnames)
    prior_preds = load_predictions(prior_dnames, just_devin_570)
    dump(len(prior_preds))

    plausible_instances = get_plausible(prior_preds)
    dump(len(plausible_instances))

    if prior_preds:
        # Just keep trying to solve instances that exist in the previous runs
        all_instances = set(prior_preds.keys())
    else:
        all_instances = set(dataset_instances)

    remaining_instances = set(all_instances)
    remaining_instances -= done_instances
    remaining_instances -= plausible_instances

    remaining_instances = list(remaining_instances)

    if instances:
        remaining_instances = [inst for inst in remaining_instances if inst in instances]
    # random.shuffle(remaining_instances)

    dump(sorted(remaining_instances))
    dump(len(remaining_instances))

    print()
    print("press enter...")
    # input()

    if not CHAT_LOGS_DNAME.exists():
        CHAT_LOGS_DNAME.mkdir()

    chat_history_dname = CHAT_LOGS_DNAME / models_slug
    chat_history_dname.mkdir(exist_ok=True)

    random.shuffle(remaining_instances)

    if threads > 1:
        # process_one_instance_lox = lox.process(threads)(process_one_instance)
        # process_one_instance_func = process_one_instance_lox.scatter
        # gather = process_one_instance_lox.gather
        def process_one_instance_wrapper(instance_id):
            return process_one_instance(
                dataset[instance_id],
                num_tries,
                models,
                temperature,
                model_name_or_path,
                out_dname,
            )

        Parallel(n_jobs=threads)(
            delayed(process_one_instance_wrapper)(instance_id)
            for instance_id in remaining_instances
        )

    else:
        for instance_id in remaining_instances:
            # if instance_id in done_instances:
            #     print("skipping", instance_id)
            #     continue

            process_one_instance(
                dataset[instance_id],
                num_tries,
                models,
                temperature,
                model_name_or_path,
                out_dname,
            )

            print("#" * 60)
        # input()

    # if threads > 1:
    #     gather()


def main(
    prefix,
    models,
    num_tries,
    temperature,
    threads,
    prior_dnames,
    instances,
    dataset_split,
    dataset_portion,
    random_seed,
):
    models_json = Path(".aider.models.json")
    if models_json.exists():
        print(f"Registering {models_json}")
        register_litellm_models([str(models_json)])

    # Load the SWE Bench dataset
    # dataset = get_full_dataset()
    dataset = get_lite_dataset(dataset_split)

    just_devin_570 = False

    if just_devin_570:
        # Filter it to the Devin 570
        devin_insts = get_devin_instance_ids()
        dataset = dict((inst, entry) for inst, entry in dataset.items() if inst in devin_insts)

    # bad_ids = [
    #     #     "pylint-dev__astroid-1333",
    #     #     "sqlfluff__sqlfluff-1517",
    #     #     # "sqlfluff__sqlfluff-1625",
    #     #     "sqlfluff__sqlfluff-1733",
    #     "sqlfluff__sqlfluff-1763",
    # ]
    # dataset = dict((inst, entry) for inst, entry in dataset.items() if inst in bad_ids)

    process_instances(
        prefix,
        dataset,
        models,
        num_tries,
        temperature,
        threads,
        prior_dnames,
        instances,
        just_devin_570,
        dataset_portion,
        random_seed,
    )


from motleycrew.common import Defaults


def set_model_as_motleycrew_default(llm_family, model_name):
    Defaults.DEFAULT_LLM_FAMILY = llm_family
    Defaults.DEFAULT_LLM_NAME = model_name


if __name__ == "__main__":
    from motleycrew.common import LLMFamily

    # Configure 1 or more models to use to try and find plausible solutions
    #
    # models = ["openrouter/deepseek/deepseek-chat"]
    # models = ["gpt-4o", "openrouter/anthropic/claude-3-opus"]
    # models = ["openrouter/anthropic/claude-3-opus"]

    # models = ["gpt-4-1106-preview"]
    # models = ["openrouter/anthropic/claude-3.5-sonnet"]
    # models = ["claude-3-5-sonnet-20240620"]

    models = [(LLMFamily.OPENAI, "gpt-4o")]  # (LLMFamily.OPENAI, "gpt-4o")]  # ,

    # How many attempts per model to try and find a plausible solutions?
    num_tries = 3
    # How many threads to use for attempting instances in parallel
    threads = 12

    # Any predictions/ dirs provided on the command line are treated
    # as earlier, higher priority runs.  If a plausible solution was
    # found for an instance already, we don't need to keep looking in
    # this run.
    prior_dnames = sys.argv[1:]

    instances = [
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
    ]

    # instances = ["pytest-dev__pytest-7432", "scikit-learn__scikit-learn-14894"]

    # What temperature to use during chat completions
    temperature = 0
    prefix = "p6"
    status = main(
        prefix=prefix,
        models=models,
        num_tries=num_tries,
        temperature=temperature,
        threads=threads,
        prior_dnames=prior_dnames,
        instances=instances,
        dataset_split="test",
        dataset_portion=0.2,
        random_seed=3,
    )
    sys.exit(status)
