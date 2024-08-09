#!/usr/bin/env python

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from joblib import Parallel, delayed

USE_ORIG_AIDER = False

if USE_ORIG_AIDER:
    from aider.coders import Coder

    construct_coder = Coder.create

else:
    from aider.coders import MotleyCrewCoder

    construct_coder = MotleyCrewCoder

from aider.io import InputOutput
from aider.models import Model, register_litellm_models

from entry_point import entry_point

from dump import dump
from tests import run_tests
from utils import get_full_dataset  # noqa: F401
from utils import get_lite_dataset  # noqa: F401
from utils import get_devin_instance_ids, get_plausible, load_predictions, pick_winner

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


def get_coder(model, git_dname, chat_history_file, test_cmd, temperature, oracle_files=None):
    """
    Get an instance of aider to work with the given LLM `model` at `temperature`
    on the code in `git_dname`. Will store the markdown chat logs in
    the `chat_history_file`. Tells aider it can use the `test_cmd` to
    run tests after the LLM edits files.

    If `oracle_files` are provided, they are added to the aider chat automatically.
    """
    if oracle_files and git_dname:
        oracle_files = [Path(git_dname) / fname for fname in oracle_files]

    model = Model(model)

    io = InputOutput(
        yes=True,  # Say yes to every suggestion aider makes
        chat_history_file=None,  # Log the chat here
        input_history_file="/dev/null",  # Don't log the "user input"
    )

    dump(git_dname)

    coder = construct_coder(
        main_model=model,
        io=io,
        git_dname=git_dname,
        map_tokens=2048,  # Use 2k tokens for the repo map
        stream=False,
        auto_commits=False,  # Don't bother git committing changes
        fnames=oracle_files,
        auto_test=True,  # Automatically run the test_cmd after making changes
        test_cmd=test_cmd,
        # verbose=True,
        # edit_format="udiff",
        max_chat_history_tokens=8 * 1024,
    )
    coder.temperature = temperature

    # Take at most 4 steps before giving up.
    # Usually set to 5, but this reduces API costs.
    coder.max_reflections = 4

    # Add announcement lines to the markdown chat log
    coder.show_announcements()

    # messages = coder.format_messages()
    # utils.show_messages(messages)

    return coder


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

                # Get an instance of aider
                coder = get_coder(
                    model,
                    git_tempdir,
                    chat_history_file,
                    test_cmd,
                    temperature,
                    oracle_files,
                )

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
                    problem_statement,
                    coder.repo_map,
                    coder,
                    test_cmd,
                    result_writer,
                    None,  # coder.main_model.name,
                    chat_history_file,
                    gold_files,
                    entry["patch"],
                )

                if run_result is None:
                    coder.edit_outcome = False
                    coder.lint_outcome = False
                    coder.test_outcome = False
                    run_failed = True
                    continue
                else:
                    coder.edit_outcome = True
                    coder.lint_outcome = True
                    coder.test_outcome = True
                    run_failed = False

                added_files = run_result["files"]
                tests_passed = run_result["result"]

                dump(instance_id)
                dump(gold_files)
                dump(added_files)

                # Keep track of API costs
                cost += coder.total_cost

                # Get the diff between the current state and the original commit
                print(">>>>>>>> Start diff_versus_commit <<<<<<<<")
                model_patch = diff_versus_commit(git_tempdir, base_commit)
                print(">>>>>>>> Finished diff_versus_commit <<<<<<<<")
                dump(model_patch)

            # TODO: catch fails of

            # Record the results for the logs
            result = dict(
                # Required args for running eval tests
                instance_id=instance_id,
                model_name_or_path=model_name_or_path,
                model_patch=model_patch,
                # For computing stats
                model=model,
                temperature=temperature,
                cost=coder.total_cost,
                added_files=added_files,
                gold_files=gold_files,
                edited_files=files_in_patch(model_patch),
                edit_outcome=coder.edit_outcome,
                lint_outcome=coder.lint_outcome,
                test_outcome=coder.test_outcome,
            )
            result["try"] = attempt  # `try` is a python keyword
            results.append(result)

            print(">>>>>>>> Result <<<<<<<<")
            dump(result)

            # Did we get a successful edit, lint and test? If so, we found a plausible solution!
            if model_patch and coder.edit_outcome and coder.lint_outcome and coder.test_outcome:
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
        all_instances = set(dataset.keys())

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
    input()

    if not CHAT_LOGS_DNAME.exists():
        CHAT_LOGS_DNAME.mkdir()

    chat_history_dname = CHAT_LOGS_DNAME / models_slug
    chat_history_dname.mkdir(exist_ok=True)

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
            for instance_id in sorted(remaining_instances)
        )

    else:
        for instance_id in sorted(remaining_instances):
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


def main(prefix, models, num_tries, temperature, threads, prior_dnames, instances):
    models_json = Path(".aider.models.json")
    if models_json.exists():
        print(f"Registering {models_json}")
        register_litellm_models([str(models_json)])

    # Load the SWE Bench dataset
    # dataset = get_full_dataset()
    dataset = get_lite_dataset()

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
    threads = 4

    # Any predictions/ dirs provided on the command line are treated
    # as earlier, higher priority runs.  If a plausible solution was
    # found for an instance already, we don't need to keep looking in
    # this run.
    prior_dnames = sys.argv[1:]

    instances = [
        # "astropy__astropy-14365",
        # "django__django-11049",
        # "django__django-12983",
        # "django__django-15781",
        # "matplotlib__matplotlib-23476",
        # "matplotlib__matplotlib-24334",
        # "mwaskom__seaborn-3010",
        # "psf__requests-863",
        # "pydata__xarray-4248",
        # "pytest-dev__pytest-5103",
        # "pytest-dev__pytest-11143",
        # "scikit-learn__scikit-learn-13584",
        # "scikit-learn__scikit-learn-25500",
        # "sphinx-doc__sphinx-8273",
        # "sphinx-doc__sphinx-8282",
        # "sympy__sympy-13146",
        # "sympy__sympy-13773",
        # "sympy__sympy-24066",
    ]

    # What temperature to use during chat completions
    temperature = 0
    prefix = "dev3"

    status = main(
        prefix=prefix,
        models=models,
        num_tries=num_tries,
        temperature=temperature,
        threads=threads,
        prior_dnames=prior_dnames,
        instances=instances,
    )
    sys.exit(status)
