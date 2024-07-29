from typing import Any
from aider.codemap.repomap import RepoMap

from motleycrew import MotleyCrew
from file_finder import get_file_finder_task
from bug_fixer import get_bug_fixer_task


def entry_point(
    problem_statement: str,
    repo_map: RepoMap,
    coder: Any,
    gold_files: list[str],
    result_writer,
    llm_name: str | None = None,
):
    crew = MotleyCrew()
    file_finder_task = get_file_finder_task(problem_statement, repo_map, crew, llm_name)
    result = crew.run()
    output = file_finder_task.output

    if not set(output["files"]).intersection(set(gold_files)):
        result_writer(output)
    else:
        crew = MotleyCrew()
        bug_fixer_task = get_bug_fixer_task(
            coder, output["files"], problem_statement, repo_map, crew, llm_name
        )
        result = crew.run()
        output = bug_fixer_task.output
        # result_writer(output)

    print("yay!")


# So, the plan is:
# Create a repo map driven by the issue description
# Give the agent a tool to get details for desired objects
# Give the agent an output handler to return a filename and line number?
