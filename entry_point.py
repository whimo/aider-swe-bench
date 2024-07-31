from typing import Any, Callable
from aider.codemap.repomap import RepoMap

from motleycrew import MotleyCrew
from motleycrew.common import logger
from aider.coders.motleycrew_coder.motleycrew_coder import MotleyCrewCoder

from file_finder import get_file_finder_task
from bug_fixer import get_bug_fixer_task


def entry_point(
    problem_statement: str,
    repo_map: RepoMap,
    coder: MotleyCrewCoder,
    existing_test_runner: Callable,
    result_writer,
    llm_name: str | None = None,
):
    try:
        crew = MotleyCrew()
        file_finder_task = get_file_finder_task(problem_statement, repo_map, crew, llm_name)
        result = crew.run()
        output = file_finder_task.output
        if not isinstance(output, dict) or "entity" not in output or len(output["entity"]) != 2:
            print(output)
            return None

        # if not output["entity"][1] in set(gold_files):
        #     result_writer(output)
        #     print("ouch!")

        # Now run the bug-fixing task
        crew = MotleyCrew()
        bug_fixer_task = get_bug_fixer_task(
            coder,
            output["entity"],
            problem_statement,
            existing_test_runner,
            repo_map,
            crew,
            llm_name,
        )
        result = crew.run()
        output2 = bug_fixer_task.output
        if output2 != "Tests passed!":
            return None
        result_writer(output)

        print("yay!")
        return {"files": [output["entity"][1]], "result": output2}
    except Exception as e:
        logger.error(e)
        return None


# So, the plan is:
# Create a repo map driven by the issue description
# Give the agent a tool to get details for desired objects
# Give the agent an output handler to return a filename and line number?
