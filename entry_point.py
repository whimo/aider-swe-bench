import os.path
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider.codemap.repomap import RepoMap, search_terms_from_message
from aider.coders.motleycrew_coder.inspect_object_tool import InspectEntityTool

from motleycrew import MotleyCrew
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.common.exceptions import InvalidOutput
from file_finder import get_file_finder_task


def entry_point(
    problem_statement: str,
    repo_map: RepoMap,
    gold_files: List[str],
    instance_id: str,
    result_writer,
    llm_name: str | None = None,
):

    crew = MotleyCrew()
    file_finder_task = get_file_finder_task(problem_statement, repo_map, crew, llm_name)
    result = crew.run()
    output = file_finder_task.output
    output["gold_files"] = gold_files
    output["instance_id"] = instance_id
    print(output)
    if output["file"] not in gold_files:
        print("Oops!")
    result_writer(output)

    print("yay!")


# So, the plan is:
# Create a repo map driven by the issue description
# Give the agent a tool to get details for desired objects
# Give the agent an output handler to return a filename and line number?
