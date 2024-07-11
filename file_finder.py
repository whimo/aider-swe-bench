import os.path

from langchain_core.tools import StructuredTool
from motleycrew import MotleyCrew
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingAgent
from motleycrew.common.exceptions import InvalidOutput

from typing import List

from aider.codemap.repomap import RepoMap, search_terms_from_message
from aider.coders.motleycrew_coder.inspect_object_tool import InspectEntityTool


def entry_point(
    problem_statement: str,
    repo_map: RepoMap,
    llm_name: str,
    gold_files: List[str],
    instance_id: str,
    result_writer,
):

    llm = init_llm(LLMFramework.LANGCHAIN, LLMFamily.OPENAI, llm_name=llm_name)

    repo_map_str = repo_map.repo_map_from_message(problem_statement, llm=None)

    message = f"""Below is a real GitHub issue from a popular GitHub repository.
    The issue was filed some time ago.
    The repo has been checked out at the commit that existed at the moment the issue was filed.
    If you are already familiar with this repo, be cautious!
    You are working with an old version of the repo!
    Filenames, directory names, file contents, etc may be different than what you're used to.
    
    Your task is to identify the single Python file that needs to be modified to fix the issue.
    You are given an initial overview of the repo:
    {repo_map_str}
    
    The issue is as follows:
    {problem_statement}
    
    You can use the inspect_object tool to get more information about specific entities in the repo.
    ONLY use the inspect_object tool as long as NECESSARY to identify the file that needs to be modified.
    NEVER call the inspect_object tool more than 3 times.
    """

    inspect_entity_tool = InspectEntityTool(repo_map)

    def check_file(rel_file_name: str):
        abs_path = str(repo_map.file_group.abs_root_path(rel_file_name))
        if not rel_file_name.endswith(".py"):
            raise InvalidOutput(f"File {rel_file_name} is not a Python file")
        if not os.path.exists(abs_path):
            raise InvalidOutput(f"File {abs_path} does not exist")

        return {"file": rel_file_name}

    output_handler = StructuredTool.from_function(
        name="output_handler",
        description="Output handler",
        func=check_file,
    )

    file_finder = ReActToolCallingAgent(
        name="File-to-fix_finder",
        tools=[inspect_entity_tool],
        output_handler=output_handler,
        chat_history=True,
        verbose=True,
    )

    crew = MotleyCrew()
    task = SimpleTask(
        crew=crew,
        name="Identify the .py file that should be modified to fix the issue",
        description=message,
        agent=file_finder,
    )

    result = crew.run()
    output = task.output
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
