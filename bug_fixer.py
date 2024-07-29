import os.path
from typing import List, Dict, Any

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider.codemap.repomap import RepoMap, search_terms_from_message
from aider.coders.motleycrew_coder.inspect_object_tool import InspectEntityTool

from motleycrew import MotleyCrew
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from aider.coders.motleycrew_coder.file_edit_tools import get_file_edit_tools
from aider.coders.motleycrew_coder.motleycrew_coder import MotleyCrewCoder


def get_bug_fixer_task(
    coder: MotleyCrewCoder,
    rel_file_names: List[str],
    problem_statement: str,
    repo_map: RepoMap,
    crew: MotleyCrew,
    llm_name: str | None = None,
) -> SimpleTask:
    if llm_name is None:
        llm = None
    else:
        llm = init_llm(LLMFramework.LANGCHAIN, LLMFamily.OPENAI, llm_name=llm_name)

    repo_map_str = repo_map.repo_map_from_message(
        problem_statement, rel_added_fnames=set(rel_file_names), llm=llm
    )

    message = f"""Below is a real GitHub issue from a popular GitHub repository.
    The issue was filed some time ago.
    The repo has been checked out at the commit that existed at the moment the issue was filed.
    If you are already familiar with this repo, be cautious!
    You are working with an old version of the repo!
    Filenames, directory names, file contents, etc may be different than what you're used to.

    The issue is as follows:
    {problem_statement}
    
    You must try to fix the issue by modifying only the following files from the repo:
    {rel_file_names}
    
    Here is a summary of the repo, with a special focus on the files that need to be modified:
    {repo_map_str}
    
    You can use the inspect_entity tool to get more information about specific entities in the repo.
    ONLY use the inspect_entity tool as long as NECESSARY to identify the file that needs to be modified.
    NEVER call the inspect_entity tool more than 3 times.
    
    Your task is to fix the issue by modifying one of the files listed above, using the edit_file tool.
    """

    inspect_entity_tool = InspectEntityTool(repo_map)

    add_files_tool, get_modifiable_files_tool, file_edit_tool = get_file_edit_tools(coder)

    # TODO: have the output_handler run existing tests!
    # TODO: have the output handler write a test for the issue and use it to check the fix?
    add_files_tool.add_files(rel_file_names)
    tools = [inspect_entity_tool, file_edit_tool, get_modifiable_files_tool]

    file_finder = ReActToolCallingMotleyAgent(
        name="File-to-fix_finder",
        tools=tools,
        # TODO: is this the right way to inject the fake chat history?
        prompt_prefix=coder.create_prompt("").partial(
            tools=tools
        ),  # get usage examples as fake chat history
        chat_history=True,
        verbose=True,
    )

    task = SimpleTask(
        crew=crew,
        name="Apply the fixes to the issue",
        description=message,
        agent=file_finder,
    )

    return task
