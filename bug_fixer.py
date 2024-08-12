from typing import List, Callable

from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.tools import render_text_description

from aider.codemap.repomap import RepoMap
from aider.coders.motleycrew_coder.file_edit_tools import get_file_edit_tools
from aider.coders.motleycrew_coder.inspect_entity_tool import InspectEntityTool
from aider.coders.motleycrew_coder.motleycrew_coder import MotleyCrewCoder
from motleycrew import MotleyCrew
from motleycrew.agents import MotleyOutputHandler
from motleycrew.agents.langchain.tool_calling_react import ReActToolCallingMotleyAgent
from motleycrew.agents.langchain.tool_calling_react_prompts import ToolCallingReActPromptsForOpenAI
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.common.llms import init_llm, LLMFramework, LLMFamily
from motleycrew.tasks import SimpleTask


class BugFixerPrompts(ToolCallingReActPromptsForOpenAI):
    reminder_message_with_output_handler = SystemMessagePromptTemplate.from_template(
        """# `edit_file` tool call Rules:
    
Every *SEARCH* argument must *EXACTLY MATCH* the existing source code, character for character, including all comments, docstrings, etc.

`edit_file` tool will replace *all* matching occurrences.
Include enough lines to make the SEARCH blocks unique.

Include *ALL* the code being searched and replaced!

Only call `edit_file` for files that the user has added to the chat!

To move code within a file, use 2 `edit_file` calls: 1 to delete it from its current location, 1 to insert it in the new location.

If you want to put code in a new file, call the `edit_file` tool with:
- A new file path, including dir name if needed
- An empty `SEARCH` argument
- The new file's contents in the `REPLACE` argument

ONLY EVER RETURN CODE IN THE ARGUMENTS OF THE `edit_file` TOOL CALL!

You have access to the following tools:
{tools}

All changes to files must be made using the `edit_file` tool.

What is the next step towards answering the question?
Your reply MUST begin with "Thought:" and describe what the next step should be.
The reply must contain the tool call or calls that you described in the thought.
TOOL CALLS WITHOUT A THOUGHT WILL NOT BE ACCEPTED!

If you have sufficient information to answer the question, call the output handler tool:
{output_handler}

Write your reply, starting with "Thought:":
"""
    )


def get_bug_fixer_task(
    coder: MotleyCrewCoder,
    entity_to_modify: List[str],
    problem_statement: str,
    existing_test_runner: Callable,
    repo_map: RepoMap,
    crew: MotleyCrew,
    llm_name: str | None = None,
) -> SimpleTask:
    if llm_name is None:
        llm = None
    else:
        llm = init_llm(LLMFramework.LANGCHAIN, LLMFamily.OPENAI, llm_name=llm_name)

    # mod_fname = entity_to_modify[1]
    # mod_entity = entity_to_modify[0]

    repo_map_str = repo_map.repo_map_from_message(problem_statement, llm=llm)

    message = f"""Below is a real GitHub issue from a popular GitHub repository.
The issue was filed some time ago.
The repo has been checked out at the commit that existed at the moment the issue was filed.
If you are already familiar with this repo, be cautious!
You are working with an old version of the repo!
Filenames, directory names, file contents, etc may be different than what you're used to.

Propose changes to update the repo to fix the problem below.
The issue is as follows:

{problem_statement}

Here is a summary of the repo, with a special focus on the files that need to be modified:
{repo_map_str}

You must FIRST identify the entity that needs to be modified and ONLY THEN make changes to the code.

You can use the inspect_entity tool to get more information about specific entities in the repo.
ONLY use the inspect_entity tool as long as NECESSARY to figure out the modifications.
NEVER call the inspect_entity tool more than 5 times.
"""

    inspect_entity_tool = InspectEntityTool(repo_map)

    add_files_tool, get_modifiable_files_tool, file_edit_tool = get_file_edit_tools(coder)

    # TODO: have the output_handler run existing tests!
    # TODO: have the output handler write a test for the issue and use it to check the fix?
    # add_files_tool.add_files([mod_fname])
    tools = [inspect_entity_tool, file_edit_tool, add_files_tool, get_modifiable_files_tool]

    class BugFixerOutputHandler(MotleyOutputHandler):
        _name = "return_to_user"
        iteration = 0

        def handle_output(self):
            # return "Tests passed!"
            self.iteration += 1
            if self.iteration >= self.max_iterations:
                return "Tests passed!"

            out = existing_test_runner()
            if out is None:
                return "Tests passed!"
            else:
                raise InvalidOutput("Existing tests failed:\n" + out)

    file_finder = ReActToolCallingMotleyAgent(
        name="File-to-fix_finder",
        tools=tools,
        # TODO: is this the right way to inject the fake chat history?
        prompt_prefix=coder.create_prompt("").partial(
            tools=render_text_description(tools)
        ),  # get usage examples as fake chat history
        output_handler=BugFixerOutputHandler(max_iterations=3),
        prompt=BugFixerPrompts().prompt_template_with_output_handler,
        chat_history=True,
        verbose=True,
        max_iterations=15,
    )

    task = SimpleTask(
        crew=crew,
        name="Apply the fixes to the issue",
        description=message,
        agent=file_finder,
    )

    return task
