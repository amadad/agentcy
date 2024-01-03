user_proxy = UserProxyAgent(
    name="user_proxy",
    description="Acts as a proxy for the user, capable of executing code and handling user interactions within predefined guidelines.",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1,
    code_execution_config={"work_dir": "/logs"},
    system_message="Use the research and write_content functions as needed to answer the user's questions.",
    function_map={
        "research": research,
        "write_content": write_content,
        "save_markdown": save_markdown,
    }
)

user_proxy.register_function(
    function_map={
        "research": research,
        "write_content": write_content,
        "save_markdown": save_markdown
    }
)