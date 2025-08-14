def get_task(task_type):
    if task_type == "codegen":
        from .codegen import CodeGenTask
        return CodeGenTask
    elif task_type == "coderepair":
        from .coderepair import CodeRepairTask
        return CodeRepairTask
    elif task_type == "testgen":
        from .testgen import TestGenTask
        return TestGenTask
    else:
        raise ValueError(f"Unknown task type: {task_type}")
