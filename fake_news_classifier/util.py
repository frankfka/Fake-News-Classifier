from datetime import datetime


# Get a log directory for tensorboard
def get_tb_logdir(unique_name):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    return f"../logs/{unique_name}-{timestamp}"


# Log a message to console
# TODO: Maybe add time
def log(msg, level="INFO", header=False):
    if header:
        print(
            f"""
            ========================================
            {msg}
            ========================================
            """
        )
    else:
        print(f"{level}: {msg}")
