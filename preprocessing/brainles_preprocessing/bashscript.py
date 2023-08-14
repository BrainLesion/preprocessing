import subprocess
import platform
import datetime
import time


class ScriptRunner:
    """
    A class for running Bash scripts and capturing output in log files.

    Args:
        script_path (str): Path to the Bash script to be executed.
        log_path (str): Path to the log file where the script output will be saved.

    Attributes:
        script_path (str): Path to the Bash script.
        log_path (str): Path to the log file.
        platform_command (str): Command to execute the script based on the platform.

    Methods:
        run(input_params=None): Execute the script and capture the output in the log file.

    Example:
        # Create an instance of ScriptRunner
        runner = ScriptRunner(script_path, log_path)

        # Specify input parameters
        input_params = ['-param1', 'value1', '-param2', 'value2']

        # Call the run method to execute the script and capture the output in the log file
        runner.run(input_params)
    """

    def __init__(self, script_path, log_path):
        self.script_path = script_path
        self.log_path = log_path
        self.platform_command = "cmd /c" if platform.system() == "Windows" else "bash"

    def run(self, input_params=None):
        """
        Execute the script and capture the output in the log file.

        Args:
            input_params (list, optional): List of input parameters to be passed to the script.
                Defaults to None.

        Returns:
            None
        """
        try:
            start_time = time.time()  # Record the start time
            with open(self.log_path, "a") as log_file:
                script_name = self.script_path.split("/")[
                    -1
                ]  # Extract the script name from the path
                log_file.write(f"\n{'=' * 40}\n")
                log_file.write(f"--- Executing {script_name} ---\n")

                if input_params:
                    log_file.write(f"Input Parameters: {input_params}\n")

                process = subprocess.Popen(
                    [self.platform_command, self.script_path] + input_params,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Function to write a log line with a timestamp
                def write_log_line(stream, line):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{timestamp}] {stream}: {line}")

                # Read and write each event with timestamp to the log file while the process is running
                while process.poll() is None:
                    stdout_line = process.stdout.readline()
                    stderr_line = process.stderr.readline()
                    if stdout_line:
                        write_log_line("stdout", stdout_line)
                    if stderr_line:
                        write_log_line("stderr", stderr_line)

                # Write any remaining output after the process has finished
                for stdout_line in process.stdout.readlines():
                    write_log_line("stdout", stdout_line)
                for stderr_line in process.stderr.readlines():
                    write_log_line("stderr", stderr_line)

                end_time = time.time()  # Record the end time
                total_duration = end_time - start_time
                log_file.write(f"{'=' * 40}\n")
                log_file.write(
                    f"--- Finished {script_name} in {total_duration:.2f} seconds ---\n"
                )

            print("Script executed successfully. Check the log file for details.")
        except subprocess.CalledProcessError as e:
            print("Error executing script:", e)


# Specify the path to the Bash script and the path to the log file
script_path = "script.sh"  # Change this to the actual path of your script
log_path = "script_log.txt"  # Change this to the desired log file path

# Create an instance of ScriptRunner
runner = ScriptRunner(script_path, log_path)

# Specify input parameters
input_params = [
    "-param1",
    "value1",
    "-param2",
    "value2",
]  # Modify this with your input parameters

# Call the run method to execute the script and capture the output in the log file
runner.run(input_params)
