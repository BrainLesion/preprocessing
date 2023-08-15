import subprocess
import platform
import datetime
import time
import os


class ScriptRunner:
    """
    A class for running Bash scripts and capturing output in log files.

    Args:
        script_path (str): Path to the Bash script to be executed.
        log_path (str): Path to the log file where the script output will be saved.

    Methods:
        run(input_params=None): Execute the script and capture the output in the log file.

    Example:
        # Create an instance of ScriptRunner
        runner = ScriptRunner(script_path, log_path)

        # Specify input parameters
        input_params = ['-param1', 'value1', '-param2', 'value2']

        # Call the run method to execute the script and capture the output in the log file
        success, error = runner.run(input_params)
        if success:
            print("Script executed successfully. Check the log file for details.")
        else:
            print("Script execution failed:", error)
    """

    def __init__(self, script_path, log_path):
        self.runner_path = os.path.dirname(os.path.abspath(__file__))
        self.script_path = os.path.abspath(script_path)
        self.log_path = log_path
        self.platform_command = "cmd /c" if platform.system() == "Windows" else "bash"

    def _write_log_line(self, stream, line, log_file):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {stream}: {line}")

    def run(self, input_params=None):
        """
        Execute the script and capture the output in the log file.

        Args:
            input_params (list, optional): List of input parameters to be passed to the script.
                Defaults to None.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean (True if script executed successfully, False otherwise)
                             and an error message (empty string if no error occurred).
        """
        try:
            start_time = time.time()
            with open(self.log_path, "a") as log_file:
                script_name = self.script_path.split("/")[-1]
                log_file.write(f"\n{'=' * 80}\n")
                log_file.write(f"--- Executing {script_name} ---\n")
                log_file.write(f"\n{'=' * 80}\n")

                if input_params:
                    log_file.write(f"Input Parameters: {input_params}\n")

                process = subprocess.Popen(
                    [self.platform_command, self.script_path] + input_params,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                while process.poll() is None:
                    stdout_line = process.stdout.readline()
                    stderr_line = process.stderr.readline()
                    if stdout_line:
                        self._write_log_line("stdout", stdout_line, log_file)
                    if stderr_line:
                        self._write_log_line("stderr", stderr_line, log_file)

                for stdout_line in process.stdout.readlines():
                    self._write_log_line("stdout", stdout_line, log_file)
                for stderr_line in process.stderr.readlines():
                    self._write_log_line("stderr", stderr_line, log_file)

                end_time = time.time()
                total_duration = end_time - start_time
                log_file.write(f"{'=' * 80}\n")
                log_file.write(
                    f"--- Finished {script_name} in {total_duration:.2f} seconds ---\n"
                )
                log_file.write(f"\n{'=' * 80}\n")

            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"Error executing script: {e}"


if __name__ == "__main__":
    # TODO
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
    success, error = runner.run(input_params)
    if success:
        print("Script executed successfully. Check the log file for details.")
    else:
        print("Script execution failed:", error)
