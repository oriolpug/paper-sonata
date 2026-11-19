# Paper Orchestra Frontend

This directory contains the Streamlit frontend for Paper Orchestra.

## How to Launch the Demo Locally

1. Ensure you have activated your environment and installed dependencies.
2. Navigate to the project root directory:
   ```bash
   cd paper-orchestra
   uv sync --extra frontend
   ```
3. Run the Streamlit application:
   ```bash
   uv run streamlit run frontend/app.py
   ```
   (or `streamlit run frontend/app.py` from an activated `.venv`)
4. Open the URL provided in the terminal (usually `http://localhost:8501` or similar) in your browser.

## How to Kill Unfinished Writing Processes

The frontend launches a background process (`paper_writing_cli.py`) to handle the paper generation. 

### Automatic Cleanup
On Linux, the application is configured to automatically kill the background process if the Streamlit server process is stopped. Simply pressing **Ctrl+C** in the terminal where you ran `streamlit run` should clean up the running pipeline.

### Manual Cleanup
If you disconnect from the frontend or the server stops unexpectedly without cleaning up, you may have orphan processes running in the background. You can kill them manually using:

```bash
pkill -f paper_writing_cli.py
```

Or to find the process ID (PID) first:
```bash
pgrep -f paper_writing_cli.py
```
And then kill it:
```bash
kill <PID>
```
