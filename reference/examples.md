# Examples
## EASY
### Ask about a YouTube video
uv run insights --app-dir "$INSIGHTS_APP_DIR" "ask https://www.youtube.com/watch?v=zPMPqzjM0Fw what is this about?"

### Ingest a URL
uv run insights --app-dir "$INSIGHTS_APP_DIR" "ingest https://www.youtube.com/watch?v=zPMPqzjM0Fw"

### Start chat on a source
uv run insights --app-dir "$INSIGHTS_APP_DIR" "chat https://www.youtube.com/watch?v=zPMPqzjM0Fw"

## MID
### Show conversations for a source
uv run insights --app-dir "$INSIGHTS_APP_DIR" "show conversations for https://www.youtube.com/watch?v=zPMPqzjM0Fw"

## HARD
### Resume a conversation about a topic for a given source
uv run insights --app-dir "$INSIGHTS_APP_DIR" "resume conversation https://www.youtube.com/watch?v=zPMPqzjM0Fw about best IDE to use"

uv run insights --app-dir "$INSIGHTS_APP_DIR" "resume last conversation on https://www.youtube.com/watch?v=zPMPqzjM0Fw"

### Check if a specific source exists
uv run insights --app-dir "$INSIGHTS_APP_DIR" "check if I have this source https://www.youtube.com/watch?v=zPMPqzjM0Fw"

### Search sources by topic and type (YouTube/PDF)
uv run insights --app-dir "$INSIGHTS_APP_DIR" "check if I have a youtube video about a user that cant stand using IDEs anymore"

uv run insights --app-dir "$INSIGHTS_APP_DIR" "check if I have a pdf doc about a project of agentic AI with ViveBioTech"