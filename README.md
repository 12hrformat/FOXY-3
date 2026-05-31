# DRAGON-AI

Self-hosted Discord AI chatbot using `discord.py` and a local Ollama model.

## Requirements

- Python 3.10+
- Ollama running locally on `http://localhost:11434`
- A pulled local model, for example:

```bash
ollama pull mistral
```

or:

```bash
ollama pull llama3
```

## Install

```bash
cd /home/dragon/my_projects/DRAGON-AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
export DISCORD_TOKEN="your_discord_bot_token"
export OLLAMA_MODEL="mistral"
export OLLAMA_URL="http://localhost:11434/api/generate"
export BOT_CREATOR="dragon"
```

The bot needs Discord's Message Content Intent enabled in the Discord Developer Portal.

## Run

```bash
python main.py
```

## Data Files

- `brain.json`: full interaction log.
- `memory.json`: manually editable user memory. The bot saves most user messages, filters blocked language, and skips obvious secrets/spam.

Both files are auto-created if missing. If either file becomes corrupted, the bot backs it up as `*.corrupt-YYYYMMDD-HHMMSS.json` and starts a clean file.

## Admin Commands

- `!memory <user>`: show memory for a user id, mention, or username text.
- `!forget <user>`: delete all memory for a user id, mention, or username text.
- `!clearbrain`: wipe `brain.json`.
- `!delmem <index>`: delete one memory entry by index from `!memory`.
