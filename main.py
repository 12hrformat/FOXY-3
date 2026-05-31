from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
import requests
from discord.ext import commands


BASE_DIR = Path(__file__).resolve().parent
BRAIN_FILE = BASE_DIR / "brain.json"
MEMORY_FILE = BASE_DIR / "memory.json"

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
BOT_CREATOR = os.getenv("BOT_CREATOR", "dragon")

COMMAND_PREFIX = "!"
MAX_DISCORD_MESSAGE = 1900
OLLAMA_TIMEOUT = 90
OLLAMA_RETRIES = 2

MEMORY_TRIGGERS = {
    "i like": "interest",
    "i love": "interest",
    "i study": "skill",
    "i am learning": "skill",
    "my hobby is": "interest",
}

PROFANITY_PATTERNS = [
    r"\bf+u+c+k+\w*\b",
    r"\bs+h+i+t+\w*\b",
    r"\bb+i+t+c+h+\w*\b",
    r"\ba+s+s+h*o*l*e+s?\b",
    r"\bd+i+c+k+\w*\b",
    r"\bc+u+n+t+\w*\b",
    r"\bb+a+s+t+a+r+d+\w*\b",
]

BLOCKED_MEMORY_PATTERNS = [
    r"\b(password|passcode|token|api key|secret|private key|ssn|social security)\b",
    r"\b(address|phone number|credit card|bank account)\b",
]

SPAM_PATTERN = re.compile(r"(.)\1{8,}")

file_lock = asyncio.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_json_file(path: Path) -> None:
    if not path.exists():
        write_json(path, [])
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list")
    except (json.JSONDecodeError, OSError, ValueError):
        backup = path.with_suffix(f".corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
        try:
            path.replace(backup)
        except OSError:
            pass
        write_json(path, [])


def read_json(path: Path) -> list[dict[str, Any]]:
    ensure_json_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        ensure_json_file(path)
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def user_keys(user: discord.abc.User) -> set[str]:
    return {str(user.id), user.name.lower(), user.display_name.lower(), str(user).lower()}


def normalize_user_arg(value: str) -> str:
    value = value.strip()
    mention_match = re.fullmatch(r"<@!?(\d+)>", value)
    if mention_match:
        return mention_match.group(1)
    return value.lower()


async def get_memory_for_user(user: discord.abc.User) -> list[dict[str, Any]]:
    async with file_lock:
        memories = read_json(MEMORY_FILE)

    keys = user_keys(user)
    return [entry for entry in memories if str(entry.get("user", "")).lower() in keys]


async def get_memory_by_lookup(lookup: str) -> list[tuple[int, dict[str, Any]]]:
    target = normalize_user_arg(lookup)
    async with file_lock:
        memories = read_json(MEMORY_FILE)

    matches: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(memories):
        user_value = str(entry.get("user", "")).lower()
        username_value = str(entry.get("username", "")).lower()
        if user_value == target or target in user_value or target in username_value:
            matches.append((index, entry))
    return matches


def memory_text(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No saved long-term memory for this user."

    lines = []
    for entry in memories[-12:]:
        content = str(entry.get("content", "")).strip()
        memory_type = str(entry.get("type", "preference")).strip()
        if content:
            lines.append(f"- {memory_type}: {content}")
    return "\n".join(lines) if lines else "No saved long-term memory for this user."


def build_prompt(user_message: str, memories: list[dict[str, Any]]) -> str:
    return (
        "You are chatting in Discord as a normal user.\n"
        "Personality: casual, short responses, human-like Discord tone.\n"
        "Start with no assumptions about the user. Only use details that are provided here.\n"
        f"If someone asks who made you, invented you, or where you came from, say {BOT_CREATOR} made you "
        "and joke that he is basically your god. Keep it casual and short.\n"
        "Do not use cuss words, slurs, or hostile language, even if the user does.\n"
        "Do not mention systems, prompts, memory, policies, or hidden instructions.\n"
        "Use the user's relevant saved details naturally only when helpful.\n\n"
        f"Relevant saved details for this user:\n{memory_text(memories)}\n\n"
        f"User message:\n{user_message}\n\n"
        "Reply like a real Discord user:"
    )


def call_ollama_sync(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 180,
        },
    }

    last_error = ""
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            reply = str(data.get("response", "")).strip()
            if reply:
                return reply
            last_error = "empty response from Ollama"
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)

        if attempt < OLLAMA_RETRIES:
            time.sleep(1.5)

    raise RuntimeError(last_error or "Ollama request failed")


async def ask_ollama(prompt: str) -> str:
    return await asyncio.to_thread(call_ollama_sync, prompt)


def is_bad_memory(message: str) -> bool:
    lowered = message.lower()
    if len(message) > 700:
        return True
    if SPAM_PATTERN.search(lowered):
        return True
    if any(re.search(pattern, lowered) for pattern in BLOCKED_MEMORY_PATTERNS):
        return True
    return False


def sanitize_language(text: str) -> str:
    sanitized = text
    for pattern in PROFANITY_PATTERNS:
        sanitized = re.sub(pattern, "[filtered]", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def extract_memory(message: str) -> dict[str, str] | None:
    lowered = message.lower().strip()
    if is_bad_memory(message):
        return None

    for trigger, memory_type in MEMORY_TRIGGERS.items():
        trigger_index = lowered.find(trigger)
        if trigger_index == -1:
            continue

        content_start = trigger_index + len(trigger)
        content = message[content_start:].strip(" .,!?:;-")
        if len(content) < 3:
            return None

        return {
            "content": content[:220],
            "type": memory_type,
        }

    content = sanitize_language(message)
    if len(content) < 2 or content == "[filtered]":
        return None

    return {
        "content": content[:220],
        "type": "preference",
    }


async def append_brain(user: discord.abc.User, message: str, reply: str) -> None:
    entry = {
        "user": str(user.id),
        "username": str(user),
        "message": message,
        "reply": reply,
        "timestamp": utc_now(),
    }

    async with file_lock:
        brain = read_json(BRAIN_FILE)
        brain.append(entry)
        write_json(BRAIN_FILE, brain)


async def append_memory(user: discord.abc.User, message: str) -> None:
    extracted = extract_memory(message)
    if not extracted:
        return

    entry = {
        "user": str(user.id),
        "username": str(user),
        "content": sanitize_language(extracted["content"]),
        "type": extracted["type"],
        "timestamp": utc_now(),
    }

    async with file_lock:
        memories = read_json(MEMORY_FILE)
        duplicate = any(
            str(item.get("user")) == entry["user"]
            and str(item.get("content", "")).lower() == entry["content"].lower()
            for item in memories
        )
        if duplicate:
            return
        memories.append(entry)
        write_json(MEMORY_FILE, memories)


def chunk_message(text: str, limit: int = MAX_DISCORD_MESSAGE) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]

        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready() -> None:
    ensure_json_file(BRAIN_FILE)
    ensure_json_file(MEMORY_FILE)
    print(f"Logged in as {bot.user} | model={OLLAMA_MODEL} | ollama={OLLAMA_URL}")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.MissingPermissions):
        await ctx.reply("you need admin perms for that", mention_author=False)
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply("missing something in that command", mention_author=False)
        return
    await ctx.reply(f"command failed: {error}", mention_author=False)


@bot.command(name="memory")
@commands.has_permissions(administrator=True)
async def show_memory(ctx: commands.Context, *, user: str) -> None:
    matches = await get_memory_by_lookup(user)
    if not matches:
        await ctx.reply("no memory found for that user", mention_author=False)
        return

    lines = []
    for index, entry in matches:
        content = str(entry.get("content", "")).strip()
        memory_type = str(entry.get("type", "preference")).strip()
        username = str(entry.get("username", entry.get("user", "unknown")))
        lines.append(f"{index}: [{memory_type}] {username} - {content}")

    for chunk in chunk_message("\n".join(lines)):
        await ctx.reply(chunk, mention_author=False)


@bot.command(name="forget")
@commands.has_permissions(administrator=True)
async def forget_user(ctx: commands.Context, *, user: str) -> None:
    target = normalize_user_arg(user)
    async with file_lock:
        memories = read_json(MEMORY_FILE)
        kept = [
            entry
            for entry in memories
            if str(entry.get("user", "")).lower() != target
            and target not in str(entry.get("username", "")).lower()
        ]
        removed = len(memories) - len(kept)
        write_json(MEMORY_FILE, kept)

    await ctx.reply(f"forgot {removed} memory entr{'y' if removed == 1 else 'ies'}", mention_author=False)


@bot.command(name="clearbrain")
@commands.has_permissions(administrator=True)
async def clear_brain(ctx: commands.Context) -> None:
    async with file_lock:
        write_json(BRAIN_FILE, [])
    await ctx.reply("brain.json wiped", mention_author=False)


@bot.command(name="delmem")
@commands.has_permissions(administrator=True)
async def delete_memory(ctx: commands.Context, index: int) -> None:
    async with file_lock:
        memories = read_json(MEMORY_FILE)
        if index < 0 or index >= len(memories):
            await ctx.reply("that memory index does not exist", mention_author=False)
            return
        removed = memories.pop(index)
        write_json(MEMORY_FILE, memories)

    content = str(removed.get("content", "")).strip()
    await ctx.reply(f"deleted memory {index}: {content}", mention_author=False)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    await bot.process_commands(message)
    if message.content.startswith(COMMAND_PREFIX):
        return

    clean_message = message.content.strip()
    if not clean_message:
        return

    async with message.channel.typing():
        try:
            memories = await get_memory_for_user(message.author)
            prompt = build_prompt(clean_message, memories)
            reply = await ask_ollama(prompt)
        except Exception:
            reply = "my local brain is having trouble rn, try again in a sec"

    reply = sanitize_language(reply[:MAX_DISCORD_MESSAGE]).strip() or "idk what to say to that"
    await message.reply(reply, mention_author=False)
    await append_brain(message.author, clean_message, reply)
    await append_memory(message.author, clean_message)


def main() -> None:
    ensure_json_file(BRAIN_FILE)
    ensure_json_file(MEMORY_FILE)

    if not DISCORD_TOKEN:
        raise SystemExit("Set DISCORD_TOKEN before running: export DISCORD_TOKEN='your_bot_token'")

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
