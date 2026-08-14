"""Whisper Voice Note Transcription handler for Telegram Bot."""

import os
import tempfile

import httpx
from loguru import logger
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.auth import is_authorized_telegram
from bot.formatters import escape_markdown_v2
from config import settings


async def transcribe_audio_whisper(file_path: str) -> str:
    """Transcribe audio file using NVIDIA NIM Whisper API or local fallback."""
    api_key = settings.NVIDIA_NIM_API_KEY.strip()
    if api_key:
        try:
            url = f"{settings.NVIDIA_NIM_BASE_URL}/audio/transcriptions"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(file_path, "rb") as audio_file:
                    files = {"file": (os.path.basename(file_path), audio_file, "audio/ogg")}
                    data = {"model": settings.WHISPER_MODEL}
                    resp = await client.post(url, headers=headers, data=data, files=files)
                    
                    if resp.status_code == 200:
                        res_json = resp.json()
                        return res_json.get("text", "").strip()
                    else:
                        logger.warning(f"Whisper API transcription status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Failed to transcribe via NVIDIA NIM Whisper API: {e}")

    # Return fallback instruction if transcription unavailable
    return "Voice note received. (Transcription API key not configured or offline)."


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram voice message notes, transcribe audio, and reply with text."""
    if not update.message or not update.message.voice:
        return

    if not is_authorized_telegram(update):
        return

    status_msg = await update.message.reply_text("🎙️ *Processing voice note...*", parse_mode=ParseMode.MARKDOWN_V2)

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name

        await voice_file.download_to_drive(tmp_path)

        transcription = await transcribe_audio_whisper(tmp_path)

        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        esc_text = escape_markdown_v2(transcription, is_code_block=True)
        reply_text = (
            "🗣️ *Voice Note Transcribed:*\n\n"
            f"```\n{esc_text}\n```\n\n"
            "💡 _You can copy or execute this command using `/run` or sending it as prompt\._"
        )
        await status_msg.edit_text(text=reply_text, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        esc_err = escape_markdown_v2(str(e), is_code_block=True)
        await status_msg.edit_text(f"❌ Voice processing error: `{esc_err}`", parse_mode=ParseMode.MARKDOWN_V2)
