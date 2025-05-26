import asyncio
import os

import aiohttp
import discord
from dotenv import load_dotenv

load_dotenv()


WEBHOOK_URL = (
    os.environ.get("TESTING_WEBHOOK_URL")
    if os.environ.get("ENVIRONMENT") == "local"
    else os.environ.get("LIVE_WEBHOOK_URL")
)

if not WEBHOOK_URL:
    raise ValueError(
        "❌ WEBHOOK_URL is not set. Please check your .env file and environment variables."
    )


async def send_progress_embed(
    progress: float,
    roundsTotal: int,
    currentRound: int,
    eta: str,
    id: str,
    sendSilent: bool = False,
    logger=None,
):
    embed = discord.Embed(
        title="Creating Match Graphs",
        color=discord.Color.green() if progress == 100 else discord.Color.blurple(),
    )

    embed.add_field(
        name="Current Round", value=f"Round {currentRound}/{roundsTotal}", inline=True
    )
    embed.add_field(name="Total Progress", value=f"{progress}/100%", inline=True)

    embed.add_field(name="Estimated finish", value=eta, inline=False)
    filled_blocks = int(progress // 5)
    empty_blocks = 20 - filled_blocks
    bar = "█" * filled_blocks + "░" * empty_blocks
    embed.add_field(name="Progress", value=bar, inline=False)
    embed.add_field(name="Match ID", value=f"{id}", inline=False)

    embed.url = (
        "https://example.com/details"  # Replace with url to the jupyter notebook
    )
    embed.set_thumbnail(
        url="https://as2.ftcdn.net/jpg/05/56/17/61/1000_F_556176185_wmiwJtRkwDEs73iWgGuY0vugaZtV0AzD.jpg"
    )  # Replace with the URL of the thumbnail image
    async with aiohttp.ClientSession() as session:
        webhook = discord.Webhook.from_url(WEBHOOK_URL, session=session)
        await webhook.send(
            embed=embed,
            silent=sendSilent,
            username="Graph Updates",
            avatar_url="https://as2.ftcdn.net/jpg/05/56/17/61/1000_F_556176185_wmiwJtRkwDEs73iWgGuY0vugaZtV0AzD.jpg",
        )

    if logger:
        logger.info(f"✅ Discord Webhook Sent Successfully {id} {progress}%")
    else:
        print(f"✅ Discord Webhook Sent Successfully {id} {progress}%")


# Example: send 0-100 progress


# Send an error embed to Discord
async def send_error_embed(error_message: str, id: str, sendSilent=False, logger=None):
    embed = discord.Embed(
        title="❌ Error Creating Match Graphs",
        description=error_message,
        color=discord.Color.red(),
    )
    embed.add_field(name="Match ID", value=id, inline=False)
    embed.url = (
        "https://example.com/details"  # Optional: link to error logs or notebook
    )
    embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/5368/5368327.png")
    async with aiohttp.ClientSession() as session:
        webhook = discord.Webhook.from_url(WEBHOOK_URL, session=session)
        await webhook.send(embed=embed, silent=sendSilent, username="Graph Updates")

    if logger:
        logger.error(f"❌ Error embed sent: {error_message}")
    else:
        print(f"❌ Error embed sent: {error_message}")


# Send a warning embed to Discord
async def send_warning_embed(
    warning_message: str, id: str, sendSilent=False, logger=None
):
    embed = discord.Embed(
        title="⚠️ Warning During Match Graph Creation",
        description=warning_message,
        color=discord.Color.gold(),
    )
    embed.add_field(name="Match ID", value=id, inline=False)
    embed.url = "https://example.com/details"  # Optional: link to logs or further info
    embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/1538/1538491.png")
    async with aiohttp.ClientSession() as session:
        webhook = discord.Webhook.from_url(WEBHOOK_URL, session=session)
        await webhook.send(embed=embed, silent=sendSilent, username="Graph Updates")
    if logger:
        logger.warning(f"⚠️ Warning embed sent: {warning_message}")
    else:
        print(f"⚠️ Warning embed sent: {warning_message}")


if __name__ == "__main__":

    # Example usage
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        send_progress_embed(
            progress=50,
            roundsTotal=100,
            currentRound=50,
            eta="10 minutes",
            id="12345",
            sendSilent=False,
        )
    )
