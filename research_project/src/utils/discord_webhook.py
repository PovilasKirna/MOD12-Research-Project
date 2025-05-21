import asyncio
import random
import time
from datetime import datetime, timedelta

import aiohttp
import discord

WEBHOOK_URL = "https://discord.com/api/webhooks/1374409513947627570/1J-3R2ifsxisq0bfe0j4b--aBTaNwSxgruiXOVXlas2lL1Jd4IfvANgcsHrwTvzcl7qW"


async def send_progress_embed(
    progress: float,
    roundsTotal: int,
    currentRound: int,
    eta: str,
    id: str,
    sendSilent: bool = False,
):
    embed = discord.Embed(
        title="Creating Match Graphs",
        color=discord.Color.green() if progress == 100 else discord.Color.blurple(),
    )

    embed.add_field(
        name="Current Round", value=f"Round {currentRound}/{roundsTotal}", inline=True
    )
    embed.add_field(name="Total Progress", value=f"{progress}/100%", inline=True)

    embed.add_field(name="Estimated time left", value=eta, inline=False)
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


# Example: send 0-100 progress


# Send an error embed to Discord
async def send_error_embed(error_message: str, id: str, sendSilent=False):
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


# Send a warning embed to Discord
async def send_warning_embed(warning_message: str, id: str, sendSilent=False):
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


def calculate_eta(
    start_time: float, current_round: int, rounds_framecount: list[int]
) -> str:
    if current_round == 0:
        return "Calculating ETA..."
    if current_round >= len(rounds_framecount):
        return "All rounds completed."

    elapsed = time.time() - start_time
    processed_frames = sum(rounds_framecount[:current_round])

    total_frames = sum(rounds_framecount)

    estimated_total_time = (elapsed / processed_frames) * total_frames

    remaining_time = estimated_total_time - elapsed
    eta = timedelta(seconds=max(0, int(remaining_time)))
    finish_time = datetime.now() + eta
    return f"Estimated finish: {finish_time.strftime('%H:%M')} (ETA: {str(eta)})"


async def run_fake_progress_test():
    start_time = time.time()
    roundsTotal = 28
    rounds_framecount = [
        random.randint(150, 300) for _ in range(roundsTotal)
    ]  # Random frame counts for each round e.g [150, 200, 250, ...]

    print("Starting progress test...")
    print(f"Total rounds: {roundsTotal}")
    print(f"Rounds framecount: {rounds_framecount}")
    print(
        f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    for i in range(roundsTotal + 1):
        progress = round(i / roundsTotal * 100, 2)
        eta = calculate_eta(
            start_time=start_time,
            current_round=i,
            rounds_framecount=rounds_framecount,
        )
        await send_progress_embed(
            progress=progress,
            roundsTotal=roundsTotal,
            currentRound=i,
            eta=eta,
            id="test_match_123",
            sendSilent=i
            not in [0, roundsTotal],  # Send silent for first and last round
        )
        await asyncio.sleep(0.1)  # Simulate time taken for each round


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_fake_progress_test())
