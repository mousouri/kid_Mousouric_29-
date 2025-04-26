import asyncio
import websockets
import json
from advanced_premium_bot import AdvancedPremiumBot
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotWebSocket:
    def __init__(self):
        self.bot = AdvancedPremiumBot()
        self.connected_clients = set()

    async def handle_client(self, websocket, path):
        try:
            self.connected_clients.add(websocket)
            logger.info(f"New client connected. Total clients: {len(self.connected_clients)}")

            # Send initial data
            initial_data = {
                "type": "init",
                "data": {
                    "account": {
                        "balance": self.bot.performance_stats.get('balance', 0),
                        "equity": self.bot.performance_stats.get('equity', 0),
                        "margin": self.bot.performance_stats.get('margin', 0),
                        "free_margin": self.bot.performance_stats.get('free_margin', 0)
                    },
                    "performance": self.bot.performance_stats,
                    "open_trades": self.bot.get_open_trades()
                }
            }
            await websocket.send(json.dumps(initial_data))

            # Keep connection alive and send updates
            while True:
                # Get latest data
                price_data = self.bot.get_latest_prices()
                trade_updates = self.bot.get_trade_updates()
                
                if price_data or trade_updates:
                    update_data = {
                        "type": "update",
                        "data": {
                            "prices": price_data,
                            "trades": trade_updates
                        }
                    }
                    await websocket.send(json.dumps(update_data))
                
                await asyncio.sleep(1)  # Update every second

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.connected_clients.remove(websocket)

    async def start_server(self, host="localhost", port=8001):
        server = await websockets.serve(self.handle_client, host, port)
        logger.info(f"WebSocket server started on ws://{host}:{port}")
        
        # Initialize the bot
        if not self.bot.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        # Start the bot
        await self.bot.run()
        
        await server.wait_closed()

if __name__ == "__main__":
    bot_ws = BotWebSocket()
    asyncio.run(bot_ws.start_server()) 