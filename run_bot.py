import asyncio
import logging
from advanced_premium_bot import (
    create_bot,
    CurrencyPairManager,
    RiskManager,
    StrategyManager,
    OrderRouter,
    TelegramNotifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

async def main():
    try:
        # Create and initialize the bot
        bot = create_bot()
        logging.info("Bot created successfully")
        
        # Start the bot
        await bot.run()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        logging.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {e}") 