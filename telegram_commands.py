from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
import logging
from advanced_premium_bot import AdvancedPremiumBot
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global bot instance
trading_bot = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the trading bot"""
    global trading_bot
    
    if trading_bot and trading_bot.running:
        await update.message.reply_text("Bot is already running!")
        return
        
    try:
        trading_bot = AdvancedPremiumBot()
        trading_bot.run()
        await update.message.reply_text("✅ Trading bot started successfully!")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        await update.message.reply_text(f"❌ Error starting bot: {str(e)}")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the trading bot"""
    global trading_bot
    
    if not trading_bot or not trading_bot.running:
        await update.message.reply_text("Bot is not running!")
        return
        
    try:
        trading_bot.shutdown()
        await update.message.reply_text("🛑 Trading bot stopped successfully!")
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        await update.message.reply_text(f"❌ Error stopping bot: {str(e)}")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get bot status"""
    global trading_bot
    
    if not trading_bot:
        await update.message.reply_text("Bot is not initialized!")
        return
        
    try:
        status = "🟢 Running" if trading_bot.running else "🔴 Stopped"
        positions = len(trading_bot.positions)
        performance = trading_bot.performance_metrics
        
        message = (
            f"🤖 Bot Status: {status}\n"
            f"📊 Open Positions: {positions}\n"
            f"💰 Performance:\n"
            f"  - Total Trades: {performance.get('total_trades', 0)}\n"
            f"  - Win Rate: {performance.get('win_rate', 0):.2f}%\n"
            f"  - Total P/L: {performance.get('total_pl', 0):.2f}"
        )
        
        await update.message.reply_text(message)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        await update.message.reply_text(f"❌ Error getting status: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help message"""
    help_text = (
        "🤖 Trading Bot Commands:\n\n"
        "/start - Start the trading bot\n"
        "/stop - Stop the trading bot\n"
        "/status - Get bot status\n"
        "/help - Show this help message"
    )
    await update.message.reply_text(help_text)

def main():
    """Start the Telegram bot"""
    # Get bot token from environment variable
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return
        
    # Create application
    application = Application.builder().token(token).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Start the bot
    logger.info("Starting Telegram bot...")
    application.run_polling()

if __name__ == '__main__':
    main() 