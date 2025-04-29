# Advanced Trading Bot with Telegram Control

A sophisticated trading bot that can be controlled via Telegram commands. The bot supports multiple trading strategies including scalping, trend following, and more.

## Features

- Telegram-based control interface
- Multiple trading strategies (scalping, trend following, etc.)
- Real-time market monitoring
- Performance tracking
- Risk management
- Position sizing optimization

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Telegram Bot**
   - Open Telegram and search for @BotFather
   - Send `/newbot` command
   - Follow instructions to create your bot
   - Save the bot token provided by BotFather

3. **Get Your Chat ID**
   - Open Telegram and search for @userinfobot
   - Send any message to get your chat ID

4. **Create .env File**
   Create a `.env` file in the project root with the following content:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   TRADING_SYMBOLS=EURUSD,GBPUSD,USDJPY
   ```

5. **Start the Bot**
   ```bash
   python telegram_commands.py
   ```

## Telegram Commands

- `/start` - Start the trading bot
- `/stop` - Stop the trading bot
- `/status` - Get current bot status and performance
- `/help` - Show available commands

## Bot Status Messages

The bot will send you notifications for:
- Bot startup and shutdown
- Trade openings and closings
- Errors and warnings
- Performance updates

## Requirements

- Python 3.8 or higher
- MetaTrader 5 installed and running
- Active internet connection
- Telegram account

## Security Notes

- Keep your `.env` file secure and never share it
- Only share your bot token with trusted individuals
- Monitor your bot's activity regularly
- Set appropriate risk parameters in the configuration

## Support

For issues or questions, please open an issue in the repository. 