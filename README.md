# Advanced Premium Trading Bot

A sophisticated trading bot with advanced features including risk management, multiple strategies, and Telegram notifications.

## Features

- Multiple trading strategies (Scalping, Breakout, Pullback)
- Advanced risk management
- Real-time market analysis
- Telegram notifications with:
  - Trading quotes
  - Market insights
  - Trading jokes and memes
  - Daily trivia
  - Trading terms of the day
  - Performance updates
  - Market condition updates

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your Telegram bot:
   - Create a bot using BotFather on Telegram
   - Get your bot token and chat ID
   - The bot will use these credentials automatically

## Usage

1. Start the bot:
   ```bash
   python run_bot.py
   ```

2. The bot will:
   - Initialize all components
   - Start monitoring the market
   - Send Telegram notifications
   - Execute trades based on strategies
   - Log all activities to bot.log

## Configuration

The bot comes with default configuration for:
- Trading pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, EURJPY, GBPJPY
- Risk parameters
- Trading hours
- Technical indicators

To modify the configuration, edit the `create_bot()` function in `advanced_premium_bot.py`.

## Logging

The bot logs all activities to `bot.log` and the console. Logs include:
- Trade executions
- Strategy signals
- Risk management events
- Error messages
- Performance metrics

## Telegram Notifications

You will receive:
- Trading quotes (hourly)
- Market insights (every 30 minutes)
- Trading jokes (every 2 hours)
- Trading memes (every 3 hours)
- Trading trivia (every 4 hours)
- Trading term of the day (daily)
- Trade notifications (real-time)
- Performance updates (daily)
- Market condition updates (periodic)

## Support

For issues or questions, please check the logs and documentation. 