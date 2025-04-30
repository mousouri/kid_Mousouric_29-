import logging
import random
import time
from typing import Dict
from telegram import Application

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Application.builder().token(bot_token).build()
        self.chat_id = chat_id
        
        # Trading content
        self.quotes = [
            "The market is a device for transferring money from the active to the patient. - Warren Buffett",
            "Risk comes from not knowing what you're doing. - Warren Buffett",
            "The stock market is filled with individuals who know the price of everything, but the value of nothing. - Philip Fisher",
            "The four most dangerous words in investing are: 'this time it's different.' - Sir John Templeton",
            "The individual investor should act consistently as an investor and not as a speculator. - Ben Graham",
            "In investing, what is comfortable is rarely profitable. - Robert Arnott",
            "The market, like the Lord, helps those who help themselves. - Andre Kostolany",
            "The goal of a successful trader is to make the best trades. Money is secondary. - Alexander Elder",
            "The best way to measure your investing success is not by whether you're beating the market but by whether you've put in place a financial plan and a behavioral discipline that are likely to get you where you want to go. - Benjamin Graham"
        ]
        
        self.market_insights = [
            "Market volatility often presents the best opportunities for disciplined traders.",
            "Remember: Trends are your friends, but they don't last forever.",
            "The most profitable trades often come from the most uncomfortable entries.",
            "Patience is not just a virtue in trading, it's a necessity.",
            "Successful trading is about consistency, not home runs.",
            "The market rewards those who can control their emotions.",
            "Every trade is a new opportunity, but not every opportunity should be a trade.",
            "Risk management is not just about protecting capital, it's about preserving opportunity.",
            "The best traders are those who can adapt to changing market conditions.",
            "Trading success comes from a combination of strategy, discipline, and emotional control."
        ]
        
        # Trading jokes
        self.trading_jokes = [
            "Why did the trader break up with his girlfriend? Because she kept setting stop losses on their relationship!",
            "What do you call a trader who's always late? A stop-loss!",
            "Why did the trader go broke? He kept buying the dip... and the dip kept dipping!",
            "What's a trader's favorite type of music? Heavy Metal!",
            "Why did the trader bring a ladder to the trading floor? To reach new highs!",
            "What do you call a trader who's always wrong? A contrarian indicator!",
            "Why did the trader get kicked out of the casino? He kept asking for a Fibonacci retracement!",
            "What's a trader's favorite exercise? Pull-ups!",
            "Why did the trader get a dog? For the bull market!",
            "What do you call a trader who's always early? A premature entry!"
        ]
        
        # Trading memes (as text descriptions)
        self.trading_memes = [
            "📈 When you buy the dip and it keeps dipping\n\n#TradingMeme #BuyTheDip",
            "😅 When your stop loss gets hit and price immediately reverses\n\n#TradingMeme #StopLoss",
            "🤡 When you try to catch a falling knife\n\n#TradingMeme #Rekt",
            "💎 When you hold through a drawdown and it pays off\n\n#TradingMeme #DiamondHands",
            "📉 When you sell and price immediately rockets\n\n#TradingMeme #PaperHands",
            "🤔 When you see a perfect setup but your account is empty\n\n#TradingMeme #NoFunds",
            "😎 When you nail a perfect entry\n\n#TradingMeme #PerfectEntry",
            "😱 When you check your account after a weekend gap\n\n#TradingMeme #GapRisk",
            "💪 When you resist the urge to revenge trade\n\n#TradingMeme #Discipline",
            "🎯 When you hit your daily profit target\n\n#TradingMeme #ProfitTarget"
        ]
        
        # Trading trivia/questions
        self.trading_trivia = [
            {
                "question": "What is the most traded currency pair in the world?",
                "answer": "EUR/USD",
                "explanation": "EUR/USD accounts for about 24% of all forex trading volume."
            },
            {
                "question": "What does 'ATR' stand for in trading?",
                "answer": "Average True Range",
                "explanation": "ATR is a technical indicator that measures market volatility."
            },
            {
                "question": "What is the 'Golden Cross' in technical analysis?",
                "answer": "When the 50-day moving average crosses above the 200-day moving average",
                "explanation": "It's considered a bullish signal indicating potential upward momentum."
            },
            {
                "question": "What is the 'Death Cross' in technical analysis?",
                "answer": "When the 50-day moving average crosses below the 200-day moving average",
                "explanation": "It's considered a bearish signal indicating potential downward momentum."
            },
            {
                "question": "What is the 'Fibonacci Retracement' based on?",
                "answer": "The Fibonacci sequence of numbers",
                "explanation": "It uses key ratios (23.6%, 38.2%, 50%, 61.8%, 78.6%) to identify potential support and resistance levels."
            }
        ]
        
        # Trading terms of the day
        self.trading_terms = [
            {
                "term": "Liquidity",
                "definition": "The degree to which an asset can be quickly bought or sold without affecting its price.",
                "example": "Major currency pairs like EUR/USD have high liquidity, while exotic pairs have lower liquidity."
            },
            {
                "term": "Volatility",
                "definition": "A statistical measure of the dispersion of returns for a given security or market index.",
                "example": "High volatility means the price can change dramatically in a short time period."
            },
            {
                "term": "Spread",
                "definition": "The difference between the bid and ask price of a currency pair.",
                "example": "If EUR/USD bid is 1.1000 and ask is 1.1002, the spread is 2 pips."
            },
            {
                "term": "Pip",
                "definition": "The smallest price move that a given exchange rate can make.",
                "example": "If EUR/USD moves from 1.1000 to 1.1001, it has moved 1 pip."
            },
            {
                "term": "Leverage",
                "definition": "The use of borrowed capital to increase the potential return of an investment.",
                "example": "With 100:1 leverage, you can control $100,000 with $1,000 of capital."
            }
        ]
        
        self.last_quote_time = None
        self.last_insight_time = None
        self.last_joke_time = None
        self.last_meme_time = None
        self.last_trivia_time = None
        self.last_term_time = None
        
        self.quote_interval = 3600  # 1 hour
        self.insight_interval = 1800  # 30 minutes
        self.joke_interval = 7200  # 2 hours
        self.meme_interval = 10800  # 3 hours
        self.trivia_interval = 14400  # 4 hours
        self.term_interval = 86400  # 24 hours
        
    async def send_quote(self) -> None:
        """Send a random trading quote"""
        try:
            if (self.last_quote_time is None or 
                time.time() - self.last_quote_time > self.quote_interval):
                quote = random.choice(self.quotes)
                message = f"📊 *Trading Wisdom*\n\n{quote}\n\n#TradingQuote #MarketWisdom"
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                self.last_quote_time = time.time()
        except Exception as e:
            logging.error(f"Error sending quote: {e}")
            
    async def send_market_insight(self) -> None:
        """Send a market insight"""
        try:
            if (self.last_insight_time is None or 
                time.time() - self.last_insight_time > self.insight_interval):
                insight = random.choice(self.market_insights)
                message = f"💡 *Market Insight*\n\n{insight}\n\n#MarketInsight #TradingTips"
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                self.last_insight_time = time.time()
        except Exception as e:
            logging.error(f"Error sending market insight: {e}")
            
    async def send_trade_notification(self, trade_info: Dict) -> None:
        """Send a trade notification with emojis and formatting"""
        try:
            direction_emoji = "🟢" if trade_info['direction'] == 'buy' else "🔴"
            profit_emoji = "💰" if trade_info['profit'] > 0 else "💸"
            
            message = (
                f"{direction_emoji} *New Trade Executed*\n\n"
                f"*Symbol:* {trade_info['symbol']}\n"
                f"*Direction:* {trade_info['direction'].upper()}\n"
                f"*Entry Price:* {trade_info['entry_price']:.5f}\n"
                f"*Position Size:* {trade_info['position_size']:.2f}\n"
                f"*Stop Loss:* {trade_info['stop_loss']:.5f}\n"
                f"*Take Profit:* {trade_info['take_profit']:.5f}\n"
                f"*Confidence:* {trade_info['confidence']:.2%}\n\n"
                f"{profit_emoji} *Profit:* {trade_info['profit']:.2f}\n"
                f"*Risk/Reward:* {trade_info['risk_reward']:.2f}\n\n"
                f"#TradeAlert #{trade_info['symbol']} #{trade_info['direction']}"
            )
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logging.error(f"Error sending trade notification: {e}")
            
    async def send_performance_update(self, performance: Dict) -> None:
        """Send a performance update with charts and stats"""
        try:
            message = (
                "📈 *Daily Performance Update*\n\n"
                f"*Total Trades:* {performance['total_trades']}\n"
                f"*Win Rate:* {performance['win_rate']:.2%}\n"
                f"*Total Profit:* {performance['total_profit']:.2f}\n"
                f"*Average Profit:* {performance['avg_profit']:.2f}\n"
                f"*Max Drawdown:* {performance['max_drawdown']:.2%}\n\n"
                f"*Best Pair:* {performance['best_pair']}\n"
                f"*Worst Pair:* {performance['worst_pair']}\n\n"
                "#PerformanceUpdate #TradingStats"
            )
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logging.error(f"Error sending performance update: {e}")
            
    async def send_market_condition_update(self, conditions: Dict) -> None:
        """Send a market condition update"""
        try:
            volatility_emoji = "🌊" if conditions['volatility'] > 0.5 else "🌊" if conditions['volatility'] > 0.3 else "🌊"
            trend_emoji = "📈" if conditions['trend'] == 'up' else "📉" if conditions['trend'] == 'down' else "➡️"
            
            message = (
                f"{volatility_emoji} *Market Conditions Update*\n\n"
                f"*Volatility:* {conditions['volatility']:.2%}\n"
                f"*Trend:* {trend_emoji} {conditions['trend'].upper()}\n"
                f"*Volume:* {conditions['volume']:.2%} of average\n"
                f"*Active Sessions:* {', '.join(conditions['active_sessions'])}\n\n"
                f"*Best Pairs to Trade:*\n"
            )
            
            for pair in conditions['best_pairs'][:3]:
                message += f"• {pair}\n"
                
            message += "\n#MarketUpdate #TradingConditions"
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logging.error(f"Error sending market condition update: {e}")
            
    async def send_joke(self) -> None:
        """Send a random trading joke"""
        try:
            if (self.last_joke_time is None or 
                time.time() - self.last_joke_time > self.joke_interval):
                joke = random.choice(self.trading_jokes)
                message = f"😂 *Trading Joke*\n\n{joke}\n\n#TradingJoke #TradingHumor"
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                self.last_joke_time = time.time()
        except Exception as e:
            logging.error(f"Error sending joke: {e}")
            
    async def send_meme(self) -> None:
        """Send a trading meme"""
        try:
            if (self.last_meme_time is None or 
                time.time() - self.last_meme_time > self.meme_interval):
                meme = random.choice(self.trading_memes)
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=meme,
                    parse_mode='Markdown'
                )
                self.last_meme_time = time.time()
        except Exception as e:
            logging.error(f"Error sending meme: {e}")
            
    async def send_trivia(self) -> None:
        """Send a trading trivia question"""
        try:
            if (self.last_trivia_time is None or 
                time.time() - self.last_trivia_time > self.trivia_interval):
                trivia = random.choice(self.trading_trivia)
                message = (
                    f"❓ *Trading Trivia*\n\n"
                    f"*Question:* {trivia['question']}\n\n"
                    f"*Answer:* {trivia['answer']}\n"
                    f"*Explanation:* {trivia['explanation']}\n\n"
                    f"#TradingTrivia #LearnTrading"
                )
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                self.last_trivia_time = time.time()
        except Exception as e:
            logging.error(f"Error sending trivia: {e}")
            
    async def send_term_of_the_day(self) -> None:
        """Send the trading term of the day"""
        try:
            if (self.last_term_time is None or 
                time.time() - self.last_term_time > self.term_interval):
                term = random.choice(self.trading_terms)
                message = (
                    f"📚 *Trading Term of the Day*\n\n"
                    f"*Term:* {term['term']}\n"
                    f"*Definition:* {term['definition']}\n"
                    f"*Example:* {term['example']}\n\n"
                    f"#TradingTerm #LearnTrading"
                )
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                self.last_term_time = time.time()
        except Exception as e:
            logging.error(f"Error sending term of the day: {e}") 