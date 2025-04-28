import requests
from typing import Optional
from config import Config

class Notifier:
    def __init__(self, config: Config):
        self.config = config
        self.telegram_token = config.telegram_token
        self.telegram_chat_id = config.telegram_chat_id
        
    def send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False
            
    def notify_trade_opened(self, symbol: str, direction: str, price: float) -> None:
        """Send trade opened notification"""
        message = (
            f"🚀 <b>Trade Opened</b>\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Price: {price}"
        )
        self.send_telegram_message(message)
        
    def notify_trade_closed(self, symbol: str, direction: str, price: float, profit: float) -> None:
        """Send trade closed notification"""
        message = (
            f"🏁 <b>Trade Closed</b>\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Price: {price}\n"
            f"Profit: {profit}"
        )
        self.send_telegram_message(message)
        
    def notify_error(self, error: str) -> None:
        """Send error notification"""
        message = f"❌ <b>Error</b>\n{error}"
        self.send_telegram_message(message)
        
    def notify_warning(self, warning: str) -> None:
        """Send warning notification"""
        message = f"⚠️ <b>Warning</b>\n{warning}"
        self.send_telegram_message(message) 