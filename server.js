const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const path = require('path');
const zlib = require('zlib');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Enable CORS
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Store connected clients
const clients = new Set();

// Premium Bot Configuration
const botConfig = {
    maxTrades: 5,
    riskLevel: 3,
    stopLoss: 30,
    takeProfit: 50,
    maxDailyLoss: 5,
    tradingPairs: ['EUR/USD', 'GBP/USD', 'USD/JPY'],
    tradingHours: {
        start: '08:00',
        end: '20:00'
    },
    strategies: {
        trendFollowing: true,
        breakout: true,
        scalping: false
    }
};

// Simulated data with premium features
let currentPrice = 1.0987;
let trades = [];
let account = {
    balance: 10245.67,
    equity: 10532.12,
    margin: 1245.00,
    marginPercent: 12,
    freeMargin: 9287.12,
    openTrades: 0,
    dailyPL: 286.45,
    totalPL: 1245.67,
    winRate: 68.5,
    averageWin: 45.23,
    averageLoss: -32.15
};

// Compression function
function compressData(data) {
    return zlib.deflateSync(JSON.stringify(data));
}

// Technical Analysis Functions
function calculateRSI(prices, period = 14) {
    let gains = 0;
    let losses = 0;
    
    for (let i = 1; i < period; i++) {
        const difference = prices[i] - prices[i - 1];
        if (difference >= 0) {
            gains += difference;
        } else {
            losses -= difference;
        }
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}

function calculateMACD(prices) {
    const ema12 = prices.slice(-12).reduce((a, b) => a + b) / 12;
    const ema26 = prices.slice(-26).reduce((a, b) => a + b) / 26;
    return ema12 - ema26;
}

// Trading Strategy Functions
function analyzeMarket(prices) {
    const rsi = calculateRSI(prices);
    const macd = calculateMACD(prices);
    
    return {
        rsi,
        macd,
        signal: rsi > 70 ? 'SELL' : rsi < 30 ? 'BUY' : 'NEUTRAL',
        strength: Math.abs(macd)
    };
}

// WebSocket connection handler
wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('Client connected');

    // Send initial data with compression
    const initialData = {
        type: 'init',
        data: {
            price: currentPrice,
            account: account,
            trades: trades,
            config: botConfig
        }
    };
    ws.send(compressData(initialData));

    ws.on('close', () => {
        clients.delete(ws);
        console.log('Client disconnected');
    });
});

// Simulate price updates with market analysis
let priceHistory = Array(50).fill(currentPrice);
setInterval(() => {
    const change = (Math.random() - 0.5) * 0.0005;
    currentPrice += change;
    priceHistory.push(currentPrice);
    priceHistory.shift();
    
    const analysis = analyzeMarket(priceHistory);
    
    // Broadcast price update with compression
    const updateData = {
        type: 'price_update',
        price: currentPrice,
        analysis: analysis
    };
    
    const compressedData = compressData(updateData);
    
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(compressedData);
        }
    });
}, 1000);

// API Routes
app.post('/api/trade', (req, res) => {
    const { type, pair, lots } = req.body;
    
    // Validate trade against bot configuration
    if (!botConfig.tradingPairs.includes(pair)) {
        return res.status(400).json({ error: 'Invalid trading pair' });
    }
    
    if (trades.length >= botConfig.maxTrades) {
        return res.status(400).json({ error: 'Maximum number of trades reached' });
    }
    
    // Create new trade with premium features
    const trade = {
        id: Date.now(),
        type,
        pair,
        lots,
        openPrice: currentPrice,
        currentPrice: currentPrice,
        profit: 0,
        timeAgo: 'Just now',
        stopLoss: currentPrice - (type === 'BUY' ? botConfig.stopLoss : -botConfig.stopLoss) * 0.0001,
        takeProfit: currentPrice + (type === 'BUY' ? botConfig.takeProfit : -botConfig.takeProfit) * 0.0001,
        strategy: 'Premium Bot'
    };
    
    trades.unshift(trade);
    account.openTrades = trades.length;
    
    // Broadcast trade update
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({
                type: 'trade_update',
                trade: trade
            }));
        }
    });
    
    res.json({ success: true, trade });
});

app.post('/api/trade/:id/close', (req, res) => {
    const tradeId = parseInt(req.params.id);
    const tradeIndex = trades.findIndex(t => t.id === tradeId);
    
    if (tradeIndex === -1) {
        return res.status(404).json({ error: 'Trade not found' });
    }
    
    const trade = trades[tradeIndex];
    const profit = (currentPrice - trade.openPrice) * trade.lots * 100000;
    
    // Update account with premium statistics
    account.balance += profit;
    account.equity = account.balance;
    account.dailyPL += profit;
    account.totalPL += profit;
    
    // Update win rate
    const isWin = profit > 0;
    account.winRate = ((account.winRate * (account.openTrades - 1)) + (isWin ? 100 : 0)) / account.openTrades;
    
    // Update average win/loss
    if (isWin) {
        account.averageWin = ((account.averageWin * (account.openTrades - 1)) + profit) / account.openTrades;
    } else {
        account.averageLoss = ((account.averageLoss * (account.openTrades - 1)) + profit) / account.openTrades;
    }
    
    // Remove trade
    trades.splice(tradeIndex, 1);
    account.openTrades = trades.length;
    
    // Broadcast updates
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({
                type: 'account_update',
                account: account
            }));
        }
    });
    
    res.json({ success: true, profit, trade });
});

// Bot Configuration API
app.get('/api/bot/config', (req, res) => {
    res.json(botConfig);
});

app.post('/api/bot/config', (req, res) => {
    const newConfig = req.body;
    Object.assign(botConfig, newConfig);
    res.json({ success: true, config: botConfig });
});

// Backtesting API
app.post('/api/backtest', (req, res) => {
    const { strategy, pair, timeframe, startDate, endDate } = req.body;
    
    // Simulate backtesting with realistic data
    const backtestResults = {
        totalTrades: Math.floor(Math.random() * 100) + 50,
        winRate: (Math.random() * 30 + 50).toFixed(1),
        profitFactor: (Math.random() * 1.5 + 0.5).toFixed(1),
        drawdown: (Math.random() * 15 + 5).toFixed(1),
        netProfit: (Math.random() * 5000 + 1000).toFixed(2),
        averageWin: (Math.random() * 50 + 20).toFixed(2),
        averageLoss: (Math.random() * 30 + 10).toFixed(2),
        maxConsecutiveWins: Math.floor(Math.random() * 10) + 5,
        maxConsecutiveLosses: Math.floor(Math.random() * 5) + 2,
        sharpeRatio: (Math.random() * 2 + 0.5).toFixed(2),
        monthlyReturns: Array.from({length: 12}, () => (Math.random() * 10 - 2).toFixed(1)),
        equityCurve: Array.from({length: 100}, (_, i) => ({
            date: new Date(Date.now() - (100 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            equity: (10000 + i * 100 + Math.random() * 50).toFixed(2)
        }))
    };
    
    res.json(backtestResults);
});

// Start server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Premium Bot Server running on port ${PORT}`);
    console.log(`Web interface available at http://localhost:${PORT}`);
}).on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`Port ${PORT} is already in use. Please try a different port.`);
        console.error('You can set a different port by setting the PORT environment variable.');
    } else {
        console.error('Server error:', err);
    }
    process.exit(1);
}); 