import json
import hashlib
import time
import secrets
import requests
import pandas as pd
import numpy as np
import ta

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Union, Dict, Any, TypeVar, Generic

    
class BitunixError(Exception):
    pass

class BitunixNetworkError(BitunixError):
    pass

class BitunixAPIError(BitunixError):
    pass

@dataclass
class APIConfig:
    base_url: str = "https://fapi.bitunix.com"
    timeout: int = 10

T = TypeVar('T')

@dataclass
class BitunixResponse(Generic[T]):
    code: int
    msg: str
    data: T

@dataclass
class Position:
    positionId: str
    symbol: str
    marginCoin: str
    qty: float
    entryValue: float
    side: str  # "LONG" or "SHORT"
    marginMode: str  # "ISOLATION" or "CROSS"
    positionMode: str  # "ONE_WAY" or "HEDGE"
    leverage: int
    fee: float
    funding: float
    realizedPNL: float
    margin: float
    unrealizedPNL: float
    liqPrice: float
    marginRate: float
    avgOpenPrice: float
    ctime: datetime
    mtime: datetime


class BitunixAuth:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
    
    def _generate_signature(self, nonce: str, timestamp: str, query_params: str = "", body: str = "") -> str:
        digest_input = f"{nonce}{timestamp}{self.api_key}{query_params}{body}"
        digest = hashlib.sha256(digest_input.encode()).hexdigest()
        return hashlib.sha256(f"{digest}{self.secret_key}".encode()).hexdigest()
    
    def get_headers(self, query_params: str = "", body: str = "") -> Dict[str, str]:
        nonce = secrets.token_hex(16)
        timestamp = str(int(time.time() * 1000))
        return {
            "api-key": self.api_key,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": self._generate_signature(nonce, timestamp, query_params, body),
            "Content-Type": "application/json"
        }

class BitunixClient:
    def __init__(self, auth: BitunixAuth, config: APIConfig):
        self._auth = auth
        self._config = config

    @staticmethod
    def _handle_response(response: requests.Response) -> Any:
        if response.status_code != 200:
            try:
                error_detail = response.json()
            except (json.JSONDecodeError, AttributeError):
                error_detail = {"status": response.status_code}
            raise BitunixNetworkError(f"HTTP {response.status_code} error: {error_detail}")
        
        typed_response = BitunixResponse(**response.json())
        if typed_response.code != 0:
            raise BitunixAPIError(f"Bitunix API error code {typed_response.code}: {typed_response.msg}")
        return typed_response.data

    def get(self, endpoint: str, query_params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self._config.base_url}{endpoint}"
        
        sorted_params = ""
        if query_params:
            sorted_items = sorted(query_params.items(), key=lambda x: x[0])
            sorted_params = "".join(f"{key}{value}" for key, value in sorted_items)
        
        headers = self._auth.get_headers(query_params=sorted_params)
        
        try:
            response = requests.get(
                url=url,
                headers=headers,
                params=query_params,
                timeout=self._config.timeout
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise BitunixNetworkError(f"Request failed: {e}")

    def post(self, endpoint: str, data: Dict[str, Any]) -> Any:
        url = f"{self._config.base_url}{endpoint}"
        data_str = json.dumps(data, separators=(',', ':'))
        headers = self._auth.get_headers(body=data_str)
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=data_str,
                timeout=self._config.timeout
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise BitunixNetworkError(f"Request failed: {e}")

class BitunixFutures:
    API_PATH = "/api/v1/futures"

    def __init__(self, api_key: str, secret_key: str, config: Optional[APIConfig] = None):
        self._config = config or APIConfig()
        self._auth = BitunixAuth(api_key, secret_key)
        self._client = BitunixClient(self._auth, self._config)
        self._trading_pairs_info: Optional[pd.DataFrame] = None
        self._current_symbol_info: Optional[Dict[str, Any]] = None

    # ==================
    # Helper Methods
    # ==================

    def _ensure_trading_pairs_info(self, symbol: str) -> None:
        if self._trading_pairs_info is None:
            self._trading_pairs_info = self.get_trading_pairs()

        if symbol not in self._trading_pairs_info.index:
            raise ValueError(f"Symbol {symbol} not found in trading pairs")

        self._current_symbol_info = self._trading_pairs_info.loc[symbol].to_dict()

    def _qty_to_precision(self, symbol: str, amount: float, rounding_mode: str = "TRUNCATE") -> str:
        try:
            self._ensure_trading_pairs_info(symbol)
            min_amount = float(self._current_symbol_info['minTradeVolume'])
            
            if amount < min_amount:
                raise ValueError(f"Amount {amount} is less than minimum {min_amount}")
            
            return self._apply_precision(amount, self._current_symbol_info['basePrecision'], rounding_mode)
            
        except Exception as e:
            raise ValueError(f"Failed to calculate amount precision: {str(e)}")

    def _price_to_precision(self, symbol: str, price: float, rounding_mode: str = "ROUND") -> str:
        try:
            self._ensure_trading_pairs_info(symbol)
            return self._apply_precision(price, self._current_symbol_info['quotePrecision'], rounding_mode)
            
        except Exception as e:
            raise ValueError(f"Failed to calculate price precision: {str(e)}")

    @staticmethod
    def _apply_precision(value: float, precision: int, rounding_mode: str) -> str:
        multiplier = 10 ** precision
        scaled = value * multiplier
        scaled = int(scaled) if rounding_mode == "TRUNCATE" else round(scaled)
        return f"{scaled / multiplier:g}"

    # ==================
    # Account Methods
    # ==================

    def get_account_balance(self, margin_coin: str) -> str:
        endpoint = self.API_PATH + "/account"
        query_params = {"marginCoin": margin_coin}
        response_data = self._client.get(endpoint, query_params)
        return response_data["available"]

    def set_position_mode(self, hedge_mode: bool) -> Dict[str, Any]:
        endpoint = self.API_PATH + "/account/change_position_mode"
        data = {
            "positionMode": "HEDGE" if hedge_mode else "ONE_WAY"
        }
        return self._client.post(endpoint, data)

    def set_margin_mode(self, symbol: str, margin_mode: str = "ISOLATION", margin_coin: str = "USDT") -> Dict[str, Any]:
        margin_mode = margin_mode.upper()
        if margin_mode not in ['CROSS', 'ISOLATION']:
            raise ValueError("margin_mode must be either 'CROSSED' or 'ISOLATED'")

        endpoint = self.API_PATH + "/account/change_margin_mode"
        data = {
            "symbol": symbol,
            "marginMode": margin_mode,
            "marginCoin": margin_coin
        }
        return self._client.post(endpoint, data)

    def set_leverage(self, symbol: str, leverage: int, margin_coin: str = "USDT") -> Dict[str, Any]:
        endpoint = self.API_PATH + "/account/change_leverage"
        data = {
            "symbol": symbol,
            "leverage": leverage,
            "marginCoin": margin_coin
        }
        return self._client.post(endpoint, data)

    # ==================
    # Market Methods
    # ==================

    def get_kline(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = 100,
        kline_type: str = "LAST_PRICE"
    ) -> pd.DataFrame:
        endpoint = self.API_PATH + "/market/kline"
        query_params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "type": kline_type
        }
        if start_time is not None:
            query_params["startTime"] = start_time
        if end_time is not None:
            query_params["endTime"] = end_time

        raw_data = self._client.get(endpoint, query_params)
        return self._convert_raw_klines_to_dataframe(raw_data)

    @staticmethod
    def _convert_raw_klines_to_dataframe(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(raw_data)
        df['datetime'] = pd.to_datetime(df['time'].astype(np.float64), unit='ms')
        df = df.set_index('datetime').sort_index(ascending=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'quoteVol', 'baseVol']
        df[numeric_cols] = df[numeric_cols].astype(float)

        return df.drop('time', axis=1)

    def get_trading_pairs(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        endpoint = self.API_PATH + "/market/trading_pairs"
        
        query_params = {}
        if symbols:
            query_params["symbols"] = ",".join(symbols)
            
        raw_data = self._client.get(endpoint, query_params)
        return self._convert_trading_pairs_to_dataframe(raw_data)

    @staticmethod
    def _convert_trading_pairs_to_dataframe(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        data = np.array([tuple(d.values()) for d in raw_data],
                       dtype=[(k, object) for k in raw_data[0].keys()])
        
        df = pd.DataFrame(data)
        
        return df.set_index('symbol')

    # ==================
    # Trade Methods
    # ==================

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "BUY" or "SELL"
        trade_side: str,  # "OPEN" or "CLOSE"
        order_type: str,  # "LIMIT" or "MARKET"
        price: Optional[float] = None,
        position_id: Optional[str] = None,
        effect: str = "GTC",  # "IOC", "FOK", "GTC", "POST_ONLY"
        client_id: Optional[str] = None,
        reduce_only: bool = False,
        tp_price: Optional[float] = None,
        tp_stop_type: str = "LAST_PRICE",  # "MARK_PRICE" or "LAST_PRICE"
        tp_order_type: str = "MARKET",  # "LIMIT" or "MARKET"
        tp_order_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: str = "LAST_PRICE",  # "MARK_PRICE" or "LAST_PRICE"
        sl_order_type: str = "MARKET",  # "LIMIT" or "MARKET"
        sl_order_price: Optional[float] = None,
    ) -> Dict[str, str]:
        endpoint = self.API_PATH + "/trade/place_order"
        
        if order_type == "LIMIT" and price is None:
            raise ValueError("Price is required for LIMIT orders")
            
        if trade_side == "CLOSE" and position_id is None:
            raise ValueError("Position ID is required when trade_side is CLOSE")

        order_data = {
            "symbol": symbol,
            "qty": self._qty_to_precision(symbol, qty),
            "side": side,
            "tradeSide": trade_side,
            "orderType": order_type,
            "effect": effect,
            "price": self._price_to_precision(symbol, price) if price is not None else None,
            "positionId": position_id,
            "clientId": client_id,
            "reduceOnly": reduce_only,
            "tpPrice": self._price_to_precision(symbol, tp_price) if tp_price is not None else None,
            "tpStopType": tp_stop_type,
            "tpOrderType": tp_order_type,
            "tpOrderPrice": self._price_to_precision(symbol, tp_order_price) if tp_order_price is not None else None,
            "slPrice": self._price_to_precision(symbol, sl_price) if sl_price is not None else None,
            "slStopType": sl_stop_type,
            "slOrderType": sl_order_type,
            "slOrderPrice": self._price_to_precision(symbol, sl_order_price) if sl_order_price is not None else None,
        }

        return self._client.post(endpoint, {k: v for k, v in order_data.items() if v is not None})

    # ==================
    # Position Methods
    # ==================

    def get_pending_positions(
        self, 
        symbol: Optional[str] = None, 
        position_id: Optional[str] = None
    ) -> Optional[Position]:
        if not symbol:
            raise ValueError("Symbol is required")

        endpoint = self.API_PATH + "/position/get_pending_positions"
        
        query_params = {"symbol": symbol}
        if position_id:
            query_params["positionId"] = position_id
            
        raw_data = self._client.get(endpoint, query_params)

        if len(raw_data) > 1:
            raise ValueError("Multiple positions found. Currently only one-way mode is supported")

        return Position(**raw_data[0]) if raw_data else None

    def flash_close_position(self, position_id: str) -> Dict[str, str]:
        if not position_id:
            raise ValueError("Position ID is required")
            
        endpoint = self.API_PATH + "/trade/flash_close_position"
        data = {"positionId": position_id}
        
        return self._client.post(endpoint, data)


if __name__ == "__main__":

    # ==================
    # Bot Parameters
    # ==================
    SYMBOL = "ETHUSDT"

    # Account settings
    LEVERAGE = 1
    MARGIN_MODE = "ISOLATION"  # or "CROSS"

    # Trading parameters
    POSITION_SIZE_PCT = 50.0  # Percentage of account balance to use per trade
    TP_PCT = 5.0              # Take profit percentage
    SL_PCT = 5.0              # Stop loss percentage

    # RSI parameters
    TIMEFRAME = "4h"
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70

    VERBOSE = True           # Control output messages

    # ==================
    # Initialize Client
    # ==================
    with open("LiveTradingBots/code/strategies/bitunix_bot_template/credentials.json", "r") as f:
        key = json.load(f)

    client = BitunixFutures(
        api_key=key.get("api_key"),
        secret_key=key.get("secret_key")
    )

    # ==================
    # Account Setup
    # ==================
    try:
        MARGIN_COIN = "USDT"
        HEDGE_MODE = False

        position = client.get_pending_positions(SYMBOL)
        if not position:
            client.set_position_mode(hedge_mode=HEDGE_MODE)
            client.set_leverage(SYMBOL, LEVERAGE, MARGIN_COIN)
            client.set_margin_mode(SYMBOL, MARGIN_MODE, MARGIN_COIN)

        balance = float(client.get_account_balance(MARGIN_COIN))
        position_size_usd = balance * (POSITION_SIZE_PCT / 100)

        if VERBOSE:
            print(f"\nRunning bot for {SYMBOL} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
            print(f"  > Position Mode: {'Hedge' if HEDGE_MODE else 'One-Way'} | Margin: {MARGIN_MODE} | Leverage: {LEVERAGE}x")
            print(f"  > Account balance: {balance} {MARGIN_COIN}")

    except (BitunixAPIError, BitunixNetworkError) as e:
        print(f"Failed to setup account: {e}")
        exit(1)

    # ==================
    # Market Data & RSI
    # ==================
    try:
        if VERBOSE:
            print("\nFetching market data...")

        klines = client.get_kline(
            symbol=SYMBOL,
            interval=TIMEFRAME,
            limit=100
        )
        close_price = klines['close'].iloc[-2]

        rsi = ta.momentum.rsi(
            close=klines['close'],
            window=RSI_PERIOD,
        ).dropna()

        current_rsi = rsi.iloc[-2]

        if VERBOSE:
            print(f"  > Last close price: {close_price:.2f} {MARGIN_COIN}")
            print(f"  > RSI: {current_rsi:.2f} | Overbought threshold: {RSI_OVERBOUGHT}")

    except Exception as e:
        print(f"Failed to fetch market data or compute RSI: {e}")
        exit(1)

    # ==================
    # Trading Signals
    # ==================
    try:
        previous_rsi = rsi.iloc[-3]
        entry_condition = previous_rsi <= RSI_OVERBOUGHT < current_rsi
        exit_condition = previous_rsi >= RSI_OVERBOUGHT > current_rsi

        if VERBOSE:
            print(f"  > Entry Signal: {'OK' if entry_condition else 'NOK'}")
            print(f"  > Exit Signal: {'OK' if exit_condition else 'NOK'}")

    except Exception as e:
        print(f"Error in signal analysis: {e}")
        exit(1)

    # =====================
    # Entry Order Placement
    # =====================
    try:
        if entry_condition:
            if VERBOSE:
                print("\nPlacing entry order...")

            tp_price = close_price * (1 + TP_PCT/100)
            sl_price = close_price * (1 - SL_PCT/100)
            qty = position_size_usd / close_price

            if VERBOSE:
                print(f"  > Position size: {position_size_usd:.2f} {MARGIN_COIN} ({POSITION_SIZE_PCT}% of balance)")
                print(f"  > Take Profit price (+{TP_PCT}%): {tp_price:.2f} {MARGIN_COIN}")
                print(f"  > Stop Loss price (-{SL_PCT}%): {sl_price:.2f} {MARGIN_COIN}")

            order = client.place_order(
                symbol=SYMBOL,
                qty=qty,
                side="BUY",
                trade_side="OPEN",
                order_type="MARKET",
                tp_price=tp_price,
                sl_price=sl_price,
            )

            if VERBOSE:
                print(f"  > Status: Position opened successfully (Id: {order.get('orderId', 'N/A')})")

    except Exception as e:
        print(f"Error in order placement: {e}")
        exit(1)

    # =====================
    # Exit Order Placement
    # =====================
    try:
        if exit_condition:
            if VERBOSE:
                print("\nChecking for positions to exit...")

            if position is None:
                if VERBOSE:
                    print("  > No open position found")
            else:
                if VERBOSE:
                    print(f"  > Found position: {position.positionId}")
                    print(f"  > Current PnL: {float(position.unrealizedPNL):.2f} {MARGIN_COIN}")

                result = client.flash_close_position(position.positionId)

                if VERBOSE:
                    print(f"  > Status: Position closed successfully")

    except Exception as e:
        print(f"Error in exit order placement: {e}")
        exit(1)

    if VERBOSE:
        print(f"\n >>> Trading session completed. See you next time ;)")
