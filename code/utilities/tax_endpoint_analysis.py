import datetime
import time
from typing import Dict, Any, List, Optional, Union
import ccxt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pydantic import BaseModel


EXCHANGES: Dict[str, Dict[str, Any]] = {
    "bitget": {
        "exchange_object": ccxt.bitget,
        "tax_record_limit": 500,
        "usdt_futures": {
            "product_type": 'USDT-FUTURES',
            "records_column_names": ["id", "symbol", "marginCoin", "futureTaxType", "amount", "fee", "ts"],
            "tax_type": "futureTaxType",
            "trading_types": ["open_long", "close_long", "open_short", "close_short", "contract_margin_settle_fee"],
        },
    },
}

def convert_date_to_timestamp(date_str: str) -> int:
    return int(datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

def convert_timestamp_to_date(timestamp: int) -> str:
    return datetime.datetime.fromtimestamp(int(timestamp) / 1000).strftime("%Y-%m-%d %H:%M:%S")

class AnalysisResult(BaseModel):
    total_trades: int = 0
    capital: float = 0.0
    total_volume: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    win_rate: float = 0.0
    fees: float = 0.0
    pair_pnl: float = 0.0
    total_pair_pnl: float = 0.0
    pair_funding_fees: float = 0.0
    funding_fees: float = 0.0
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    longs_pnl: float = 0.0
    shorts_pnl: float = 0.0
    longs_win_rate: float = 0.0
    shorts_win_rate: float = 0.0
    longs_trades_count: int = 0
    shorts_trades_count: int = 0


class RecordsProcessor:
    def __init__(self, client: ccxt.Exchange, config: Dict[str, Any], portefolio_start_date: str, sleep: bool = True) -> None:
        self.client = client
        self.portefolio_start_date = portefolio_start_date
        self.product_type = config["product_type"]
        self.record_limit = config["record_limit"]
        self.interval_ms = config["interval_ms"]
        self.column_names = config["records_column_names"]
        self.tax_type = config["tax_type"]
        self.trading_types = config["trading_types"]
        self.records_raw: List[Dict[str, Any]] = []
        self.records_raw_df: Optional[pd.DataFrame] = None
        self.records: Optional[pd.DataFrame] = None
        self.trading_records: Optional[pd.DataFrame] = None
        self.extra_records: Optional[pd.DataFrame] = None
        self.pairs: Optional[List[str]] = None
        self.trades: Optional[pd.DataFrame] = None
        self._process_records(portefolio_start_date, sleep)

    def _process_records(self, portefolio_start_date: str, sleep: bool = True) -> None:
        self._fetch_records(portefolio_start_date, sleep)
        self._convert_records()
        self._set_pairs()
        self._complement_records()
        self._create_trades_table()

    def _fetch_records(self, portefolio_start_date: str, sleep: bool = True) -> None:
        start_timestamp = convert_date_to_timestamp(portefolio_start_date)
        current_timestamp = int(time.time() * 1000)
        while start_timestamp < current_timestamp:
            end_timestamp = start_timestamp + self.interval_ms
            self.records_raw.extend(
                self._fetch_records_within_interval(
                    start_timestamp, end_timestamp, current_timestamp, sleep
                )
            )
            start_timestamp = end_timestamp

    def _fetch_records_within_interval(
        self, 
        start_timestamp: int, 
        end_timestamp: int, 
        current_timestamp: int, 
        sleep: bool = True
    ) -> List[Dict[str, Any]]:
        total = []
        while start_timestamp < current_timestamp:
            if sleep:
                time.sleep(1)
            
            response = self.client.privateTaxGetV2TaxFutureRecord({
                "productType": self.product_type,
                "startTime": start_timestamp,
                "endTime": min(end_timestamp, current_timestamp),
            })
            current_records = response["data"]
            
            start_date = convert_timestamp_to_date(start_timestamp)
            end_date = convert_timestamp_to_date(min(end_timestamp, current_timestamp))
            print(f"Fetching data from {start_date} to {end_date}, found {len(current_records)} entries")
            
            total.extend(current_records)
            
            if len(current_records) < self.record_limit:
                break
            else:
                last_id = current_records[-1]["id"]
                total.extend(
                    self._fetch_additional_records(
                        start_timestamp, end_timestamp, current_timestamp, last_id
                    )
                )
        return total

    def _fetch_additional_records(
        self, 
        start_timestamp: int, 
        end_timestamp: int, 
        current_timestamp: int, 
        last_id: str, 
        sleep: bool = False
    ) -> List[Dict[str, Any]]:
        additional_records = []
        while True:
            if sleep:
                time.sleep(1)
                
            response = self.client.privateTaxGetV2TaxFutureRecord({
                "productType": self.product_type,
                "startTime": start_timestamp,
                "endTime": min(end_timestamp, current_timestamp),
                "idLessThan": last_id,
            })
            more_records = response["data"]
            print(f"Fetching additional data from {start_timestamp} to {min(end_timestamp, current_timestamp)}, found {len(more_records)} entries")

            additional_records.extend(more_records)

            if len(more_records) < self.record_limit:
                break
            else:
                last_id = more_records[-1]["id"]
                
        return additional_records

    def _convert_records(self) -> None:
        self.records = pd.DataFrame(self.records_raw, columns=self.column_names)
        self.records_raw_df = self.records.copy()
        self.records["date"] = pd.to_datetime(self.records["ts"].apply(lambda x: convert_timestamp_to_date(x)))
        self.records.set_index("date", inplace=True)

    def _set_pairs(self) -> None:
        if "symbol" in self.records.columns:
            self.pairs = self.records["symbol"].unique().tolist()
        elif "coin" in self.records.columns:
            self.pairs = self.records["coin"].unique().tolist()

    def _complement_records(self) -> None:
        self.records.loc[:, "amount"] = self.records["amount"].astype(float)
        self.records.loc[:, "fee"] = self.records["fee"].astype(float)
        self.records.loc[:, "pnl"] = self.records.apply(
            lambda row: row["fee"] + row["amount"] 
            if row[self.tax_type] in ["open_long", "close_long", "open_short", "close_short"] 
            else 0, 
            axis=1
        )
        self.records.loc[:, "funding_fee"] = self.records.apply(
            lambda row: row["amount"] 
            if row[self.tax_type] == "contract_margin_settle_fee" 
            else 0, 
            axis=1
        )
        self.records.loc[:, "transfer"] = self.records.apply(
            lambda row: row["amount"] 
            if row[self.tax_type] in ["trans_from_exchange", "trans_to_exchange"] 
            else 0, 
            axis=1
        )
        self.records.loc[:, "cumulativePnl"] = (self.records["pnl"] + self.records["funding_fee"]).cumsum()
        self.records.loc[:, "cumulativeCapital"] = (self.records["pnl"] + self.records["funding_fee"] + self.records["transfer"]).cumsum()
        
        self.records = self.records.reindex(columns=[
            self.tax_type, 'symbol', 'amount', 'fee', 'pnl', 'cumulativePnl', 
            'cumulativeCapital', 'transfer', "funding_fee", 'id', 'ts'
        ])
        
        self.trading_records = self.records[self.records[self.tax_type].isin(self.trading_types)]
        self.extra_records = self.records[~self.records[self.tax_type].isin(self.trading_types)]

    def _create_trades_table(self) -> None:
        close_trades = self.trading_records[self.trading_records[self.tax_type].isin(["close_long", "close_short"])]
        close_trades = close_trades.copy()
        close_trades.loc[:, "type"] = close_trades[self.tax_type].apply(
            lambda x: "long" if x == "close_long" else ("short" if x == "close_short" else "other")
        )
        trades = close_trades.reset_index()[["date", "symbol", "type", "pnl"]]
        trades.reset_index(drop=True, inplace=True)
        self.trades = trades


class RecordsAnalyzer:
    def __init__(self, records: pd.DataFrame, tax_type: str) -> None:
        self.records = records
        self.tax_type = tax_type
        self.results: Dict[str, Dict[str, Any]] = {}

    def analyse_global(self) -> None:
        analysis = self._analyse_records(self.records)
        self.results["global"] = analysis.model_dump()

    def analyse_by_pair(self, pairs: List[str]) -> None:
        for pair in pairs:
            pair_records = self.records[self.records["symbol"] == pair]            
            analysis = self._analyse_records(pair_records)
            self.results[pair] = analysis.model_dump()

    def _analyse_records(self, record: pd.DataFrame) -> AnalysisResult:
        if record.empty:
            return AnalysisResult()

        total_trades = record.loc[record[self.tax_type].isin(["close_long", "close_short"])].shape[0]
        
        capital = record["cumulativeCapital"].iloc[-1]
        pnl = record["windowPnl"].iloc[-1]
        pnl_pct = record["windowPnLPct"].iloc[-1]

        pair_pnl = record.loc[record[self.tax_type].isin(["close_long", "close_short"]), "pnl"].sum()
        pair_funding_fees = record["funding_fee"].sum()
        total_pair_pnl = pair_pnl + pair_funding_fees
        
        closing_trades = record[record[self.tax_type].isin(["close_long", "close_short"])]
        total_volume = abs(closing_trades["amount"]).sum()

        win_rate = (record.loc[record[self.tax_type].isin(["close_long", "close_short"]) & (record["pnl"] > 0)].shape[0] / total_trades * 100) if total_trades > 0 else 0
        fees = record["fee"].sum()
        funding_fees = record["funding_fee"].sum()

        long_trades = record.loc[record[self.tax_type] == "close_long"]
        short_trades = record.loc[record[self.tax_type] == "close_short"]
        
        longs_trades_count = long_trades.shape[0]
        shorts_trades_count = short_trades.shape[0]
        
        longs_pnl = long_trades["pnl"].sum()
        shorts_pnl = short_trades["pnl"].sum()
        
        longs_win_rate = (long_trades[long_trades["pnl"] > 0].shape[0] / longs_trades_count * 100) if longs_trades_count > 0 else 0
        shorts_win_rate = (short_trades[short_trades["pnl"] > 0].shape[0] / shorts_trades_count * 100) if shorts_trades_count > 0 else 0

        return AnalysisResult(
            total_trades=total_trades,
            capital=capital,
            total_volume=total_volume,
            pnl=pnl,
            win_rate=win_rate,
            pnl_pct=pnl_pct,
            fees=fees,
            funding_fees=funding_fees,
            pair_pnl=pair_pnl,
            total_pair_pnl=total_pair_pnl,
            pair_funding_fees=pair_funding_fees,
            first_date=self._convert_timestamp_to_date(record["ts"].min()),
            last_date=self._convert_timestamp_to_date(record["ts"].max()),
            longs_pnl=longs_pnl,
            shorts_pnl=shorts_pnl,
            longs_win_rate=longs_win_rate,
            shorts_win_rate=shorts_win_rate,
            longs_trades_count=longs_trades_count,
            shorts_trades_count=shorts_trades_count,
        )

    @staticmethod
    def _convert_timestamp_to_date(timestamp: int) -> str:
        return datetime.datetime.fromtimestamp(int(timestamp) / 1000).strftime("%Y-%m-%d %H:%M:%S")


class RecordsManager:
    def __init__(
        self, 
        api_setup: Dict[str, Any],
        portefolio_start_date: str,
        exchange: str = "bitget",
        market: str = "usdt_futures",
        filename: Optional[str] = None,
        sleep: bool = True
    ) -> None:
        self._client = EXCHANGES[exchange]["exchange_object"](api_setup)
        self._config = {
            "record_limit": EXCHANGES[exchange]["tax_record_limit"],
            "product_type": EXCHANGES[exchange][market]["product_type"],
            "records_column_names": EXCHANGES[exchange][market]["records_column_names"],
            "tax_type": EXCHANGES[exchange][market]["tax_type"],
            "trading_types": EXCHANGES[exchange][market]["trading_types"],
            "interval_ms": 30 * 24 * 60 * 60 * 1000
        }
        self._filename = filename
        self.results: Dict[str, Any] = {}
        self.records_to_analyse: Optional[pd.DataFrame] = None
        
        processor = RecordsProcessor(self._client, self._config, portefolio_start_date, sleep)
        self.records = processor.records
        self.records_raw_df = processor.records_raw_df
        self.trading_records = processor.trading_records
        self.extra_records = processor.extra_records
        self.pairs = processor.pairs
        self.trades = processor.trades
        
        if self._filename:
            self._save_records_to_csv(self.records)

    def analyse(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = self.records.index.min()
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = self.records.index.max()
        
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        self.records_to_analyse = self.records.loc[start_date:end_date].copy()
        
        if start_date > self.records.index.min():
            start_pnl = self.records.loc[:start_date, "cumulativePnl"].iloc[-1]
            start_capital = abs(self.records.loc[:start_date, "cumulativeCapital"].iloc[-1])
        else:
            start_pnl = 0
            start_capital = abs(self.records.loc[start_date:end_date, "cumulativeCapital"].iloc[0]) or 1
        
        self.records_to_analyse.loc[:, "windowPnl"] = self.records_to_analyse["cumulativePnl"] - start_pnl
        self.records_to_analyse.loc[:, "windowPnLPct"] = (self.records_to_analyse["windowPnl"] / start_capital) * 100
        
        analyser = RecordsAnalyzer(self.records_to_analyse, self._config["tax_type"])
        analyser.analyse_global()
        analyser.analyse_by_pair(self.pairs)
        self.results = analyser.results

    def print_global_analysis(self) -> None:
        global_analysis = self.results.get("global", {})
        print("\n ** Global Analysis ** \n")
        print(f"Dates: {global_analysis.get('first_date')} -> {global_analysis.get('last_date')}")
        print(f"Final Capital: {round(global_analysis.get('capital', 0), 2)}")
        print(f"Trades: {global_analysis.get('total_trades')}")
        print(f"Win Rate: {round(global_analysis.get('win_rate', 0), 2)} %")
        print(f"PnL: {round(global_analysis.get('pnl', 0), 2)} $")
        print(f"PnL Pct: {round(global_analysis.get('pnl_pct', 0), 2)} %")
        print(f"Fees: {round(global_analysis.get('fees', 0), 2)} $")
        print(f"Funding Fees: {round(global_analysis.get('funding_fees', 0), 2)} $\n ")

        print("Longs:")
        print(f"  - Trades: {global_analysis.get('longs_trades_count')}")
        print(f"  - PnL: {round(global_analysis.get('longs_pnl', 0), 2)} $")
        print(f"  - Win Rate: {round(global_analysis.get('longs_win_rate', 0), 2)} %")

        print("Shorts:")
        print(f"  - Trades: {global_analysis.get('shorts_trades_count')}")
        print(f"  - PnL: {round(global_analysis.get('shorts_pnl', 0), 2)} $")
        print(f"  - Win Rate: {round(global_analysis.get('shorts_win_rate', 0), 2)} %")

    def _save_records_to_csv(self, records: pd.DataFrame) -> None:
        records.to_csv(self._filename + ".csv", index=True)

    def plot_over_time(self, metric: str, show_transfers: bool = False) -> None:
        plt.figure(figsize=(8, 4))
        if metric == "PnL":
            plt.plot(self.records_to_analyse.index, self.records_to_analyse["windowPnl"], color="blue", label="P&L ($)")
            plt.ylabel("P&L ($)", fontsize=14)
        elif metric == "PnL Pct":
            plt.plot(self.records_to_analyse.index, self.records_to_analyse["windowPnLPct"], color="blue", label="P&L (%)")
            plt.ylabel("Cumulative P&L (%)", fontsize=14)
        elif metric == "Capital":
            plt.plot(self.records_to_analyse.index, self.records_to_analyse["cumulativeCapital"], color="green", label="Capital ($)")
            plt.ylabel("Cumulative Capital ($)", fontsize=14)
        else:
            raise ValueError("Unsupported metric for plot_over_time")

        plt.title(f"{metric} Over Time", fontsize=16)
        plt.grid(True, linestyle=':', color='gray', alpha=0.5)
        plt.xticks(rotation=45, fontsize=10)
        plt.xlabel('')
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        if show_transfers:
            transfer_dates = self.records_to_analyse[self.records_to_analyse["transfer"] != 0].index
            for i, date in enumerate(transfer_dates):
                plt.axvline(x=date, color='purple', linestyle='--', linewidth=0.8, label='Transfers' if i == 0 else "")
        
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_per_pair(self, metric: str, include_funding_fees: bool = True) -> None:
            if metric not in ["PnL", "PnL Pct", "Funding Fees", "Win Rate", "Trades"]:
                raise ValueError("Unsupported metric for plot_per_pair")

            plot_data = []
            for pair, data in self.results.items():
                if pair == "global":
                    continue
                if metric == "PnL":
                    pnl = data["total_pair_pnl"] if include_funding_fees else data["pair_pnl"]
                    plot_data.append((pair, pnl))
                    ylabel = "P&L ($)"
                    title = "P&L per Pair (Including Funding Fees)" if include_funding_fees else "P&L per Pair (Without Funding Fees)"
                elif metric == "Funding Fees":
                    plot_data.append((pair, data["pair_funding_fees"]))
                    ylabel = "Funding Fees ($)"
                    title = "Funding Fees per Pair"
                elif metric == "Win Rate":
                    plot_data.append((pair, data["win_rate"]))
                    ylabel = "Win Rate (%)"
                    title = "Win Rate per Pair"
                elif metric == "Trades":
                    plot_data.append((pair, data["total_trades"]))
                    ylabel = "Number of Trades"
                    title = "Number of Trades per Pair"

            plot_df = pd.DataFrame(plot_data, columns=["symbol", metric])
            plot_df = plot_df.sort_values(by="symbol")

            plt.figure(figsize=(8, 5))
            sns.barplot(x="symbol", y=metric, data=plot_df, palette="RdYlGn", hue=metric, dodge=False, legend=False)
            plt.title(title, fontsize=15)
            plt.ylabel(ylabel, fontsize=13)
            plt.xticks(rotation=45, fontsize=9)
            plt.yticks(fontsize=10)
            plt.xlabel('')
            plt.grid(True, linestyle=':', color='gray', alpha=0.5)
            plt.tight_layout()
            plt.show()

    def plot_per_trade_type(self, metric: str, results: str = "global") -> None:
        data = self.results[results]
        
        if metric == "PnL":
            plot_data = {
                "Trade Type": ["Longs", "Shorts"],
                "Value": [data["longs_pnl"], data["shorts_pnl"]]
            }
            ylabel = "P&L ($)"
        elif metric == "Trades":
            plot_data = {
                "Trade Type": ["Longs", "Shorts"],
                "Value": [data["longs_trades_count"], data["shorts_trades_count"]]
            }
            ylabel = "Number of Trades"
        elif metric == "Win Rate":
            plot_data = {
                "Trade Type": ["Longs", "Shorts"],
                "Value": [data["longs_win_rate"], data["shorts_win_rate"]]
            }
            ylabel = "Win Rate (%)"
        else:
            raise ValueError("Unsupported metric for plot_per_trade_type")

        plt.figure(figsize=(5, 3))
        sns.barplot(x="Trade Type", y="Value", data=pd.DataFrame(plot_data), palette="RdYlGn", hue="Value", dodge=False, legend=False)
        plt.title(f"{metric} per Trade Type ({results.upper()})", fontsize=16)
        plt.ylabel(ylabel, fontsize=13)
        plt.xticks(rotation=0, fontsize=13)
        plt.xlabel('')
        plt.grid(True, linestyle=':', color='gray', alpha=0.3)
        plt.tight_layout()
        plt.show()