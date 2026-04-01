import json
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import crypto
import argparse
import asyncio



if __name__ == "__main__":
    symbols = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","LTCUSDT"]
    parser = argparse.ArgumentParser(description="Crypto prediction tool")
    parser.add_argument("--data_fetch", action="store_true", help="fetching data")
    parser.add_argument("--train", action="store_true", help="Training mode")
    parser.add_argument("--predict", action="store_true", help="Prediction mode")
    parser.add_argument("--window_days", type=int, default=10, help="Number of past days in each training window")
    parser.add_argument("--horizon", type=int, default=8, help="Number of resampled steps ahead to predict")
    parser.add_argument("--resample_hours", type=int, default=3, help="Resampling interval in hours")
    parser.add_argument("--months", type=int, default=18, help="Number of past months to fetch (-1 for full history)")
    parser.add_argument("--step", type=int, default=1, help="Stride between sliding windows")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--get_best_config",action="store_true",help="find best eval config based on multiple periods")
    parser.add_argument("--backtest",action="store_true",help="backtesting")
    parser.add_argument("--cutoff",type=int,default=0,help="how many days ago  should there be data fetch cutoff")
    parser.add_argument("--symbol",type=str,default="BTCUSDT",help="pair to train on")
    parser.add_argument("--paper_trade",action="store_true",help="Prediction mode")
    parser.add_argument("--trend_follow_backtest", action="store_true", help="Run trend following paper trade backtest")

    args = parser.parse_args()
    crypto.load_config()
    ##.connect()
    # crypto.api_up()

    if args.predict:
        crypto.eval_live(months=args.months,window_days=args.window_days,resample_hours=args.resample_hours,horizon=args.horizon)
    if args.data_fetch:
        if args.symbol!="all":
            crypto.make_dataset(
                symbol=args.symbol, # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
                months=args.months,
                window_days=args.window_days,
                resample_hours=args.resample_hours,
                horizon=args.horizon,
                step=args.step,
                cutoff=args.cutoff
            )
        else:
            for symbol in symbols:
                crypto.make_dataset(
                    symbol=symbol,
                    # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
                    months=args.months,
                    window_days=args.window_days,
                    resample_hours=args.resample_hours,
                    horizon=args.horizon,
                    step=args.step,
                    cutoff=args.cutoff
                )
                time.sleep(1)



    if args.train:
        if args.symbol!="all":
        # Prefer new dataset training if available
            _,_,_,_ = crypto.Train_val(dataset_dir=f"{args.symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",EPOCHS=args.epochs)
        # Fallback or alternative: crypto.TrainAll(hours_collect=int(args.hours_collect))
        else:

            for symbol in symbols:
                _,_,_,_ = crypto.Train_val(dataset_dir=f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",
                                EPOCHS=args.epochs)



    if args.backtest:
        i = 180
        for symbol in symbols:
            path = f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}/eval_results.json"
            if os.path.exists(path):
                os.remove(path)

        while i < 445:

            for symbol in symbols:
                crypto.make_dataset(
                    symbol=symbol,
                    # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
                    months=args.months,
                    window_days=args.window_days,
                    resample_hours=args.resample_hours,
                    horizon=args.horizon,
                    step=args.step,
                    cutoff=i
                )
                time.sleep(1)

            for symbol in symbols:
                _, _, _, _ = crypto.Train_val(
                    dataset_dir=f"{symbol}_{args.window_days}_{args.resample_hours}_{args.horizon}",
                    EPOCHS=args.epochs)

            i += 40

    if args.paper_trade:
        i = 0
        drawdowns = []
        rois = []
        while i < 180:
            drawdown, roi = crypto.paper_trade_historical(months=args.months, window_days=args.window_days,
                                                          resample_hours=args.resample_hours,
                                                          horizon=args.horizon, cutoff=i)
            drawdowns.append(drawdown)
            rois.append(roi)
            i += 30
        print(f"mean roi {np.mean(rois)}")
        print(f"mean drawdown {np.mean(drawdowns)}")
        print(f"median roi {np.median(rois)}")
        print(f"median drawdown {np.median(drawdowns)}")