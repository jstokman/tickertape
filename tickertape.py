'''
Evidence Based Technical Analysis Toolkit for personal use only
This is an abridged verion of something I put together over the summer of 2015 and when I had the time between projects
Inspired by: https://www.amazon.com/Evidence-Based-Technical-Analysis-Scientific-Statistical/dp/0470008741
I try to use vectorized operations as much as possible, but have to occasionally loop when neccessary
See links in comments above methods to view the original formulas
Once imported into Jupyter notebook API looks like this for generating feature rich dataframes for EDA and modelling
df = Tape(get_frame(indices=['SPY'])).moving_average(periods=[25,50,200]).volatility(sample_size=500).returns(periods=[30]).concat()
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import json
import requests

'''
I have a end of day data in BigQuery and have separate methods for loading intraday data
I have ETF symbol sets as pickled series on my hard drive
'''
def get_frame(symbols=[], indices=[]):
    for i in indices:
        symbols = symbols + list(pd.read_pickle('./pickles/indices/' + i + '.pickle').symbol)
    def quote(x): return "'" + x + "'"
    symbols_query = "(" + ",".join(map(quote, symbols)) + ")"
    frame = pd.io.gbq.read_gbq("SELECT symbol, date, adjusted_open as o, adjusted_high as h, adjusted_low as l, adjusted_close as c, adjusted_volume as v " +
                             "FROM us_stock.eod3 WHERE symbol in " + symbols_query + " AND year(date) >= 2000 ORDER BY symbol ASC, date ASC;",
                            project_id="antikythera-1203", index_col="date")
    frame.index = pd.to_datetime(frame.index)
    return frame

class Tape:
    def __init__(self, frame):
        g = frame.groupby('symbol')
        self.group = g
        self.series_set = {}
        print('init')
        for k in g.groups.keys():
            self.series_set[k] = frame[frame.symbol == k].copy()

    def concat(self):
        all = []
        for k, v in self.series_set.items():
            all.append(v)
        return pd.concat(all)


    def returns(self, periods=[3]):
        for period in periods:
            label1 = 'r' + str(period)
            label2 = 'ro' + str(period)
            label3 = 'roo' + str(period)
            for symbol, frame in self.series_set.items():
                frame[label1] = (frame.shift(period * -1).c / frame.c - 1)
                frame[label2] = (frame.shift(period * -1).c / frame.o - 1)
                frame[label3] = (frame.shift(period * -1).o / frame.o - 1)
        return self

    def volatility(self, periods=[1], sample_size=252):
        for period in periods:
            label1 = 'v' + str(period)
            label2 = 'ac' + str(period)
            label3 = 'c' + str(period)
            for symbol, frame in self.series_set.items():
                frame[label3] = (frame.c / frame.shift(period).c - 1)
                frame[label1] = frame[label3].rolling(window=sample_size,center=False).std()
                frame[label2] = frame[label3].rolling(window=sample_size,center=False).mean()
        return self

    def rate_of_change(self, periods=[4], sample_size=252):
        for period in periods:
            #Rate of Change
            label1 = 'roc' + str(period)
            label2 = 'rocz' + str(period)
            for symbol, frame in self.series_set.items():
                #Calculation
                frame[label1] = ( frame.c / frame.shift(period).c - 1)
                #Zscores
                roc = frame[label1]
                mean = roc.rolling(window=sample_size,center=False).mean()
                std = roc.rolling(window=sample_size,center=False).std()
                frame[label2] = (roc - mean) / std
        return self

    def overnight(self, periods=[1], sample_size=500):
        for period in periods:
            #Rate of Change
            label1 = 'ovn' + str(period)
            label2 = 'ovnz' + str(period)
            for symbol, frame in self.series_set.items():
                #Calculation
                frame[label1] = ( frame.o / frame.shift(period).c - 1)
                #Zscores
                roc = frame[label1]
                mean = roc.rolling(window=sample_size,center=False).mean()
                std = roc.rolling(window=sample_size,center=False).std()
                frame[label2] = (roc - mean) / std
        return self


    def moving_average(self, periods=[200]):
        for period in periods:
            label1 = 'ma' + str(period)
            label2 = 'maz' + str(period)
            for symbol, frame in self.series_set.items():
                frame[label1] = frame.c.rolling(window=period,center=False).mean()
                mean = frame[label1]
                std = frame.c.rolling(window=period,center=False).std()
                frame[label2] = (frame.c - mean) / std
        return self

    # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon
    def aroon(self, periods=[25]):
        for period in periods:
            label1 = 'aroon' + str(period) + 'up'
            label2 = 'aroon' + str(period) + 'down'
            label3 = 'aroon' + str(period) + 'combo'
            label4 = 'aroon' + str(period) + 'cross'
            def cross(x):
                if x[0] == 0:
                    if x[1] == 1:
                        return 1
                    else:
                        return 0
                if x[0] == 1:
                    if x[1] == 0:
                        return -1
                    else:
                        return 0
            for symbol, frame in self.series_set.items():
                frame[label1] = frame.h.rolling(window=period,center=False).apply(lambda x: ((period - (period - (np.argmax(x) + 1))) / period) * 100)
                frame[label2] = frame.l.rolling(window=period,center=False).apply(lambda x: ((period - (period - (np.argmin(x) + 1))) / period) * 100)
                frame[label3] = frame[label1] + (frame[label2] * -1)
                g = frame[label1] > frame[label2]
                frame[label4] = g.rolling(window=2,center=False).apply(cross)
        return self

    # https://www.marketvolume.com/technicalanalysis/efficiencyratio.asp
    def efficiency_ratio(self, periods=[30], sample_size=500):
        for period in periods:
            label = 'er' + str(period)
            for symbol, frame in self.series_set.items():
                abs_daily_change = abs(frame.c - frame.shift(1).c)
                abs_period_change = abs(frame.c - frame.shift(period).c)
                sum_change = pd.rolling_sum(abs_daily_change, period)
                frame[label] = abs_period_change / sum_change
                frame['m' + label] = pd.rolling_mean(frame[label], sample_size)
                std = pd.rolling_std(frame[label], sample_size)
                frame['erz' + str(period)] = (frame[label] - frame['m' + label]) / std
        return self

    # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
    def stochastic(self, periods=[14]):
        for period in periods:
            label = 'sk'
            for symbol, frame in self.series_set.items():
                hh = pd.rolling_max(frame.h, period)
                ll = pd.rolling_min(frame.l, period)
                frame[label] = (frame.c - ll) / (hh - ll) * 100
        return self

    def correlation(self, compare, periods=1):
        compare = (compare.c / compare.shift(periods).c - 1)
        for symbol, frame in self.series_set.items():
            change = (frame.c / frame.shift(periods).c - 1)
            frame['corr'] = change.rolling(window=250).corr(compare) #pd.rolling_corr(change, compare, 250)
            frame['diff'] = change - compare
        return self

    # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi
    def rsi(self, periods=[14], sample_size=500):
        for symbol, frame in self.series_set.items():
            frame['change'] = frame.c - frame.shift(1).c
            frame['gain'] = frame[frame.change > 0].change
            frame['loss'] = frame[frame.change < 0].change * -1
            frame['gain'] = frame.gain.fillna(0)
            frame['loss'] = frame.loss.fillna(0)
            for period in periods:
                label = 'rsi' + str(period)
                frame['avg_gain' + str(period)] = pd.rolling_mean(frame.gain, period)
                frame['avg_loss' + str(period)] = pd.rolling_mean(frame.loss, period)
                frame['rs' + str(period)] = frame['avg_gain' + str(period)] / frame['avg_loss' + str(period)]
                frame['rsi' + str(period)] = 100 - (100 / (1 + frame['rs' + str(period)]))
                #Zscores
                rs = frame['rs' + str(period)]
                mean = pd.rolling_mean(rs, sample_size)
                std = pd.rolling_std(rs, sample_size)
                frame['rsz' + str(period)] = (rs - mean) / std
        return self

    # https://www.forextraders.com/forex-education/forex-indicators/relative-vigor-index-indicator-explained/
    def rvi(self, periods=[14], sample_size=252):
        for symbol, frame in self.series_set.items():
            frame['change'] = frame.c - frame.shift(1).c
            frame['gain'] = frame[frame.change > 0].change
            frame['loss'] = frame[frame.change < 0].change * -1
            frame['gain'] = frame.gain.fillna(0)
            frame['loss'] = frame.loss.fillna(0)
            for period in periods:
                label = 'rsi' + str(period)
                frame['gain_std' + str(period)] = frame.gain.rolling(window=period,center=False).std()
                frame['loss_std' + str(period)] = frame.loss.rolling(window=period,center=False).std()
                frame['rv' + str(period)] = frame['gain_std' + str(period)] / frame['loss_std' + str(period)]
                frame['rvi' + str(period)] = 100 - (100 / (1 + frame['rv' + str(period)]))
        return self

    # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    def chaikin_money_flow(self, periods=20):
        label = 'cmf' + str(periods)
        for symbol, frame in self.series_set.items():
            money_flow_multiplier = ((frame['c'] - frame['l']) - (frame['h'] - frame['c'])) / (frame['h'] - frame['l'])
            money_flow_volume = money_flow_multiplier * frame['v']
            frame[label] = money_flow_volume.rolling(window=periods,center=False).sum() / frame['v'].rolling(window=periods,center=False).sum()
        return self

    def price_features(self):
        for symbol, frame in self.series_set.items():
            frame['price_proportion'] =  abs(frame['o'] - frame['c']) / abs(frame['h'] - frame['l'])
            frame['close_proportion'] = (frame['h'] - frame['c']) / (frame['h'] - frame['l'])
            frame['open_proportion'] = (frame['h'] - frame['o']) / (frame['h'] - frame['l'])
        return self

    def relative_strength(self, comparative_frame, periods=[4]):
        self.rate_of_change(periods)
        comparative_frame = Tape(comparative_frame).rate_of_change(periods).concat()
        for period in periods:
            label1 = 'rs' + str(period)
            label2 = 'zrs' + str(period)
            for symbol, frame in self.series_set.items():
                frame[label1] = frame['roc' + str(period)] / comparative_frame['roc' + str(period)]
                frame[label2] = frame['rocz' + str(period)] / comparative_frame['rocz' + str(period)]
        return self

    def relative_rate_of_change(self, comparitive_frame, periods=14):
        label = 'rroc' + str(periods)
        comp = (comparitive_frame.c / comparitive_frame.shift(periods).c - 1)
        for symbol, frame in self.series_set.items():
            frame[label] = ( frame.c / frame.shift(periods).c - 1) - comp
        return self

    # gaps measured by volatility
    def gaps(self):
        label1 = 'gd'
        label2 = 'gu'
        for symbol, frame in self.series_set.items():
            frame[label1] = ((frame.l.shift(1) - frame.o) /  frame.l.shift(1)) / frame.v1.shift(1)
            frame[label2] = ((frame.o - frame.h.shift(1) / frame.o)) / frame.v1.shift(1)
        return self


    # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    def adx(self, periods=[14]):
        count = 0
        for period in periods:
            for symbol, frame in self.series_set.items():
                print(str(count) + ' ' + symbol)
                count = count + 1
                ranges = pd.DataFrame({'r': frame.h - frame.l, 'r1': abs(frame.h - frame.shift(1).c), 'r2': abs(frame.l - frame.shift(1).c)})
                frame['tr'] = ranges.max(axis=1)
                frame['plus_di'] = frame.h - frame.shift(1).h
                frame['minus_di'] = frame.shift(1).l - frame.l
                frame.loc[frame['plus_di'] < 0, 'plus_di'] = 0
                frame.loc[frame['minus_di'] < 0, 'minus_di'] = 0
                frame['tr14'] = frame['tr'].rolling(center=False,window=14).sum()
                frame['tr14s'] = np.nan
                frame['plus_di14'] = frame['plus_di'].rolling(center=False,window=14).sum()
                frame['plus_di14s'] = np.nan
                frame['minus_di14'] = frame['minus_di'].rolling(center=False,window=14).sum()
                frame['minus_di14s'] = np.nan
                y = 0
                previous_index = None
                for i, x in frame.iterrows():
                    if y == period:
                        frame.loc[i, 'tr14s'] = frame.loc[i, 'tr14']
                        frame.loc[i, 'plus_di14s'] = frame.loc[i, 'plus_di14']
                        frame.loc[i, 'minus_di14s'] = frame.loc[i, 'minus_di14']
                    if y > period:
                        frame.loc[i, 'tr14s'] = frame.loc[previous_index, 'tr14s'] - (frame.loc[previous_index, 'tr14s'] / 14) + frame.loc[i, 'tr']
                        frame.loc[i, 'plus_di14s'] = frame.loc[previous_index, 'plus_di14s'] - (frame.loc[previous_index, 'plus_di14s'] / 14) + frame.loc[i, 'plus_di']
                        frame.loc[i, 'minus_di14s'] = frame.loc[previous_index, 'minus_di14s'] - (frame.loc[previous_index, 'minus_di14s'] / 14) + frame.loc[i, 'minus_di']
                    previous_index = i
                    y = y + 1

                frame['+di14'] = (frame['plus_di14s'] / frame['tr14s']) * 100
                frame['-di14'] = (frame['minus_di14s'] / frame['tr14s']) * 100
                frame['dx'] = (abs((frame['+di14'] - frame['-di14'])) / (frame['+di14'] + frame['-di14'])) * 100

                frame['dxsum'] = frame['dx'].rolling(center=False,window=14).sum()
                y = 0
                previous_index = None
                for i, x in frame.iterrows():
                    if y == period * 2:
                        frame.loc[i, 'adx'] = frame.loc[i, 'dxsum']
                    if y > period * 2:
                        frame.loc[i, 'adx'] = ((frame.loc[previous_index, 'adx'] * 13) + frame.loc[i, 'dx']) / 14
                    previous_index = i
                    y = y + 1

        return self

    # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    def true_range_zscores(self, periods=30):
        label = 'trz' + str(periods)
        label2 = 'atr' + str(periods)
        for symbol, frame in self.series_set.items():
            ranges = pd.DataFrame({'r': frame.h - frame.l, 'r1': abs(frame.h - frame.shift(1).c), 'r2': abs(frame.l - frame.shift(1).c)})
            frame['tr'] = ranges.max(axis=1)
            frame[label2] = frame.tr.rolling(window=periods, center=False).mean()
            mean = frame.tr.rolling(window=periods, center=False).mean() #pd.rolling_mean(frame.tr, periods)
            std = frame.tr.rolling(window=periods, center=False).std()  #pd.rolling_std(frame.tr, periods)
            frame[label] = (frame.tr - mean) / std
        return self


    def volume_zscores(self, periods=[4], sample_size=252):
        for period in periods:
            label = 'vz' + str(period)
            label2 = 'vm' + str(period)
            for symbol, frame in self.series_set.items():
                mean = frame.v.rolling(window=sample_size,center=False).mean() #pd.rolling_mean(frame.v, periods)
                std = frame.v.rolling(window=sample_size,center=False).std()
                mean4 = frame.v.rolling(window=4,center=False).mean()
                frame[label2] = mean
                frame[label] = (mean4 - mean) / std
        return self

    def average_volume(self, window=250):
        label = 'av' + str(window)
        for symbol, frame in self.series_set.items():
            frame[label] = frame.v.rolling(window=window,center=False).mean()
        return self

    def merge_feature(self, feature_series, label):
        for symbol, frame in self.series_set.items():
            frame[label] = feature_series
        return self

    # https://www.incrediblecharts.com/indicators/linear_regression_indicator.php
    def linear_regression(self, periods=20):
        label = 'lr' + str(periods)
        X = range(1, periods + 1)
        X = sm.add_constant(X)
        for symbol, frame in self.series_set.items():
            frame[label] = pd.rolling_apply(frame['c'], periods, lambda y: sm.OLS(y,X).fit().fittedvalues[-1])
            frame[label + 'rs'] = pd.rolling_apply(frame['c'], periods, lambda y: sm.OLS(y,X).fit().rsquared)
            frame[label + 's'] = pd.rolling_apply(frame['c'], periods, lambda y: sm.OLS(y,X).fit().params[-1])
        return self

    def new_high_or_low(self, periods=252):
        label = 'nhl' + str(periods)
        def is_window_high_low(x):
            result = 0
            if x.min() == x[-1]: result = -1
            if x.max() == x[-1]: result = 1
            return result
        for symbol, frame in self.series_set.items():
            frame[label] = pd.rolling_apply(frame.c, periods, is_window_high_low)
        return self

    def range_properties(self, periods=30):
        label = 'rp' + str(periods)
        def range_proportion(x):
            result = 0
            y = x.max() - x.min()
            z = x.max() - x[-1]
            if z != 0:
                result = z / y
            return result
        for symbol, frame in self.series_set.items():
            frame[label] = frame.c.rolling(window=periods,center=False).apply(range_proportion)
        return self

    def rect(self, periods=[30], sample_size=252):
        for period in periods:
            label = 'rect' + str(period)
            for symbol, frame in self.series_set.items():
                frame[label + 'max'] = frame.h.rolling(window=period,center=False).max()
                frame[label + 'min'] = frame.l.rolling(window=period,center=False).min()
                frame[label + 'range'] = frame[label + 'max'] - frame[label + 'min']
                frame[label + 'c'] = (frame[label + 'max'] - frame.c) / frame[label + 'range']
                mean = frame[label + 'range'].rolling(window=sample_size,center=False).mean()
                std = frame[label + 'range'].rolling(window=sample_size,center=False).std()
                frame[label + 'rangez'] = (frame[label + 'range'] - mean) / std
        return self

    def hilo(self, periods=[3]):
        for period in periods:
            label1 = 're' + str(period)
            for symbol, frame in self.series_set.items():
                frame['hi'] = frame['h'].rolling(window=period,center=False).mean()
                frame['lo'] = frame['l'].rolling(window=period,center=False).mean()
        return self

    '''
    Below are different methods for measuring swing points
    https://www.investopedia.com/articles/technical/04/080404.asp
    '''

    def trade_swings(self, ticks=0):
        r = []
        for symbol, frame in self.series_set.items():
            position = 0
            exit = 0
            entry = 0
            pt = 0
            day = 0
            target = 0
            stop = 0
            for i, x in frame.iterrows():
                o = x['o']
                h = x['h']
                l = x['l']
                c = x['c']
                v = x['v']
                t = x['trend2']
                hi = x['hi']
                lo = x['lo']
                day = day + 1
                if position == 1:

                    if h > target and l > stop:
                        exit = target
                        trade = target / entry - 1
                        print(x.name, 'long exit target', c)
                        r.append(trade)
                        position = 0

                    if l < stop and h < target:
                        exit = stop
                        trade = stop / entry - 1
                        print(x.name, 'long exit stop', c)
                        r.append(trade)
                        position = 0

                    if l < stop and h > target:
                        print(x.name, 'inconclusive', c)
                        r.append(0)
                        position = 0

                    if day == 3:
                        exit = c
                        trade = c / entry - 1
                        print(x.name, 'long exit end', c)
                        r.append(trade)
                        position = 0

                if position == 0:
                    if x.entry2 > 0 and x.trend_change2 == 1 and x.trend3 == 1 and x.rect30rangez < 0 and x.rocz3 > 0 and x.maz200 < 0 and x.v1 > 0.015:
                        entry = x.entry2
                        target = x.entry2 + ((x.v1 + 1) * 3)
                        stop = x.entry2 - ((x.v1 + 1) * 2)
                        print(x.name, 'long entry', entry)
                        day = 0
                        position = 1

        return pd.Series(r)

    def retracements(self):
        count = 1
        for symbol, frame in self.series_set.items():
            print(str(count) + ' ' +symbol)
            count = count + 1
            slope = 0
            spl1 = 0
            spl2 = 0
            sph1 = 0
            frame['prt'] = np.nan
            for i, x in frame.iterrows():
                if x.sph2 > 0:
                    sph1 = x.sph2
                if x.spl2 > 0:
                    spl1 = spl2
                    spl2 = x.spl2

                if x.slope2 == 1 and sph1 - spl1 != 0:
                    frame.loc[i, 'prt'] = (sph1 - spl2) / (sph1 - spl1)

        return self

    def gann_swings(self):
        count = 1
        print('start')
        for symbol, frame in self.series_set.items():
            print(str(count) + ' ' +symbol)
            count = count + 1
            for z in range(2,4):
                label = str(z)
                successive_highs = 0
                successive_lows = 0
                trend = 0
                lentry = 0
                lexit = 0
                sentry = 0
                sexit = 0
                potential_swing_high = frame.iloc[0].h
                potential_swing_low = frame.iloc[0].l
                potential_swing_high_index = frame.index[0]
                potential_swing_low_index = frame.index[0]
                lentry_index = frame.index[0]
                sentry_index = frame.index[0]
                swing_point_low = frame.iloc[0].l
                previous_spl = swing_point_low
                swing_point_high = frame.iloc[0].h
                frame['inside_day'] = np.nan
                frame['outside_day'] = np.nan
                frame['sph' + label] = np.nan
                frame['spl' + label] = np.nan
                frame['swings' + label] = np.nan
                frame['succesive_highs'] = np.nan
                frame['succesive_lows'] = np.nan
                frame['trend' + label] = np.nan
                frame['long' + label] = np.nan
                frame['short' + label] = np.nan
                frame['trend_change' + label] = np.nan
                frame['gap' + label] = False
                slope = 0
                frame['slope' + label] = 0
                reference_bar = frame.iloc[0]
                lf = True
                sf = True
                for i, x in frame.iterrows():
                    l = x['l']
                    h = x['h']

                    # inside days
                    if i == 0: reference_bar = frame.loc[i]
                    inside_day = reference_bar.h > x.h and reference_bar.l < x.l
                    if inside_day:
                        if x.o > x.c:
                            frame.loc[i, 'inside_day'] = 1
                        else:
                            frame.loc[i, 'inside_day'] = 2


                    # outside days
                    outside_day = reference_bar.h < x.h and reference_bar.l > x.l
                    if outside_day:
                        if x.o > x.c:
                            frame.loc[i, 'outside_day'] = 1
                        else:
                            frame.loc[i, 'outside_day'] = 2

                    # succesive high and low count
                    if not inside_day:
                        if outside_day:
                            if x.o > x.c:
                                successive_lows = 1
                                successive_highs = 0
                            else:
                                successive_lows = 0
                                successive_highs = 1
                        else:
                            if x.h > reference_bar.h:
                                successive_highs = successive_highs + 1
                                successive_lows = 0
                            if x.l < reference_bar.l:
                                successive_highs = 0
                                successive_lows = successive_lows + 1

                        reference_bar = frame.loc[i]

                    frame.loc[i, 'succesive_highs'] = successive_highs
                    frame.loc[i, 'succesive_lows'] = successive_lows

                    if slope != 1 and successive_highs >= z:
                        sf = True
                        previous_spl = swing_point_low
                        swing_point_low = frame.loc[potential_swing_low_index].l
                        frame.loc[potential_swing_low_index, 'spl' + label] = swing_point_low
                        frame.loc[potential_swing_low_index, 'swings' + label] = swing_point_low
                        potential_swing_high_index = i
                        potential_swing_high = x.h
                        slope = 1

                    if slope != -1 and successive_lows >= z:
                        lf = True
                        previous_sph = swing_point_high
                        swing_point_high = frame.loc[potential_swing_high_index].h
                        frame.loc[potential_swing_high_index, 'sph' + label] = swing_point_high
                        frame.loc[potential_swing_high_index, 'swings' + label] = swing_point_high
                        potential_swing_low_index = i
                        potential_swing_low = x.l
                        slope = -1

                    if slope == -1 and x.l < potential_swing_low:
                        potential_swing_low_index = i
                        potential_swing_low = x.l
                    if slope == 1 and x.h > potential_swing_high:
                        potential_swing_high_index = i
                        potential_swing_high = x.h

                    if x.h > swing_point_high and lf == True:
                        lf = False
                        if x.o < swing_point_high:
                            frame.loc[i, 'long' + label] = swing_point_high
                        else:
                            frame.loc[i, 'long' + label] = x.o
                            frame.loc[i, 'gap' + label] = True

                    if x.l < swing_point_low and sf == True:
                        sf = False
                        if x.o > swing_point_low:
                            frame.loc[i, 'short' + label] = swing_point_low
                        else:
                            frame.loc[i, 'short' + label] = x.o
                            frame.loc[i, 'gap' + label] = True


                    if trend != 1 and x.h > swing_point_high:
                        trend = 1
                        frame.loc[i, 'trend_change' + label] = 1
                    if trend != -1 and x.l < swing_point_low:
                        trend = -1
                        frame.loc[i, 'trend_change' + label] = -1

                    frame.loc[i, 'trend' + label] = trend
                    frame.loc[i, 'slope' + label] = slope

                frame['swings' + label] = frame['swings' + label].interpolate(method='time')

        return self

    def swing_points(self, periods=5):
        label = 'sp' + str(periods)
        for symbol, frame in self.series_set.items():
            successive_highs = 1
            successive_lows = 1
            potential_swing_high = frame.iloc[0].c
            potential_swing_low = frame.iloc[0].c
            potential_swing_high_index = frame.index[0]
            potential_swing_low_index = frame.index[0]
            swing_point_low = frame.iloc[0].c
            swing_point_high = frame.iloc[0].c
            trend = 0
            frame['sph'] = np.nan
            frame['spl'] = np.nan
            frame['swings'] = np.nan
            previous = None
            slope = 0
            for i, x in frame.iterrows():
                l = x['c']
                h = x['c']
                if successive_highs > periods:
                    if previous is not 'High':
                        swing_point_high = frame.loc[potential_swing_high_index].c
                        frame.loc[potential_swing_high_index, 'sph'] = swing_point_high
                        frame.loc[potential_swing_high_index, 'swings'] = swing_point_high
                        previous = 'High'
                    successive_highs = 1
                    potential_swing_high = h
                    potential_swing_high_index = i
                elif h >= potential_swing_high:
                    potential_swing_high = h
                    potential_swing_high_index = i
                    successive_highs = 1
                elif h < potential_swing_high:
                    successive_highs = successive_highs + 1

                if successive_lows > periods:
                    if previous is not 'Low':
                        swing_point_low = frame.loc[potential_swing_low_index].c
                        frame.loc[potential_swing_low_index, 'spl'] = swing_point_low
                        frame.loc[potential_swing_low_index, 'swings'] = swing_point_low
                        previous = 'Low'
                    successive_lows = 1
                    potential_swing_low = l
                    potential_swing_low_index = i
                elif l <= potential_swing_low:
                    potential_swing_low = l
                    potential_swing_low_index = i
                    successive_lows = 1
                elif l > potential_swing_low:
                    successive_lows = successive_lows + 1
            frame.swings = frame.swings.interpolate(method='time')

        return self

    def swing_breakouts(self):
        for symbol, frame in self.series_set.items():
            spl = 0
            sph = 0
            lb = False
            hb = False
            direction = 0
            frame['position'] = np.nan
            for i, x in frame.iterrows():
                #set breaks
                if x['c'] > sph and sph != 0 and hb == False:
                    hb = True
                    frame.loc[i, 'sphb'] = 1
                    direction = 1
                if x['c'] < spl and spl != 0 and lb == False:
                    lb = True
                    frame.loc[i, 'splb'] = 1
                    direction = 0
                #set points
                if x['spl'] > 0:
                    spl = x['spl']
                    if lb: lb = False
                if x['sph'] > 0:
                    sph = x['sph']
                    if hb: hb = False

                frame.loc[i, 'position'] = frame.loc[i, 'r1'] * direction


        return self

    def trend(self):
        for symbol, frame in self.series_set.items():
            spla = 0
            splb = 0
            spha = 0
            sphb = 0
            for i, x in frame.iterrows():
                if x['spl'] > 0:
                    splb = spla
                    spla = x['spl']
                if x['sph'] > 0:
                    sphb = spha
                    spha = x['sph']

                if spha != 0 and sphb != 0:
                    if spha > sphb:
                        frame.loc[i, 'hh'] = 1
                    else:
                       frame.loc[i, 'hh'] = -1

                if spla != 0 and splb != 0:
                    if spla < splb:
                        frame.loc[i, 'll'] = 1
                    else:
                       frame.loc[i, 'll'] = -1

        return self

def get_sharpe(s, n=252):
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    return sharpe_r

def get_stats(s, n=252):
    s = s.dropna()
    wins = len(s[s>0])
    losses = len(s[s<0])
    evens = len(s[s==0])

    if losses > 0:
        win_r = round(wins/losses, 5)
        mean_w = round(s[s>0].mean(), 5)
        mean_l = round(s[s<0].mean(), 5)
    else:
        win_r = 100
        mean_l = 0
        mean_w = 100
    mean_trd = round(s.mean(), 5)
    sd = round(np.std(s), 5)
    max_l = round(s.min(), 5)
    max_w = round(s.max(), 5)
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    cnt = len(s)
    print('Trades:', cnt,\
          '\nWins:', wins,\
          '\nLosses:', losses,\
          '\nBreakeven:', evens,\
          '\nWin/Loss Ratio', win_r,\
          '\nMean Win:', mean_w,\
          '\nMean Loss:', mean_l,\
          '\nMean', mean_trd,\
          '\nStd Dev:', sd,\
          '\nMax Loss:', max_l,\
          '\nMax Win:', max_w,\
          '\nSharpe Ratio:', sharpe_r)
