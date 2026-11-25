# Next-Hour Traffic Forecasting

Can the last three hourly vehicle counts, plus the clock time of the coming hour, predict next hour's traffic better than simple baselines? This project tests that on real highway data using a time-based split, so the model is only ever evaluated on data from after its training period.

**Data:** [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (I-94 westbound, Minnesota, 2012-2018), credit John Hogue / UCI, CC BY 4.0. One measuring station only. The notebook downloads the data itself.

## Approach
- One observation per hour: duplicate timestamps are checked for agreement and collapsed, and missing hours are not filled in.
- Features: counts at hours `t`, `t-1`, `t-2`, plus the next hour's hour-of-day and weekday. Next-hour weather is excluded because it would not be known in advance.
- Time-based split: train on targets before 2017, choose tree depth on 2017, test once on 2018.
- Models: repeat the last count, median count for that weekday and hour, and a depth-tuned decision tree regressor (scikit-learn).

## Results (2018 test set, 6,497 hours)

| Model | MAE | RMSE | Weekday-peak MAE |
|---|---:|---:|---:|
| Repeat last count | 587.9 | 812.2 | 759.2 |
| Typical weekday-hour | 258.7 | 480.6 | 378.9 |
| **Decision tree** | **172.4** | **261.8** | **245.3** |

Errors are in vehicles per hour. The tree cuts MAE by 33.4% against the stronger baseline and beats it in every 2018 month tested. Train, validation and test MAE are 164.9, 181.3 and 172.4, so there is no sign of heavy overfitting.

## Limitations
- One station, 2012-2018 data. Present-day use would need fresh data and revalidation.
- Weekday-rush-hour errors are higher (245.3 vs 156.5 for other hours), and individual hours can be badly wrong.
- Counts alone say nothing about speed, congestion, accidents or routes. Hours with missing recent counts were excluded.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook notebooks/next_hour_traffic_forecasting.ipynb
```
Python 3.10+. The notebook needs internet access to fetch the dataset.

## Possible next steps
Gradient-boosted trees, weather and holiday features, multi-hour horizons, and handling missing hours.
