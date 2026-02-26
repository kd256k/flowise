import time
import json
import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from inference import ReservoirInferenceService, PumpOptimizationService
from sqlalchemy import create_engine, text

window_size=180
forecast_size=15
total_forecast_size=360
input_dim = 10

# ===== Seq2Seq+Attention 대상 배수지 =====
SEQ2SEQ_RESERVOIRS = {4, 7, 13}

configs = {
    4: {"weights": "../models/a_resv_seq2seq.pth",         # ← Seq2Seq 모델
           "scaler_x": "../models/a_resv_scaler_x.pkl",    # 미사용 (체크포인트 내장)
           "scaler_y": "../models/a_resv_scaler_y.pkl",    # 미사용 (체크포인트 내장)
           "config": "../models/a_resv_config.json"
           },
    7: {"weights": "../models/d_resv_seq2seq.pth",         # ← Seq2Seq 모델
           "scaler_x": "../models/d_resv_scaler_x.pkl",    # 미사용 (체크포인트 내장)
           "scaler_y": "../models/d_resv_scaler_y.pkl",    # 미사용 (체크포인트 내장)
           "config": "../models/d_resv_config.json"
           },
    8: {"weights": "../models/e_resv_flow_model.pth", 
            "scaler_x": "../models/e_resv_scaler_x.pkl",
            "scaler_y": "../models/e_resv_scaler_y.pkl",
            "config": "../models/e_resv_config.json"
            },
    10: {"weights": "../models/g_resv_flow_model.pth", 
          "scaler_x": "../models/g_resv_scaler_x.pkl",
          "scaler_y": "../models/g_resv_scaler_y.pkl",
          "config": "../models/g_resv_config.json"
          },
    13: {"weights": "../models/j_resv_seq2seq.pth",        # ← Seq2Seq 모델
           "scaler_x": "../models/j_resv_scaler_x.pkl",    # 미사용 (체크포인트 내장)
           "scaler_y": "../models/j_resv_scaler_y.pkl",    # 미사용 (체크포인트 내장)
           "config": "../models/j_resv_config.json"
           },
    15: {"weights": "../models/l_resv_flow_model.pth", 
           "scaler_x": "../models/l_resv_scaler_x.pkl",
           "scaler_y": "../models/l_resv_scaler_y.pkl",
           "config": "../models/l_resv_config.json"
           }
}
resv_service = ReservoirInferenceService(configs, input_dim=input_dim, window_size=window_size)

def get_mysql_engine():
    try:
        user = os.environ.get('MYSQL_USER', 'user')
        password = os.environ.get('MYSQL_PASSWORD', 'password')
        host = os.environ.get('MYSQL_HOST', 'localhost')
        port = int(os.environ.get('MYSQL_PORT', '3306'))
        database = os.environ.get('MYSQL_DATABASE', 'database')
        
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        
        engine = create_engine(url)
        return engine
    except Exception as e:
        print(f'MySQL Engine creation error: {e}')
        sys.exit(1)


# ===== Seq2Seq 대상 배수지 전용: Flow 전처리 (학습과 동일) =====
def _preprocess_flow_seq2seq(series: pd.Series) -> pd.Series:
    """Flow 전처리 — IQR → interpolation → Savgol(51,2)."""
    s = series.copy()

    # 1. IQR 이상치 + 음수 + 급변동 → NaN
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    iqr_mask = (s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)
    negative_mask = s < 0
    diff = s.diff().abs()
    spike_threshold = diff.quantile(0.999)
    spike_mask = diff > spike_threshold
    s[iqr_mask | negative_mask | spike_mask] = np.nan

    # 2. Linear interpolation + clip
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.clip(lower=0)

    # 3. Savitzky-Golay filter (학습: window=51, polyorder=2)
    if len(s) >= 51:
        s = pd.Series(
            savgol_filter(s.values, window_length=51, polyorder=2),
            index=s.index
        )
        s = s.clip(lower=0)

    return s


def get_latest_window(resv: int, start_date):
    engine = get_mysql_engine()

    # 6시간(360분) 예측: 입력윈도우(525) + 검증데이터(360) + 전처리버퍼 = 1000
    row_limit = 1000

    query = text(f"""
        SELECT 
            r.collected_at, 
            r.flow_out as resv_flow,
            w.temperature,
            w.rainfall as precipitate,
            w.humidity
        FROM reservoir_minutely r
        JOIN weather w ON r.collected_at = w.collected_at
        WHERE r.facility_id = :resv 
        AND r.collected_at >= :start_date
        LIMIT {row_limit}
    """)
    
    try: 
        params = {"resv": resv, "start_date": start_date}
        df = pd.read_sql(query, engine, params=params)
        
        if df.empty:
            return None, None, None

        df['collected_at'] = pd.to_datetime(df['collected_at'])
        df = df.sort_values('collected_at').reset_index(drop=True)

        # ===== 배수지 4, 7, 13: Seq2Seq 전용 전처리 =====
        if resv in SEQ2SEQ_RESERVOIRS:
            # Flow: IQR → interpolation → Savgol(51,2)  ← 학습과 동일
            df['resv_flow'] = _preprocess_flow_seq2seq(df['resv_flow'])

            # Rainfall: 누적값 → diff → clip(0)  ← 학습과 동일
            rainfall_diff = df['precipitate'].diff().fillna(0)
            df['precipitate'] = rainfall_diff.clip(lower=0)

            # Cyclical features (6개)
            t = df['collected_at']
            minute_of_day = t.dt.hour * 60 + t.dt.minute
            df['time_sin'] = 0.5 * np.sin(2 * np.pi * minute_of_day / 1440) + 0.5
            df['time_cos'] = 0.5 * np.cos(2 * np.pi * minute_of_day / 1440) + 0.5

            dow = t.dt.dayofweek
            df['dow_sin'] = 0.5 * np.sin(2 * np.pi * dow / 7) + 0.5
            df['dow_cos'] = 0.5 * np.cos(2 * np.pi * dow / 7) + 0.5

            doy = t.dt.dayofyear
            df['season_sin'] = 0.5 * np.sin(2 * np.pi * doy / 365.25) + 0.5
            df['season_cos'] = 0.5 * np.cos(2 * np.pi * doy / 365.25) + 0.5

            columns = ['resv_flow', 'temperature', 'precipitate', 'humidity',
                       'time_sin', 'time_cos', 'dow_sin', 'dow_cos',
                       'season_sin', 'season_cos']

            train_df = df[:-total_forecast_size]
            val_df = df['resv_flow'][-total_forecast_size:].values
            return train_df[columns], train_df['collected_at'].values[-1], val_df

        # ===== 기존 배수지 (8, 10, 15): 4-feature =====
        else:
            train_df = df[:-total_forecast_size]
            val_df = df['resv_flow'][-total_forecast_size:].values
            columns = ['resv_flow', 'temperature', 'precipitate', 'humidity']
            return train_df[columns], train_df['collected_at'].values[-1], val_df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None
    finally:
        engine.dispose()


def format_to_json(prediction, last_input_time, val_df):
    last_input_time = pd.to_datetime(last_input_time)
    prediction = prediction.flatten()

    accuracy = np.mean( (1- np.abs(val_df - prediction) / (val_df+0.01)) )

    json_for_redis = json.dumps(prediction.tolist())
    date_for_redis = str(last_input_time)
    accuracy_for_redis = str(accuracy)

    print(json_for_redis)

    return json_for_redis, date_for_redis, accuracy_for_redis

def run_generator(task_id: int, suzy: int, date: str | None = None):
    start_date = date if date else "2024-01-02 00:01"
    print(f"[task_id : {task_id}] Monitoring {suzy}-reservoir flow... (start_date={start_date})")
    print(f"Cycle started at {time.ctime()}")
    try:
        input_window, last_input_time, val_df = get_latest_window(suzy, start_date=start_date)
        prediction = resv_service.predict(suzy, input_window[:window_size])
        for i in range(1, total_forecast_size // forecast_size):
            prediction = np.concatenate((
                prediction,
                resv_service.predict(suzy, input_window[i*forecast_size : window_size + i*forecast_size])
            ))

        json_pred, json_date, json_accuracy = format_to_json(prediction, last_input_time, val_df)

        print(f"Prediction from {json_date}")
        print(f"Cycle complete at {time.ctime()}")

        return json_pred, json_date, json_accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        raise

#=======================Pump=========================
#=======================Pump=========================
#=======================Pump=========================
pump_service = PumpOptimizationService()

def get_pump_input(start_time):
    engine = get_mysql_engine()
    try:
        resv_info = pd.read_sql(
            """
            SELECT i.facility_id,
                i.area AS estimated_area,
                i.min_level AS safety_min,
                i.max_level AS safety_max
            FROM reservoir_info i
            WHERE i.area > 0
            """,
            engine
        )
        if resv_info.empty:
            return None, None

        valid_ids = resv_info['facility_id'].tolist()

        query_string = f"""
            SELECT r.collected_at AS timestamp,
                r.facility_id,
                r.level,
                r.flow_out,
                t.press_out_1,
                t.press_out_2,
                t.press_out_3,
                t.press_out_4
            FROM reservoir_minutely r
            LEFT JOIN treatment_minutely t
                ON r.collected_at = t.collected_at
                AND t.facility_id = 1
            WHERE r.facility_id IN ({','.join(map(str, valid_ids))})
            AND r.collected_at >= :start_time
            LIMIT 1440
        """
        query = text(query_string)
        params = {"start_time":start_time}
        df_all = pd.read_sql(query, engine, params=params)
        if df_all.empty:
            return None, None

        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])

        pump_cols = ['press_out_1', 'press_out_2', 'press_out_3', 'press_out_4']
        df_all['actual_pumps'] = (df_all[pump_cols] >= 2.0).sum(axis=1)

        avg_outflow = (
            df_all.groupby('facility_id')['flow_out']
            .mean()
            .reset_index()
        )
        avg_outflow['dist_rate'] = avg_outflow['flow_out'] / avg_outflow['flow_out'].sum()

        resv_info = pd.merge(
            resv_info,
            avg_outflow[['facility_id', 'dist_rate']],
            on='facility_id'
        )
        return df_all, resv_info
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None
    finally:
        engine.dispose()

def run_optimizer(start_time:str):
    print(f"Optimizing pump config...")
    print(f"Cycle started at {time.ctime()}")
    try:
        df_all, resv_info = get_pump_input(start_time)
        optimization = pump_service.optimize(df_all, resv_info)
        json_result = optimization.to_json(orient='records', date_format='iso')
        return json_result
    
    except Exception as e:
        print(f"Error: {e}")
