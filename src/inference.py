import torch
import joblib
import numpy as np
import pandas as pd
import json
from flowpredictor import FlowPredictor
from seq2seq_predictor import LSTMSeq2SeqAttnModel  # ← 추가
from scipy.signal import savgol_filter

# ===== Seq2Seq+Attention 대상 배수지 =====
SEQ2SEQ_RESERVOIRS = {4, 7, 13}


#=========Resv Service========
#=========Resv Service========
#=========Resv Service========
class ReservoirInferenceService:
    def __init__(self, reservoir_configs, input_dim, window_size=180):
        self.window_size = window_size
        self.input_dim = input_dim

        self.models = {}
        self.scalers_x = {}
        self.scalers_y = {}

        for name, paths in reservoir_configs.items():
            try:
                # ===== 배수지 4, 7, 13: Seq2Seq+Attention =====
                if name in SEQ2SEQ_RESERVOIRS:
                    checkpoint = torch.load(
                        paths['weights'], map_location='cpu', weights_only=False
                    )
                    model = LSTMSeq2SeqAttnModel(
                        input_size=len(checkpoint['feature_cols']),
                        hidden_size=128,
                        num_layers=2,
                        output_size=checkpoint['output_time'],
                        embed_dim=16,
                        dropout=0.2,
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.models[name] = model
                    # 체크포인트 내장 scaler 사용 (joblib 불필요)
                    self.scalers_x[name] = checkpoint['scalers']
                    self.scalers_y[name] = checkpoint['scalers']['value']
                    print(f"[resv {name}] Seq2Seq+Attention 모델 로드 완료")
                    continue

                # ===== 기존 배수지 (8, 10, 15): FlowPredictor =====
                with open(paths['config'], 'r') as f:
                    config = json.load(f)

                self.scalers_x[name] = joblib.load(paths['scaler_x'])
                self.scalers_y[name] = joblib.load(paths['scaler_y'])

                model = FlowPredictor(
                    input_dim=4,
                    hidden_dim=config['units'],
                    output_dim=config['forecast_size'],
                    dropout=config['dropout']
                )
                model.load_state_dict(torch.load(paths['weights'], map_location=torch.device('cpu')))
                model.eval()
                self.models[name] = model

            except FileNotFoundError as e:
                print(f"[resv {name}] 파일 없음, 스킵: {e}")
                continue

    def predict(self, reservoir_name, raw_data):
        model = self.models[reservoir_name]

        # ===== 배수지 4, 7, 13: Seq2Seq+Attention (자체 정규화) =====
        if reservoir_name in SEQ2SEQ_RESERVOIRS:
            scalers = self.scalers_x[reservoir_name]
            feature_cols = ['resv_flow', 'temperature', 'precipitate', 'humidity',
                           'time_sin', 'time_cos', 'dow_sin', 'dow_cos',
                           'season_sin', 'season_cos']

            features = raw_data[feature_cols].values.astype(np.float32).copy()

            # 정규화 (scale_cols만, sin/cos는 이미 [0,1])
            col_to_scaler = {
                'resv_flow': 'value',
                'temperature': 'temperature',
                'precipitate': 'rainfall',
                'humidity': 'humidity',
            }
            for col_name, scaler_key in col_to_scaler.items():
                idx = feature_cols.index(col_name)
                d_min = float(scalers[scaler_key]['min'])
                d_max = float(scalers[scaler_key]['max'])
                if d_max - d_min > 0:
                    features[:, idx] = (features[:, idx] - d_min) / (d_max - d_min)

            # 마지막 72분만 사용 (window_size=180이 들어와도 OK)
            input_tensor = torch.FloatTensor(features[-72:]).unsqueeze(0)

            with torch.no_grad():
                prediction = model(input_tensor).cpu().numpy().flatten()

            # 역정규화
            val_min = float(scalers['value']['min'])
            val_max = float(scalers['value']['max'])
            return (prediction * (val_max - val_min) + val_min).reshape(1, -1)

        # ===== 기존 배수지 (8, 10, 15): FlowPredictor =====
        scaler_x = self.scalers_x[reservoir_name]
        scaler_y = self.scalers_y[reservoir_name]

        n_min = self.window_size
        raw_data.loc[:, 'resv_flow'] = savgol_filter(raw_data['resv_flow'], window_length=31, polyorder=1)
        scaled_data = scaler_x.transform(raw_data)
        input_tensor = torch.FloatTensor(scaled_data).view(1, n_min, 4)

        with torch.no_grad():
            prediction = model(input_tensor)

        out = scaler_y.inverse_transform(prediction.cpu().numpy())
        return out


#=========Pump Service========
#=========Pump Service========
#=========Pump Service========
class PumpOptimizationService:
    def __init__(self):
        self.PUMP_PERFORMANCE = {1: 425.8, 2: 715.3, 3: 902.5}
        self.PUMP_POWER_KW = 150

    def optimize(self, df, info_df):
        results = []
        timestamps = sorted(df['timestamp'].unique())

        current_levels = {
            rid: df[df['facility_id'] == rid]['level'].iloc[0]
            for rid in info_df['facility_id']
        }

        current_pumps = 2
        last_change_time = -90
        MIN_HOLDING_TIME = 90

        for i, ts in enumerate(timestamps):
            ts_pd = pd.Timestamp(ts)
            load_type, price = self.get_load_type(ts)
            curr_rows = df[df['timestamp'] == ts]

            danger_low = any(current_levels[rid] <= info_df.loc[info_df['facility_id']==rid, 'safety_min'].values[0] + 0.2 for rid in current_levels)

            if (i - last_change_time >= MIN_HOLDING_TIME) or danger_low:
                if danger_low:
                    new_pumps = 3
                elif load_type == "LOW":
                    target_hour = 9
                    target_end = ts_pd.replace(hour=target_hour, minute=0, second=0)
                    if ts_pd.hour >= target_hour:
                        target_end += pd.Timedelta(days=1)

                    remaining_mins = (target_end - ts_pd).total_seconds() / 60

                    if remaining_mins > 0:
                        total_required_vol = 0
                        TARGET_RATIO = 0.85

                        for _, res in info_df.iterrows():
                            f_id = res['facility_id']
                            fill_vol = (res['safety_max'] * TARGET_RATIO - current_levels[f_id]) * res['estimated_area']
                            future_demand = df[(df['facility_id'] == f_id) & (df['timestamp'] > ts_pd) & (df['timestamp'] <= target_end)]['flow_out'].sum() / 60
                            total_required_vol += (max(0, fill_vol) + future_demand)

                        required_flow_hr = (total_required_vol / remaining_mins) * 60
                        total_sim_dist_rate = info_df['dist_rate'].sum()

                        p2_capacity = self.PUMP_PERFORMANCE[2] * total_sim_dist_rate
                        p3_capacity = self.PUMP_PERFORMANCE[3] * total_sim_dist_rate

                        if required_flow_hr > p3_capacity * 0.95:
                            new_pumps = 3
                        elif required_flow_hr > p2_capacity * 0.9:
                            new_pumps = 2
                        else:
                            new_pumps = 1
                elif load_type == "HIGH":
                    new_pumps = 1
                else:
                    new_pumps = 2
                if new_pumps != current_pumps:
                    current_pumps = new_pumps
                    last_change_time = i

            fill_priority = {}
            total_priority = 0
            for _, res in info_df.iterrows():
                f_id = res['facility_id']
                gap = max(0.01, res['safety_max'] - current_levels[f_id])
                priority = res['dist_rate'] * (gap ** 2)
                fill_priority[f_id] = priority
                total_priority += priority

            theoretical_inflow_min = self.PUMP_PERFORMANCE[current_pumps] / 60
            active_resvs = [rid for rid, lvl in current_levels.items() if lvl < info_df.loc[info_df['facility_id']==rid, 'safety_max'].values[0]]
            sum_active_priority = sum(fill_priority[rid] for rid in active_resvs)

            sum_active_dist_rate = info_df[info_df['facility_id'].isin(active_resvs)]['dist_rate'].sum()
            actual_inflow_min = theoretical_inflow_min * sum_active_dist_rate
            spill = theoretical_inflow_min - actual_inflow_min

            for _, res in info_df.iterrows():
                f_id = res['facility_id']
                q_out = curr_rows[curr_rows['facility_id'] == f_id]['flow_out'].values[0] / 60 if not curr_rows[curr_rows['facility_id'] == f_id].empty else 0

                if f_id in active_resvs and sum_active_priority > 0:
                    dynamic_rate = fill_priority[f_id] / sum_active_priority
                    q_in = actual_inflow_min * dynamic_rate
                else:
                    q_in = 0

                new_level = current_levels[f_id] + (q_in - q_out) / res['estimated_area']

                current_levels[f_id] = max(
                    0.1,
                    min(new_level, res['safety_max'])
                )

            results.append({
                'timestamp': ts,
                'active_pumps': current_pumps,
                'sim_levels': current_levels.copy(),
                'sim_cost': (current_pumps * self.PUMP_POWER_KW / 60) * price,
                'spill_m3_per_min': spill
            })

        return pd.DataFrame(results)

    def get_load_type(self, ts):
        hour = ts.hour
        if 23 <= hour or hour < 9:
            return "LOW", 70.0
        elif (10 <= hour < 12) or (17 <= hour < 20) or (22 <= hour < 23):
            return "HIGH", 200.0
        else:
            return "MID", 130.0
