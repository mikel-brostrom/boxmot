from boxmot import BotSort, KalmanAmsConfig, KalmanConfig, KalmanNoiseConfig, ReIDConfig
import numpy
import numpy as np

tracker = BotSort(
    reid=ReIDConfig(
        model="osnet-x0-25-msmt17",
        device="cpu",
        precision="fp32",
        preprocessing="resize",
        batch_size=64,
        image_size=(256, 128),  # Height, width
        embedding_dim=None,    # Infer from model
        allow_download=True,
    ),
    kalman=KalmanConfig(
        variable_dt=False,
        adaptive_kf=False,
        variable_dt=False,
        ams=KalmanAmsConfig(),
        noise=KalmanNoiseConfig(
            process_position_scale=1.0,
            process_velocity_scale=1.0,
            measurement_noise_scale=1.0,
            initial_position_scale=1.0,
            initial_velocity_scale=1.0,
            reference_dt_s=1 / 30,
            time_unit=None,  # Derive from variable_dt
        ),
    ),
    use_embeddings=True,
    use_cmc=True,
    cmc_method="ecc",
    per_class=False,
)