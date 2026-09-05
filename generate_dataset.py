import numpy as np
import pandas as pd

rng = np.random.default_rng()
n = 1000

battery_power = rng.integers(500, 2000, n)
blue = rng.integers(0, 2, n)
clock_speed = np.round(rng.uniform(0.5, 3.0, n), 1)
dual_sim = rng.integers(0, 2, n)
fc = rng.integers(0, 20, n)
four_g = rng.integers(0, 2, n)
int_memory = rng.integers(2, 65, n)
m_dep = np.round(rng.uniform(0.1, 1.0, n), 1)
mobile_wt = rng.integers(80, 200, n)
n_cores = rng.integers(1, 9, n)
pc = rng.integers(0, 21, n)
px_height = rng.integers(0, 1961, n)
px_width = rng.integers(500, 2000, n)
ram = rng.integers(256, 4000, n)
sc_h = rng.integers(5, 20, n)
sc_w = rng.integers(0, 18, n)
talk_time = rng.integers(2, 21, n)
three_g = rng.integers(0, 2, n)
touch_screen = rng.integers(0, 2, n)
wifi = rng.integers(0, 2, n)

score = (
    (ram / 4000) * 0.45
    + (battery_power / 2000) * 0.15
    + (px_width * px_height / (2000 * 1961)) * 0.15
    + (int_memory / 64) * 0.1
    + (n_cores / 8) * 0.05
    + (pc / 20) * 0.05
    + (clock_speed / 3.0) * 0.05
)

quantiles = np.quantile(score, [0.25, 0.5, 0.75])
price_range = np.digitize(score, quantiles)

df = pd.DataFrame({
    "battery_power": battery_power,
    "blue": blue,
    "clock_speed": clock_speed,
    "dual_sim": dual_sim,
    "fc": fc,
    "four_g": four_g,
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "three_g": three_g,
    "touch_screen": touch_screen,
    "wifi": wifi,
    "price_range": price_range,
})

df.to_excel("mobile_price_data.xlsx", index=False, sheet_name="mobile_price_data")
df.to_csv("mobile_price_data.csv", index=False)
print(df.shape)
print(df["price_range"].value_counts())
