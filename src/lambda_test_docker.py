#Import modules
import requests 

event = {
  "phone": {
    "Unnamed: 0": 1.0,
    "battery_power": 1021,
    "blue": 1,
    "clock_speed": 0.5,
    "dual_sim": 1,
    "fc": 0,
    "four_g": 1,
    "int_memory": 53,
    "m_dep": 0.7,
    "mobile_wt": 136,
    "n_cores": 3,
    "pc": 6,
    "px_height": 905,
    "px_width": 1988,
    "ram": 2631,
    "sc_h": 17,
    "sc_w": 3,
    "talk_time": 7,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 0
  }
}

url = 'http://localhost:9000/2015-03-31/functions/function/invocations'

response = requests.post(url, json=event)
print(response.json())