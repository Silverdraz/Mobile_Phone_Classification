"""
Python Script for calling the lambda function locally with event
"""

#Import modules
import lambda_function #module to simulate invocation of lambda function

#sample example to test lambda
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


result = lambda_function.lambda_handler(event, None)
print(result)