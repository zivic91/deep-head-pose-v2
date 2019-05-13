import json
import matplotlib.pyplot as plt
import numpy as np

a = json.load(open('error_stats_master50.json', 'r'))

# Plot yaw errors
yaw_values = np.array(a['yaw_value'])
yaw_errors = np.array(a['yaw_error'])
yaw_values_bin = yaw_values // 3

plt.scatter(yaw_values, yaw_errors, s = 1)
plt.show()

# Plot pitch errors
pitch_values = np.array(a['pitch_value'])
pitch_errors = np.array(a['pitch_error'])
pitch_values_bin = pitch_values // 3

plt.scatter(pitch_values, pitch_errors, s = 1)
plt.show()

# Plot roll errors
roll_values = np.array(a['roll_value'])
roll_errors = np.array(a['roll_error'])
roll_values_bin = roll_values // 3

plt.scatter(roll_values, roll_errors, s = 1)
plt.show()

# idx_yaw = [i for i in range(len(yaw_values)) if abs(yaw_values[i]) <= 30]
# print(np.mean(yaw_errors[idx_yaw]))
#
# idx_pitch = [i for i in range(len(pitch_values)) if abs(pitch_values[i]) <= 30]
# print(np.mean(pitch_errors[idx_pitch]))
#
# idx_roll = [i for i in range(len(roll_values)) if abs(roll_values[i]) <= 30]
# print(np.mean(roll_errors[idx_roll]))
