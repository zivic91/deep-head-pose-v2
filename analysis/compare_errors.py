import json
import matplotlib.pyplot as plt
import numpy as np

a = json.load(open('error_stats_hopenet2.json', 'r'))
b = json.load(open('error_stats_master50.json', 'r'))

# Plot yaw errors
yaw_values_a = np.array(a['yaw_value'])
yaw_errors_a = np.array(a['yaw_error'])

yaw_values_b = np.array(b['yaw_value'])
yaw_errors_b = np.array(b['yaw_error'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(yaw_values_a, yaw_errors_a, s=1, c='blue')
ax1.scatter(yaw_values_b, yaw_errors_b, s=1, c='red')
plt.show()

# Plot pitch errors
pitch_values_a = np.array(a['pitch_value'])
pitch_errors_a = np.array(a['pitch_error'])

pitch_values_b = np.array(b['pitch_value'])
pitch_errors_b = np.array(b['pitch_error'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(pitch_values_a, pitch_errors_a, s=1, c='blue')
ax1.scatter(pitch_values_b, pitch_errors_b, s=1, c='red')
plt.show()

# Plot roll errors
roll_values_a = np.array(a['roll_value'])
roll_errors_a = np.array(a['roll_error'])

roll_values_b = np.array(b['roll_value'])
roll_errors_b = np.array(b['roll_error'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(roll_values_a, roll_errors_a, s=1, c='blue')
ax1.scatter(roll_values_b, roll_errors_b, s=1, c='red')
plt.show()
