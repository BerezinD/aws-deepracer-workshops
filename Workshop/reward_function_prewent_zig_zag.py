def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    SPEED_THRESHOLD = 1.0
    heading = params['heading']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    steering_angle = params['steering_angle']
    # 4 markers to reward carefully
    marker_1 = 0.1 * track_width
    marker_2 = 0.2 * track_width
    marker_3 = 0.3 * track_width
    marker_4 = 0.4 * track_width

    # init reward
    reward = 1e-3

    # reward_lane to keep tracking points for staying on track
    if all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.05:
        reward_lane = 1.0
    else:
        reward_lane = 1e-3
    # reward for prevent zig-zag
    if distance_from_center <= marker_2:
        reward_lane *= 1.0
        ABS_STEERING_THRESHOLD = 20
        SPEED_THRESHOLD = 2.0  # is that correct value?
    elif distance_from_center <= marker_3:
        reward_lane *= 0.9
        ABS_STEERING_THRESHOLD = 40
        SPEED_THRESHOLD = 1.0
    elif distance_from_center <= marker_4:
        reward_lane *= 0.8
        ABS_STEERING_THRESHOLD = 60
        SPEED_THRESHOLD = 0.5
    else:
        reward_lane *= 0.7
        ABS_STEERING_THRESHOLD = 90
        SPEED_THRESHOLD = 0.1

    # From the internet:
    reward_heading = 1.0

    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # Penalize the reward if the difference is too large
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        reward_heading *= 0.5
        SPEED_THRESHOLD = 90

    # Penalize reward if the car is steering too much
    if abs(steering_angle) > ABS_STEERING_THRESHOLD:
        reward_heading *= 0.5

    reward_speed = 1.0

    if speed < SPEED_THRESHOLD:  # Penalize if the car goes too slow
        reward_speed *= 0.5
    else:  # High reward if the car stays on track and goes fast
        reward_speed *= 1.0

    reward += 1.0 * reward_lane + 1.0 * reward_heading + 0.5 * reward_speed

    # give a reward at each time of the completing a lap
    if params['progress'] == 100:
        reward += 10000

    return float(reward)
