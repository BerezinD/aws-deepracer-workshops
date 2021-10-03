def reward_function(params):
    # these parameters are for:
    # The official track for the 2019 AWS DeepRacer Championship Cup finals,
    # this is a moderately challenging track ideal for stepping up your training and experimentation.
    # Length: 23.12 m (75.85') Width: 107 cm (42")
    # some of the tips was paste from the internet

    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    center_variance = params["distance_from_center"] / params["track_width"]
    all_wheels_on_track = params['all_wheels_on_track']
    closest_waypoints = params["closest_waypoints"]
    speed = params['speed']
    abs_steering = abs(params['steering_angle'])
    MAX_SPEED = 4

    # weird markers
    marker_1 = 0.12 * track_width
    marker_2 = 0.24 * track_width
    marker_3 = 0.36 * track_width
    marker_4 = 0.48 * track_width

    left_lane = [7, 8, 9, 13, 14, 15, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 83, 84, 88, 89, 90,
                 91, 135, 136, 137, 138, 139, 140, 141, 142, 147, 148]
    right_lane = [0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 28, 29, 30, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 117, 118,
                  122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 151, 152, 153]
    center_lane = [5, 6, 16, 17, 31, 32, 33, 46, 47, 81, 82, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                   105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 133, 134, 149, 150]
    straight_lane = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    left_end_lane = [10, 11, 12, 85, 86, 87, 143, 144, 145, 146]
    right_end_lane = [23, 24, 25, 26, 27, 119, 120, 121]

    # initial reward (to support a couple mistakes)
    reward = 2.1

    if all_wheels_on_track:
        reward += 1
    else:
        reward -= 1

    # Give higher reward if the car is in a hardcoded lap line
    if closest_waypoints[1] in left_lane and params['is_left_of_center']:
        reward += 1
    elif closest_waypoints[1] in right_lane and not params['is_left_of_center']:
        reward += 1
    elif closest_waypoints[1] in center_lane and center_variance < 0.4:
        reward += 1
    else:
        reward -= 1

    #     speed_rate = speed / MAX_SPEED
    #     reward_speed = speed_rate **2

    if closest_waypoints[1] in straight_lane and abs_steering < 5:
        speed = MAX_SPEED

    # Give higher reward if the car is closer to center line and vice versa
    reward_distance = 1

    if distance_from_center <= marker_1:
        reward_distance = 1.0
    elif distance_from_center <= marker_2:
        reward_distance = 0.7
    elif distance_from_center <= marker_3:
        reward_distance = 0.4
    elif distance_from_center <= marker_4:
        reward_distance = 0.1
    else:
        reward_distance = 1e-3  # likely crashed/ close to off track

    reward_all = reward + reward_distance

    ABS_STEERING_THRESHOLD = 15

    # Penalize reward if the car is steering too much
    if closest_waypoints[1] in straight_lane and abs_steering > 5:
        reward_all *= 0.8
    elif closest_waypoints[1] in left_end_lane and abs_steering > 10 and params['is_left_of_center']:
        reward_all *= 0.8
    elif closest_waypoints[1] in right_end_lane and abs_steering > 10 and not params['is_left_of_center']:
        reward_all *= 0.8
    elif closest_waypoints[1] in center_lane and abs_steering > ABS_STEERING_THRESHOLD:
        reward_all *= 0.8
    else:
        reward_all = reward_all

    return float(reward_all)
