import fetch_api

base_link = 'base_link'

joint_names = [ 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']

num = 7

home_joints = [1.3205522228271485, 1.399532370159912, -0.19974325208511354, 1.719844644293213, 0.0004958728740930562, 1.6597035481201172, 0]
hide_left_joints = [1.4003189731628418, -0.004444070269775391, -1.610238473699951, 1.5296304871887207, 1.5398456281463624, 1.4863637043701172, -0.03673418978117943]
hide_right_joints = [-1.5978461574523926, -0.0063614324829101565, -1.6117724588378906, -1.574379617944336, 1.5390786355773927, 1.485213096307373, -0.03711768234037399]
hold_left_joints = [1.4961928058654785, 0.9002209230163574, -0.035223617987298964, -2.019233877435303, -0.08042161404685974, -0.40695208771362307, -0.04632157081626892]
hold_right_joints = [-1.5978461574523926, 0.8338766618469238, -0.0018597220308721064, -2.105903799310303, 3.11639491975708, 0.26339751021728514, -0.03673418978117943]

HOME_JOINTS = [(joint_names[i], home_joints[i]) for i in range(num)]
HIDE_LEFT_JOINTS = [(joint_names[i], hide_left_joints[i]) for i in range(num)]
HIDE_RIGHT_JOINTS = [(joint_names[i], hide_right_joints[i]) for i in range(num)]
HOLD_LEFT_JOINTS = [(joint_names[i], hold_left_joints[i]) for i in range(num)]
HOLD_RIGHT_JOINTS = [(joint_names[i], hold_right_joints[i]) for i in range(num)]

