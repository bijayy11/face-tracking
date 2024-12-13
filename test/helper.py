def calculate_steps_for_motor(bbox_center_x, frame_center_x, iw, max_steps=5389):
    # Calculate the offset between the bounding box center and frame center
    offset = bbox_center_x - frame_center_x
   
    # Calculate the number of steps to move
    steps_to_move = int((offset / iw) * max_steps)
   
    return steps_to_move


def move_motor(steps):
    global current_step
   
    # Calculate the new position after moving
    new_current_step = current_step + steps
   
    # Check if the new position exceeds limits and adjust if necessary
    if new_current_step < left_step_limit:
        # If the new position is less than the left limit, move to the default position
        steps = default_step - current_step
        current_step = default_step
        return steps
    elif new_current_step > right_step_limit:
        # If the new position exceeds the right limit, move to the default position
        steps = default_step - current_step
        current_step = default_step
        return steps
    else:
        # If within limits, update current_step normally
        current_step = new_current_step
        return steps
