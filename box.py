import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time

# Load punch data from JSON
with open('box.json', 'r') as f:
    punch_data = json.load(f)

def clean_text(text):
    """Clean text by replacing curly apostrophes with straight apostrophes"""
    if text:
        return text.replace('\u2019', "'")  # Unicode for right single quotation mark
    return text

# Clean feedback text in all punches
for punch in punch_data['punches']:
    punch['feedback'] = clean_text(punch['feedback'])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the video files
process_video_path = 'punching.mp4'
display_video_path = 'punching.mp4'

# Open processing video (lower res)
process_cap = cv2.VideoCapture(process_video_path)
process_fps = int(process_cap.get(cv2.CAP_PROP_FPS))
process_width = int(process_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
process_height = int(process_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open display video (higher res)
display_cap = cv2.VideoCapture(display_video_path)
display_fps = int(display_cap.get(cv2.CAP_PROP_FPS))
display_width = int(display_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
display_height = int(display_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to store all processed frames
processed_frames = []

# Animation variables
last_shot_time = None
animation_duration = 1.25  # seconds (changed from 0.5 to 1.0)
current_color = (255, 255, 255)  # Start with white

def parse_timestamp(timestamp):
    # Convert timestamp (e.g., "0:07.5") to seconds
    minutes, seconds = timestamp.split(':')
    return float(minutes) * 60 + float(seconds)

def timestamp_to_frame(timestamp, fps):
    # Convert timestamp to frame number with 0.01 second offset back
    seconds = parse_timestamp(timestamp) - 0.01
    return int(max(0, seconds) * fps)  # Ensure we don't go negative

def wrap_text(text, font, scale, thickness, max_width):
    # Calculate the maximum number of characters that can fit in max_width
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_animation_color(elapsed_time, is_made):
    if elapsed_time >= animation_duration:
        return (255, 255, 255)  # Return to white
    
    # Calculate progress through animation (0 to 1)
    progress = elapsed_time / animation_duration
    
    if progress < 0.5:
        # Fade to color
        if is_made:
            # Fade to green (BGR format)
            return (
                int(255 * (1 - progress * 2)),  # B
                255,                            # G
                int(255 * (1 - progress * 2))   # R
            )
        else:
            # Fade to red (BGR format) - keeping some green and blue for brighter red
            return (
                int(255 * (1 - progress * 2)),  # B
                int(255 * (1 - progress * 2)),  # G
                255                             # R
            )
    else:
        # Fade back to white
        if is_made:
            # Fade from green to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                255,                                # G
                int(255 * ((progress - 0.5) * 2))   # R
            )
        else:
            # Fade from red to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                int(255 * ((progress - 0.5) * 2)),  # G
                255                                 # R
            )

# Convert timestamps to frame numbers and add feedback display duration
for punch in punch_data['punches']:
    punch['frame_number'] = timestamp_to_frame(punch['timestamp_of_outcome'], process_fps)
    punch['feedback_end_frame'] = punch['frame_number'] + (4 * process_fps)  # Show feedback for 4 seconds

last_head = None
last_arms = None
previous_arms = None
arm_trail = []
max_trail_length = 10
frame_count = 0
process_every_n_frames = int(process_fps / 20)  # Process at 3 fps
last_punch_result = None

print("Processing video...")

while process_cap.isOpened() and display_cap.isOpened():
    # Read frames from both videos
    process_ret, process_frame = process_cap.read()
    display_ret, display_frame = display_cap.read()
    
    if not process_ret or not display_ret:
        break

    frame_count += 1
    
    # Only process every nth frame
    if frame_count % process_every_n_frames == 0:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get the head landmark (landmark 0 is the top of the head)
            head = results.pose_landmarks.landmark[0]
            # Scale coordinates to display resolution
            head_x = int(head.x * display_width)
            head_y = int(head.y * display_height)
            last_head = (head_x, head_y)
            
            # Get arm landmarks
            landmarks = results.pose_landmarks.landmark
            # Left arm: shoulder (11), elbow (13), wrist (15)
            # Right arm: shoulder (12), elbow (14), wrist (16)
            left_shoulder = (int(landmarks[11].x * display_width), int(landmarks[11].y * display_height))
            left_elbow = (int(landmarks[13].x * display_width), int(landmarks[13].y * display_height))
            left_wrist = (int(landmarks[15].x * display_width), int(landmarks[15].y * display_height))
            
            right_shoulder = (int(landmarks[12].x * display_width), int(landmarks[12].y * display_height))
            right_elbow = (int(landmarks[14].x * display_width), int(landmarks[14].y * display_height))
            right_wrist = (int(landmarks[16].x * display_width), int(landmarks[16].y * display_height))
            
            current_arms = {
                'left_shoulder': left_shoulder,
                'left_elbow': left_elbow,
                'left_wrist': left_wrist,
                'right_shoulder': right_shoulder,
                'right_elbow': right_elbow,
                'right_wrist': right_wrist
            }
            
            # Store for trail visualization (always add current position)
            arm_trail.append({
                'left_wrist': left_wrist,
                'right_wrist': right_wrist,
                'frame': frame_count
            })
            if len(arm_trail) > max_trail_length:
                arm_trail.pop(0)
            
            previous_arms = last_arms
            last_arms = current_arms

    # Draw the arrow and name if we have a head position
    if last_head is not None:
        head_x, head_y = last_head
        arrow_height = 30  # Increased from 20 to 30
        arrow_width = 45   # Increased from 30 to 45
        arrow_tip_y = max(0, head_y - 110)  # Changed from -60 to -110 to move arrow 50px higher
        # Triangle points for the arrow
        pt1 = (head_x, arrow_tip_y + arrow_height)  # tip
        pt2 = (head_x - arrow_width // 2, arrow_tip_y)  # left
        pt3 = (head_x + arrow_width // 2, arrow_tip_y)  # right
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(display_frame, [pts], (0, 0, 255))  # Red arrow
        # Draw the name above the arrow with black border
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "kian"
        text_size = cv2.getTextSize(text, font, 2.5, 6)[0]  # Reduced from 3.0 to 2.5
        text_x = head_x - text_size[0] // 2
        text_y = arrow_tip_y - 10
        # Black border
        cv2.putText(display_frame, text, (text_x, text_y), font, 2.5, (0, 0, 0), 15, cv2.LINE_AA)  # Reduced from 3.0 to 2.5
        # White fill
        cv2.putText(display_frame, text, (text_x, text_y), font, 2.5, (255, 255, 255), 6, cv2.LINE_AA)  # Reduced from 3.0 to 2.5

    # Draw arm skeleton if we have arm positions
    if last_arms is not None:
        arm_color = (0, 255, 255)  # Yellow for arms
        arm_thickness = 4
        
        # Draw left arm
        cv2.line(display_frame, last_arms['left_shoulder'], last_arms['left_elbow'], arm_color, arm_thickness)
        cv2.line(display_frame, last_arms['left_elbow'], last_arms['left_wrist'], arm_color, arm_thickness)
        
        # Draw right arm
        cv2.line(display_frame, last_arms['right_shoulder'], last_arms['right_elbow'], arm_color, arm_thickness)
        cv2.line(display_frame, last_arms['right_elbow'], last_arms['right_wrist'], arm_color, arm_thickness)
        
        # Draw joint circles
        joint_radius = 6
        joint_color = (0, 0, 255)  # Red for joints
        cv2.circle(display_frame, last_arms['left_shoulder'], joint_radius, joint_color, -1)
        cv2.circle(display_frame, last_arms['left_elbow'], joint_radius, joint_color, -1)
        cv2.circle(display_frame, last_arms['left_wrist'], joint_radius, joint_color, -1)
        cv2.circle(display_frame, last_arms['right_shoulder'], joint_radius, joint_color, -1)
        cv2.circle(display_frame, last_arms['right_elbow'], joint_radius, joint_color, -1)
        cv2.circle(display_frame, last_arms['right_wrist'], joint_radius, joint_color, -1)


    # Calculate and display punch power based on arm speed
    if last_arms and previous_arms:
        # Calculate wrist speed for punch power visualization
        left_wrist_speed = np.sqrt((last_arms['left_wrist'][0] - previous_arms['left_wrist'][0])**2 + 
                                  (last_arms['left_wrist'][1] - previous_arms['left_wrist'][1])**2)
        right_wrist_speed = np.sqrt((last_arms['right_wrist'][0] - previous_arms['right_wrist'][0])**2 + 
                                   (last_arms['right_wrist'][1] - previous_arms['right_wrist'][1])**2)
        
        max_speed = max(left_wrist_speed, right_wrist_speed)
        
        # Show power meter when speed is significant
        if max_speed > 10:  # Threshold for showing power
            power_level = min(max_speed / 50.0, 1.0)  # Normalize to 0-1
            
            # Draw power bar in top right
            bar_x = display_width - 150
            bar_y = 150
            bar_width = 100
            bar_height = 20
            
            # Background bar
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Power level bar
            power_width = int(bar_width * power_level)
            power_color = (0, 255, 0) if power_level < 0.7 else (0, 255, 255) if power_level < 0.9 else (0, 0, 255)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + power_width, bar_y + bar_height), power_color, -1)
            
            # Border
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            
            # Label
            cv2.putText(display_frame, "POWER", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw guard position indicator
    if last_arms:
        # Check if hands are in guard position (near head level and protecting face)
        left_guard = last_arms['left_wrist'][1] < last_head[1] + 50 if last_head else False
        right_guard = last_arms['right_wrist'][1] < last_head[1] + 50 if last_head else False
        
        # Check if hands are in front of body (defensive position)
        if last_head:
            left_front = abs(last_arms['left_wrist'][0] - last_head[0]) < 150
            right_front = abs(last_arms['right_wrist'][0] - last_head[0]) < 150
        else:
            left_front = right_front = False
        
        guard_up = (left_guard and left_front) or (right_guard and right_front)
        
        # Draw guard status indicator
        guard_x = 30
        guard_y = 220
        guard_text = "GUARD UP" if guard_up else "GUARD DOWN"
        guard_color = (0, 255, 0) if guard_up else (0, 0, 255)  # Green if up, red if down
        
        # Background for guard indicator
        text_size = cv2.getTextSize(guard_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(display_frame, (guard_x - 5, guard_y - 25), (guard_x + text_size[0] + 5, guard_y + 5), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (guard_x - 5, guard_y - 25), (guard_x + text_size[0] + 5, guard_y + 5), guard_color, 2)
        
        # Guard text
        cv2.putText(display_frame, guard_text, (guard_x, guard_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, guard_color, 2)

    # Punch form analysis overlay
    if last_arms and previous_arms:
        # Calculate arm extension for form analysis
        left_arm_length = np.sqrt((last_arms['left_wrist'][0] - last_arms['left_elbow'][0])**2 + 
                                 (last_arms['left_wrist'][1] - last_arms['left_elbow'][1])**2)
        right_arm_length = np.sqrt((last_arms['right_wrist'][0] - last_arms['right_elbow'][0])**2 + 
                                  (last_arms['right_wrist'][1] - last_arms['right_elbow'][1])**2)
        
        # Calculate shoulder width for reference
        shoulder_width = np.sqrt((last_arms['left_shoulder'][0] - last_arms['right_shoulder'][0])**2 + 
                                (last_arms['left_shoulder'][1] - last_arms['right_shoulder'][1])**2)
        
        # Check if punching (arm extended relative to shoulder width)
        left_extended = left_arm_length > shoulder_width * 0.8
        right_extended = right_arm_length > shoulder_width * 0.8
        
        if left_extended or right_extended:
            # Draw punch form indicators
            form_y = 180
            
            # Check punch angle (should be straight, not downward)
            if left_extended:
                left_angle = np.arctan2(last_arms['left_wrist'][1] - last_arms['left_elbow'][1], 
                                      last_arms['left_wrist'][0] - last_arms['left_elbow'][0])
                left_good_form = abs(left_angle) < 0.5  # Within ~30 degrees of horizontal
                
                form_text = "LEFT: GOOD FORM" if left_good_form else "LEFT: IMPROVE ANGLE"
                form_color = (0, 255, 0) if left_good_form else (0, 165, 255)  # Green or orange
                cv2.putText(display_frame, form_text, (display_width - 300, form_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
            
            if right_extended:
                right_angle = np.arctan2(last_arms['right_wrist'][1] - last_arms['right_elbow'][1], 
                                       last_arms['right_wrist'][0] - last_arms['right_elbow'][0])
                right_good_form = abs(right_angle) < 0.5
                
                form_text = "RIGHT: GOOD FORM" if right_good_form else "RIGHT: IMPROVE ANGLE"
                form_color = (0, 255, 0) if right_good_form else (0, 165, 255)
                cv2.putText(display_frame, form_text, (display_width - 300, form_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)

    # Calculate current punch statistics
    current_good_punches = 0
    current_bad_punches = 0
    current_feedback = None
    last_punch_display = None
    
    for punch in punch_data['punches']:
        if punch['frame_number'] <= frame_count:
            if punch['frame_number'] == frame_count:
                # New punch detected
                last_shot_time = time.time()
                last_punch_result = punch.get('punch_quality', punch['result'])
            # Use the punch_quality field to determine good/bad punches
            if punch.get('punch_quality') == 'good':
                current_good_punches = punch.get('total_good_punches', 0)
            current_bad_punches = punch.get('total_bad_punches', 0)
            # Update last punch display info
            last_punch_display = f"{punch['bag_zone']} {punch['punch_type']}"
            # Check if we should show feedback for this punch
            if punch['frame_number'] <= frame_count <= punch['feedback_end_frame']:
                if punch['feedback'] and punch['feedback'].strip():
                    current_feedback = punch['feedback']

    # Display shot statistics in top left
    stats_font = cv2.FONT_HERSHEY_SIMPLEX
    stats_border = (0, 0, 0)  # Black border
    stats_scale = 1.8  # Made slightly smaller
    stats_thickness = 5  # Adjusted thickness
    stats_border_thickness = 10  # Adjusted border thickness
    stats_spacing = 75  # Adjusted spacing
    white_color = (255, 255, 255)

    # Position for stats (top left with padding)
    stats_x = 30
    stats_y = 80  # Moved higher up

    # Calculate animation color if needed
    if last_shot_time is not None:
        elapsed_time = time.time() - last_shot_time
        if elapsed_time < animation_duration:
            current_color = get_animation_color(elapsed_time, last_punch_result == 'good')
        else:
            current_color = white_color
            last_shot_time = None

    # Draw good punches
    good_text = f"Good Punches: {current_good_punches}"
    
    # Draw good punches with border
    cv2.putText(display_frame, good_text, 
                (stats_x, stats_y), stats_font, stats_scale, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    # Draw good punches with color
    cv2.putText(display_frame, good_text, 
                (stats_x, stats_y), stats_font, stats_scale, 
                current_color if last_punch_result == 'good' else white_color, stats_thickness, cv2.LINE_AA)
    
    # Draw bad punches
    bad_text = f"Bad Punches: {current_bad_punches}"
    
    # Draw bad punches with border
    cv2.putText(display_frame, bad_text, 
                (stats_x, stats_y + stats_spacing), stats_font, stats_scale, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    # Draw bad punches with color
    cv2.putText(display_frame, bad_text, 
                (stats_x, stats_y + stats_spacing), stats_font, stats_scale, 
                current_color if last_punch_result == 'bad' else white_color, stats_thickness, cv2.LINE_AA)

    # Display last punch in top right corner
    if last_punch_display:
        last_punch_font = cv2.FONT_HERSHEY_DUPLEX
        last_punch_scale = 1.4
        last_punch_thickness = 4
        last_punch_border_thickness = 8
        last_punch_color = (255, 255, 255)
        last_punch_border = (0, 0, 0)
        
        # Calculate position (top right with padding)
        last_punch_text = last_punch_display.upper()
        text_size = cv2.getTextSize(last_punch_text, last_punch_font, last_punch_scale, last_punch_thickness)[0]
        last_punch_x = display_width - text_size[0] - 30
        last_punch_y = 70
        
        # Draw background panel for last punch
        panel_padding = 15
        punch_panel_overlay = display_frame.copy()
        cv2.rectangle(punch_panel_overlay,
                     (last_punch_x - panel_padding, last_punch_y - 35),
                     (last_punch_x + text_size[0] + panel_padding, last_punch_y + 15),
                     (25, 25, 25), -1)
        cv2.addWeighted(punch_panel_overlay, 0.75, display_frame, 0.25, 0, display_frame)
        
        # Add border
        cv2.rectangle(display_frame,
                     (last_punch_x - panel_padding, last_punch_y - 35),
                     (last_punch_x + text_size[0] + panel_padding, last_punch_y + 15),
                     (100, 100, 100), 2)
        
        # Draw last punch text
        cv2.putText(display_frame, last_punch_text,
                   (last_punch_x, last_punch_y), last_punch_font, last_punch_scale,
                   last_punch_border, last_punch_border_thickness, cv2.LINE_AA)
        cv2.putText(display_frame, last_punch_text,
                   (last_punch_x, last_punch_y), last_punch_font, last_punch_scale,
                   last_punch_color, last_punch_thickness, cv2.LINE_AA)

    # Display feedback if available and not empty
    if current_feedback and current_feedback.strip():
        feedback_font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 1.3  # Optimized size for readability
        feedback_color = (255, 255, 255)  # White text
        feedback_border = (0, 0, 0)  # Black border
        feedback_thickness = 3  # Clean thickness
        feedback_border_thickness = 7  # Strong border for visibility
        feedback_spacing = 45  # Tight line spacing

        # Wrap text to fit within 85% of screen width for better use of space
        max_width = int(display_width * 0.85)
        wrapped_lines = wrap_text(current_feedback, feedback_font, feedback_scale, feedback_thickness, max_width)

        # Calculate total height of wrapped text
        total_height = len(wrapped_lines) * feedback_spacing
        start_y = display_height - 40 - total_height  # Position at bottom with padding

        # Draw each line centered
        for i, line in enumerate(wrapped_lines):
            text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
            feedback_x = (display_width - text_size[0]) // 2
            feedback_y = start_y + (i * feedback_spacing)

            # Draw semi-transparent background for better readability
            text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
            bg_x1 = feedback_x - 10
            bg_y1 = feedback_y - text_size[1] - 5
            bg_x2 = feedback_x + text_size[0] + 10
            bg_y2 = feedback_y + 8
            
            # Create overlay for semi-transparent background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
            
            # Draw text with border
            cv2.putText(display_frame, line, 
                        (feedback_x, feedback_y), feedback_font, feedback_scale, 
                        feedback_border, feedback_border_thickness, cv2.LINE_AA)
            cv2.putText(display_frame, line, 
                        (feedback_x, feedback_y), feedback_font, feedback_scale, 
                        feedback_color, feedback_thickness, cv2.LINE_AA)

    # Store the processed frame
    processed_frames.append(display_frame.copy())

    # Display the frame (optional)
    cv2.imshow('Boxing Training Analysis', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
process_cap.release()
display_cap.release()
cv2.destroyAllWindows()

print("Creating final video...")

# Create the final video at normal speed
final_output_path = 'final.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
final_out = cv2.VideoWriter(final_output_path, fourcc, display_fps, (display_width, display_height))

# Write all frames to the final video
for frame in processed_frames:
    final_out.write(frame)

final_out.release()

print(f"Processing complete. Final video saved to {final_output_path}")

