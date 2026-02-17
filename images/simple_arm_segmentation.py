"""
Simple Arm Segmentation - MediaPipe Vision API
Segments arm area from shoulder to wrist using detected pose landmarks
Uses MediaPipe Selfie Segmentation for accurate edge detection
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
POSE_MODEL_PATH = 'images/pose_landmarker_heavy.task'
SEGMENTATION_MODEL_PATH = 'images/selfie_multiclass_256x256.tflite'
IMAGE_PATH = 'images/real.png'

# Initialize MediaPipe Pose Detector
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

# Initialize MediaPipe Image Segmenter (Selfie Segmentation)
segmenter_base_options = python.BaseOptions(model_asset_path=SEGMENTATION_MODEL_PATH)
segmenter_options = vision.ImageSegmenterOptions(
    base_options=segmenter_base_options,
    output_category_mask=True
)
segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)

# Load and process image
image = cv2.imread(IMAGE_PATH)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to MediaPipe Image format
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# Detect pose
detection_result = detector.detect(mp_image)

if detection_result.pose_landmarks:
    h, w, _ = image.shape
    landmarks = detection_result.pose_landmarks[0]
    
    # Get left arm landmarks (11=shoulder, 13=elbow, 15=wrist)
    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]
    
    # Convert to pixel coordinates
    shoulder_coords = (int(shoulder.x * w), int(shoulder.y * h))
    elbow_coords = (int(elbow.x * w), int(elbow.y * h))
    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
    
    print(f"Shoulder: {shoulder_coords}")
    print(f"Elbow: {elbow_coords}")
    print(f"Wrist: {wrist_coords}")
    
    # ========================================================================
    # MEDIAPIPE SEGMENTATION + BOUNDARY CONSTRAINTS
    # Using MediaPipe Selfie Segmentation for accurate edge detection
    # ========================================================================
    
    print("\n[OK] Using MediaPipe Selfie Segmentation for accurate arm edges")
    
    # Define arm boundaries with small outer margin
    # Simple approach: shoulder to wrist with small buffer
    margin_pixels = 8  # Small outer margin in pixels
    x_min_boundary = min(shoulder_coords[0], wrist_coords[0]) - margin_pixels
    x_max_boundary = max(shoulder_coords[0], wrist_coords[0]) + margin_pixels
    
    # Ensure boundaries stay within image bounds
    x_min_boundary = max(0, x_min_boundary)
    x_max_boundary = min(w - 1, x_max_boundary)
    
    print(f"[OK] Arm boundaries with {margin_pixels}px margin: x from {x_min_boundary} to {x_max_boundary}")
    
    # Step 1: Use MediaPipe Segmentation to get person mask
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask
    
    # Convert to numpy array and create binary mask
    # Category 0 = background, Category 1 = person
    person_mask = (category_mask.numpy_view() > 0).astype(np.uint8) * 255
    
    # Step 2: Create ROI mask based on arm landmarks
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate arm dimensions
    upper_arm_length = np.sqrt((elbow_coords[0] - shoulder_coords[0])**2 + 
                                (elbow_coords[1] - shoulder_coords[1])**2)
    forearm_length = np.sqrt((wrist_coords[0] - elbow_coords[0])**2 + 
                             (wrist_coords[1] - elbow_coords[1])**2)
    
    # Generous thickness for ROI (from backup - proven parameters)
    upper_arm_thickness = int(upper_arm_length * 0.55)
    forearm_thickness = int(forearm_length * 0.50)
    
    # Draw ROI along arm path
    cv2.line(mask_roi, shoulder_coords, elbow_coords, 255, upper_arm_thickness)
    cv2.line(mask_roi, elbow_coords, wrist_coords, 255, forearm_thickness)
    cv2.circle(mask_roi, shoulder_coords, upper_arm_thickness // 2, 255, -1)
    cv2.circle(mask_roi, elbow_coords, max(upper_arm_thickness, forearm_thickness) // 2, 255, -1)
    cv2.circle(mask_roi, wrist_coords, forearm_thickness // 2, 255, -1)
    
    # Expand ROI (from backup - proven parameters)
    kernel_expand = np.ones((25, 25), np.uint8)
    mask_roi = cv2.dilate(mask_roi, kernel_expand, iterations=2)
    
    # Step 3: Combine MediaPipe segmentation with ROI
    # This gives us accurate edges from MediaPipe within the arm region
    mask = cv2.bitwise_and(person_mask, mask_roi)
    
    # Step 4: Morphological operations for smooth edges
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((7, 7), np.uint8)
    kernel_large = np.ones((15, 15), np.uint8)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Fill larger holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    
    # Step 5: Find and keep only the largest contour (the arm)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Smooth the contour using approximation
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Create clean mask with smoothed contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [smoothed_contour], -1, 255, -1)
        
        # Apply Gaussian blur for very smooth edges
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Step 6: Apply boundary constraints - crop mask to shoulder-wrist range
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_mask[:, x_min_boundary:x_max_boundary+1] = 255
    
    # Apply boundary constraint
    mask = cv2.bitwise_and(mask, boundary_mask)
    
    # Step 7: Final smoothing for natural edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Ensure landmarks are inside mask
    cv2.circle(mask, shoulder_coords, 8, 255, -1)
    cv2.circle(mask, elbow_coords, 8, 255, -1)
    cv2.circle(mask, wrist_coords, 8, 255, -1)
    
    print(f"[OK] Mask created with MediaPipe segmentation + boundary constraints")
    print(f"  Mask coverage: {np.count_nonzero(mask)} pixels")
    print(f"  Constrained to x-range: {x_min_boundary} to {x_max_boundary}")
    
    # Save the mask for debugging
    cv2.imwrite('debug_mask.jpg', mask)
    print(f"[OK] Debug mask saved: debug_mask.jpg")
    
    
    # ========================================================================
    # DEBUG VISUALIZATION FUNCTION - Pixel Level Analysis
    # ========================================================================
    
    def create_debug_visualization(landmark_name, point, direction_vector, mask, crop_size=150):
        """
        Create detailed pixel-level visualization of perpendicular intersection finding
        """
        # Calculate perpendicular vector
        dx, dy = direction_vector
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return None
        
        perp_x = -dy / length
        perp_y = dx / length
        arm_x = dx / length
        arm_y = dy / length
        
        # Create zoomed-in view around the landmark
        x_min_crop = max(0, point[0] - crop_size)
        x_max_crop = min(w, point[0] + crop_size)
        y_min_crop = max(0, point[1] - crop_size)
        y_max_crop = min(h, point[1] + crop_size)
        
        # Crop the image and mask
        cropped_img = image[y_min_crop:y_max_crop, x_min_crop:x_max_crop].copy()
        cropped_mask = mask[y_min_crop:y_max_crop, x_min_crop:x_max_crop].copy()
        
        # Convert to RGB for visualization
        debug_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        
        # Draw mask overlay
        mask_overlay = np.zeros_like(debug_img)
        mask_overlay[:, :, 1] = cropped_mask
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # Draw contour
        contours_crop, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_crop:
            cv2.drawContours(debug_img, contours_crop, -1, (0, 0, 0), 2)
        
        # Adjust point coordinates to cropped space
        point_crop = (point[0] - x_min_crop, point[1] - y_min_crop)
        
        # Draw perpendicular line
        line_length = crop_size
        p1 = (int(point_crop[0] - perp_x * line_length), int(point_crop[1] - perp_y * line_length))
        p2 = (int(point_crop[0] + perp_x * line_length), int(point_crop[1] + perp_y * line_length))
        cv2.line(debug_img, p1, p2, (255, 200, 200), 2)
        
        # Draw arm axis line
        a1 = (int(point_crop[0] - arm_x * line_length), int(point_crop[1] - arm_y * line_length))
        a2 = (int(point_crop[0] + arm_x * line_length), int(point_crop[1] + arm_y * line_length))
        cv2.line(debug_img, a1, a2, (255, 255, 0), 1)
        
        # Mark the landmark point
        cv2.circle(debug_img, point_crop, 5, (255, 0, 0), -1)
        cv2.circle(debug_img, point_crop, 6, (0, 0, 0), 1)
        
        # Analyze contour points
        if contours_crop:
            main_contour_crop = max(contours_crop, key=cv2.contourArea)
            contour_points_crop = main_contour_crop.reshape(-1, 2)
            
            # Find candidate points
            for threshold in [15, 25, 40]:
                candidates = []
                for cp in contour_points_crop:
                    # Convert back to original space for calculation
                    cp_orig = (cp[0] + x_min_crop, cp[1] + y_min_crop)
                    vec = (cp_orig[0] - point[0], cp_orig[1] - point[1])
                    arm_proj = abs(vec[0] * arm_x + vec[1] * arm_y)
                    
                    if arm_proj < threshold:
                        perp_proj = vec[0] * perp_x + vec[1] * perp_y
                        side = 1 if perp_proj > 0 else -1
                        candidates.append({
                            'point_crop': tuple(cp),
                            'arm_proj': arm_proj,
                            'side': side
                        })
                
                # Draw candidates
                color = (255, 0, 255) if threshold == 15 else (200, 100, 200) if threshold == 25 else (150, 50, 150)
                for cand in candidates:
                    cv2.circle(debug_img, cand['point_crop'], 2, color, -1)
                
                if len([c for c in candidates if c['side'] == 1]) > 0 and len([c for c in candidates if c['side'] == -1]) > 0:
                    break
        
        return debug_img

    
    # ========================================================================
    # PERPENDICULAR LINES AT LANDMARKS WITH CONTOUR INTERSECTIONS
    # Improved method using direct contour intersection
    # ========================================================================
    
    def find_perpendicular_intersections_v4(point, direction_vector, mask, max_length=250, is_endpoint=False):
        """
        Find intersection points on contour edges along perpendicular line
        Robust method that handles both edge-crossing and interior-passing cases
        
        For endpoints (shoulder/wrist): Use symmetric distance approach
        - Find top intersection normally
        - For bottom, place point at same distance along contour (not perpendicular search)
        
        Args:
            point: (x, y) landmark point
            direction_vector: (dx, dy) direction along the arm
            mask: binary mask of the arm
            max_length: maximum length to search for intersections
            is_endpoint: True for shoulder/wrist, False for elbow
            
        Returns:
            (top_point, bottom_point) intersection coordinates
        """
        # Calculate perpendicular vector (rotate 90 degrees)
        dx, dy = direction_vector
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return None, None
        
        # Normalized perpendicular vector
        perp_x = -dy / length
        perp_y = dx / length
        
        # Get contour points for fallback method
        contours_all, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours_all:
            return None, None
        
        main_contour = max(contours_all, key=cv2.contourArea)
        contour_points = main_contour.reshape(-1, 2)
        
        def find_edge_along_direction(start_point, direction_x, direction_y, max_dist, side_sign):
            """
            Search along a direction for mask edge
            Returns edge point or nearest contour point in that direction
            """
            edge_point = None
            last_inside_point = None
            
            # Method 1: Search along perpendicular line for edge transition
            for dist in range(1, max_dist):
                test_x = int(start_point[0] + direction_x * dist)
                test_y = int(start_point[1] + direction_y * dist)
                
                # Check bounds
                if not (0 <= test_x < w and 0 <= test_y < h):
                    break
                
                current_value = mask[test_y, test_x]
                
                # Track last inside point
                if current_value == 255:
                    last_inside_point = (test_x, test_y)
                
                # Look for transition from inside (255) to outside (0)
                if last_inside_point and current_value == 0:
                    # Found edge transition! Use the last inside point
                    edge_point = last_inside_point
                    break
            
            # Method 2: If no edge found (line passes through interior), 
            # find nearest contour point in this perpendicular direction
            if edge_point is None:
                # Filter contour points that are on the correct side of the perpendicular
                candidates = []
                for cp in contour_points:
                    # Vector from landmark to contour point
                    vec = (cp[0] - start_point[0], cp[1] - start_point[1])
                    
                    # Project onto perpendicular direction
                    perp_proj = vec[0] * direction_x + vec[1] * direction_y
                    
                    # Only consider points in the correct direction (positive projection)
                    if perp_proj > 5:  # At least 5 pixels away
                        # Also check alignment with perpendicular (not too far off-axis)
                        # Project onto arm axis to measure off-axis distance
                        arm_x = dx / length
                        arm_y = dy / length
                        arm_proj = abs(vec[0] * arm_x + vec[1] * arm_y)
                        
                        # Accept points within reasonable off-axis distance
                        if arm_proj < 60:  # Increased from 50 - more relaxed to find edges
                            distance = np.sqrt(vec[0]**2 + vec[1]**2)
                            candidates.append({
                                'point': tuple(cp),
                                'distance': distance,
                                'arm_proj': arm_proj,
                                'perp_proj': perp_proj
                            })
                
                # Sort by perpendicular projection (farthest point in that direction)
                # This ensures we get the actual edge, not an interior point
                if candidates:
                    candidates.sort(key=lambda c: -c['perp_proj'])
                    edge_point = candidates[0]['point']
            
            return edge_point
        
        # Search in positive perpendicular direction (top/side 1)
        top_point = find_edge_along_direction(point, perp_x, perp_y, max_length, 1)
        
        # For endpoints (shoulder/wrist), use symmetric distance approach for bottom
        if is_endpoint and top_point:
            # Calculate distance from landmark to top point
            top_distance = np.sqrt((top_point[0] - point[0])**2 + (top_point[1] - point[1])**2)
            
            # Find contour point at approximately same distance in opposite direction
            # Search along contour for point at similar distance below landmark
            candidates_bottom = []
            arm_x = dx / length
            arm_y = dy / length
            
            for cp in contour_points:
                vec = (cp[0] - point[0], cp[1] - point[1])
                
                # Check if point is in the opposite perpendicular direction (negative)
                perp_proj = vec[0] * (-perp_x) + vec[1] * (-perp_y)
                
                if perp_proj > 5:  # In the bottom direction
                    distance = np.sqrt(vec[0]**2 + vec[1]**2)
                    
                    # Check alignment - should be close to perpendicular line
                    arm_proj = abs(vec[0] * arm_x + vec[1] * arm_y)
                    
                    # Look for points at similar distance as top point AND well-aligned
                    distance_diff = abs(distance - top_distance)
                    
                    if distance_diff < top_distance * 0.25 and arm_proj < 40:  # Stricter: 25% tolerance and better alignment
                        candidates_bottom.append({
                            'point': tuple(cp),
                            'distance': distance,
                            'distance_diff': distance_diff,
                            'arm_proj': arm_proj
                        })
            
            # Pick the point with best combination of distance match and alignment
            if candidates_bottom:
                # Sort by distance difference first, then by alignment
                candidates_bottom.sort(key=lambda c: (c['distance_diff'], c['arm_proj']))
                bottom_point = candidates_bottom[0]['point']
            else:
                # Fallback to normal search
                bottom_point = find_edge_along_direction(point, -perp_x, -perp_y, max_length, -1)
        else:
            # For elbow or if top not found, use normal perpendicular search
            bottom_point = find_edge_along_direction(point, -perp_x, -perp_y, max_length, -1)
        
        return top_point, bottom_point
    
    # ========================================================================
    # FUNCTION: ADD SEMI-LANDMARKS (SLIDING LANDMARKS)
    # ========================================================================
    
    def add_semi_landmarks(start_point, end_point, direction_vector, mask, num_divisions=3):
        """
        Add semi-landmarks (sliding landmarks) between two anatomical landmarks
        
        Semi-landmarks are intermediate points placed at equal intervals along the arm,
        with perpendicular lines to capture arm width at each position.
        
        Args:
            start_point: (x, y) starting anatomical landmark (e.g., shoulder)
            end_point: (x, y) ending anatomical landmark (e.g., elbow)
            direction_vector: (dx, dy) direction along the arm
            mask: binary mask of the arm
            num_divisions: number of semi-landmarks to add (default 3)
            
        Returns:
            List of dictionaries with 'center', 'top', 'bottom' for each semi-landmark
        """
        semi_landmarks = []
        
        # Create semi-landmarks along the line from start to end
        for i in range(1, num_divisions + 1):
            # Calculate position along the line (equally spaced)
            ratio = i / (num_divisions + 1)
            semi_x = int(start_point[0] + (end_point[0] - start_point[0]) * ratio)
            semi_y = int(start_point[1] + (end_point[1] - start_point[1]) * ratio)
            semi_point = (semi_x, semi_y)
            
            # Find perpendicular intersections at this semi-landmark
            top_point, bottom_point = find_perpendicular_intersections_v4(
                semi_point, direction_vector, mask, max_length=250, is_endpoint=False
            )
            
            semi_landmarks.append({
                'center': semi_point,
                'top': top_point,
                'bottom': bottom_point
            })
        
        return semi_landmarks
    
    print("\n[OK] Finding perpendicular intersections at landmarks and intermediate points...")
    
    # Calculate direction vectors for each landmark
    # Shoulder: direction from shoulder to elbow
    shoulder_direction = (elbow_coords[0] - shoulder_coords[0], 
                         elbow_coords[1] - shoulder_coords[1])
    
    # Elbow: average direction (from shoulder to wrist)
    elbow_direction = (wrist_coords[0] - shoulder_coords[0],
                      wrist_coords[1] - shoulder_coords[1])
    
    # Wrist: direction from elbow to wrist
    wrist_direction = (wrist_coords[0] - elbow_coords[0],
                      wrist_coords[1] - elbow_coords[1])
    
    # ========================================================================
    # CREATE INTERMEDIATE POINTS FOR PERPENDICULAR LINES
    # Divide upper arm (shoulder to elbow) into 4 segments = 3 intermediate perpendiculars
    # Divide forearm (elbow to wrist) into 4 segments = 3 intermediate perpendiculars
    # ========================================================================
    
    num_divisions = 4  # Creates 3 intermediate points between landmarks
    
    # Upper arm intermediate points
    upper_arm_points = []
    for i in range(1, num_divisions):
        ratio = i / num_divisions
        x = int(shoulder_coords[0] + (elbow_coords[0] - shoulder_coords[0]) * ratio)
        y = int(shoulder_coords[1] + (elbow_coords[1] - shoulder_coords[1]) * ratio)
        upper_arm_points.append((x, y))
    
    # Forearm intermediate points
    forearm_points = []
    for i in range(1, num_divisions):
        ratio = i / num_divisions
        x = int(elbow_coords[0] + (wrist_coords[0] - elbow_coords[0]) * ratio)
        y = int(elbow_coords[1] + (wrist_coords[1] - elbow_coords[1]) * ratio)
        forearm_points.append((x, y))
    
    print(f"  Created {len(upper_arm_points)} intermediate points on upper arm")
    print(f"  Created {len(forearm_points)} intermediate points on forearm")
    
    # Find intersections for each landmark
    # Shoulder: Find point directly above (most vertically aligned with shoulder)
    # Search for contour point that's above shoulder and closest to vertical alignment
    contours_for_shoulder, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_for_shoulder:
        main_contour_shoulder = max(contours_for_shoulder, key=cv2.contourArea)
        contour_points_shoulder = main_contour_shoulder.reshape(-1, 2)
        
        # Find contour points above shoulder
        candidates_above = []
        for cp in contour_points_shoulder:
            if cp[1] < shoulder_coords[1]:  # Above shoulder (smaller y)
                x_dist = abs(cp[0] - shoulder_coords[0])  # Horizontal distance
                y_dist = shoulder_coords[1] - cp[1]  # Vertical distance
                
                # Only consider points that are reasonably close horizontally
                if x_dist < 50 and y_dist > 10:  # Within 50px horizontally, at least 10px above
                    candidates_above.append({
                        'point': tuple(cp),
                        'x_dist': x_dist,
                        'y_dist': y_dist
                    })
        
        # Sort by horizontal distance (most vertically aligned first)
        if candidates_above:
            candidates_above.sort(key=lambda c: c['x_dist'])
            shoulder_top = candidates_above[0]['point']
        else:
            shoulder_top = None
    else:
        shoulder_top = None
    
    shoulder_bottom = None  # No bottom perpendicular for shoulder
    print(f"DEBUG: shoulder_top = {shoulder_top}, shoulder_bottom = {shoulder_bottom}")
    
    # Elbow: Both top and bottom perpendicular points
    elbow_top, elbow_bottom = find_perpendicular_intersections_v4(
        elbow_coords, elbow_direction, mask, max_length=250, is_endpoint=False
    )
    
    # Wrist: Both top and bottom perpendicular points
    wrist_top, wrist_bottom = find_perpendicular_intersections_v4(
        wrist_coords, wrist_direction, mask, max_length=250, is_endpoint=True
    )
    
    # Store intersection data
    perpendicular_data = {
        'shoulder': {
            'center': shoulder_coords,
            'top': shoulder_top,
            'bottom': shoulder_bottom,
            'width': None
        },
        'elbow': {
            'center': elbow_coords,
            'top': elbow_top,
            'bottom': elbow_bottom,
            'width': None
        },
        'wrist': {
            'center': wrist_coords,
            'top': wrist_top,
            'bottom': wrist_bottom,
            'width': None
        }
    }
    
    # Calculate widths and debug output
    for landmark_name, data in perpendicular_data.items():
        top = data['top']
        bottom = data['bottom']
        
        if top and bottom:
            width = np.sqrt((top[0] - bottom[0])**2 + 
                          (top[1] - bottom[1])**2)
            data['width'] = width
            print(f"  {landmark_name.capitalize()}: width = {width:.1f}px, "
                  f"top={top}, bottom={bottom}")
        elif top:
            print(f"  {landmark_name.capitalize()}: top only = {top}")
        elif bottom:
            print(f"  {landmark_name.capitalize()}: bottom only = {bottom}")
        else:
            print(f"  {landmark_name.capitalize()}: Could not find intersections")
    
    # ========================================================================
    # ADD SEMI-LANDMARKS (SLIDING LANDMARKS)
    # ========================================================================
    
    print("\n[OK] Adding semi-landmarks (sliding landmarks)...")
    
    # Add 3 semi-landmarks between shoulder and elbow (upper arm)
    upper_arm_semi_landmarks = add_semi_landmarks(
        shoulder_coords, elbow_coords, shoulder_direction, mask, num_divisions=3
    )
    
    # Add 3 semi-landmarks between elbow and wrist (forearm)
    forearm_semi_landmarks = add_semi_landmarks(
        elbow_coords, wrist_coords, wrist_direction, mask, num_divisions=3
    )
    
    print(f"  Upper arm: {len(upper_arm_semi_landmarks)} semi-landmarks")
    print(f"  Forearm: {len(forearm_semi_landmarks)} semi-landmarks")
    
    # ========================================================================
    # CREATE DEBUG VISUALIZATIONS
    # ========================================================================
    
    print("\n[OK] Creating pixel-level debug visualizations...")
    
    import matplotlib.pyplot as plt
    
    debug_shoulder = create_debug_visualization('shoulder', shoulder_coords, shoulder_direction, mask)
    debug_elbow = create_debug_visualization('elbow', elbow_coords, elbow_direction, mask)
    debug_wrist = create_debug_visualization('wrist', wrist_coords, wrist_direction, mask)
    
    # Display debug visualizations
    fig_debug, axes_debug = plt.subplots(1, 3, figsize=(18, 6))
    
    if debug_shoulder is not None:
        axes_debug[0].imshow(debug_shoulder)
        axes_debug[0].set_title('Shoulder - Pixel Level Analysis', fontsize=12, fontweight='bold')
        axes_debug[0].axis('off')
    
    if debug_elbow is not None:
        axes_debug[1].imshow(debug_elbow)
        axes_debug[1].set_title('Elbow - Pixel Level Analysis', fontsize=12, fontweight='bold')
        axes_debug[1].axis('off')
    
    if debug_wrist is not None:
        axes_debug[2].imshow(debug_wrist)
        axes_debug[2].set_title('Wrist - Pixel Level Analysis', fontsize=12, fontweight='bold')
        axes_debug[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_perpendicular_analysis.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("[OK] Debug visualization saved: debug_perpendicular_analysis.jpg")
    print("  - Light blue line = perpendicular direction")
    print("  - Yellow line = arm axis direction")
    print("  - Magenta dots = candidate contour points (within threshold)")
    print("  - Red dot = landmark center")
    
    # Segment the arm area
    arm_segmented = cv2.bitwise_and(image, image, mask=mask)
    
    # Display results - Overlay mask on original image with perpendicular lines
    import matplotlib.pyplot as plt
    
    # Create visualization with mask overlay
    img_with_overlay = image.copy()
    
    # Create colored mask overlay (yellow-green like target image)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask * 0.3  # Blue channel (low)
    mask_colored[:, :, 1] = mask  # Green channel (full)
    mask_colored[:, :, 2] = mask * 0.8  # Red channel (high) - creates yellow-green
    
    # Blend the mask with original image
    alpha = 0.5  # Transparency factor
    img_with_overlay = cv2.addWeighted(img_with_overlay, 1, mask_colored, alpha, 0)
    
    # Draw contour edge - MUST use the SAME mask that was used for green overlay
    # Find contours from the FINAL mask (after all processing)
    contours_final, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_final:
        # Get the largest contour (should be the arm)
        main_contour_final = max(contours_final, key=cv2.contourArea)
        
        # Verify this contour matches the masked area
        print(f"\n[OK] Drawing contour with {len(main_contour_final)} points")
        print(f"  Contour area: {cv2.contourArea(main_contour_final):.0f} pixels")
        print(f"  Mask area: {np.count_nonzero(mask)} pixels")
        
        # Draw thick solid black contour outline
        cv2.drawContours(img_with_overlay, [main_contour_final], -1, (0, 0, 0), 3)
    
    # Draw perpendicular lines and intersection points
    cyan_color = (255, 255, 0)  # Cyan for perpendicular lines (BGR format)
    magenta_color = (255, 0, 255)  # Magenta for ALL intersection points (top/bottom)
    red_color = (0, 0, 255)  # Red for main pose landmark centers only (BGR format)
    
    for landmark_name, data in perpendicular_data.items():
        center = data['center']
        top = data['top']
        bottom = data['bottom']
        
        # Draw based on what points are available
        if top and bottom:
            # Both points available - draw thin perpendicular line
            cv2.line(img_with_overlay, top, bottom, cyan_color, 1)
            
            # Draw small MAGENTA intersection points
            cv2.circle(img_with_overlay, top, 3, magenta_color, -1)
            cv2.circle(img_with_overlay, top, 4, (0, 0, 0), 1)
            
            cv2.circle(img_with_overlay, bottom, 3, magenta_color, -1)
            cv2.circle(img_with_overlay, bottom, 4, (0, 0, 0), 1)
            
        elif top:
            # Only top point available (shoulder case)
            # Draw thin line from top to center
            cv2.line(img_with_overlay, top, center, cyan_color, 1)
            
            # Draw small MAGENTA intersection point at top
            cv2.circle(img_with_overlay, top, 3, magenta_color, -1)
            cv2.circle(img_with_overlay, top, 4, (0, 0, 0), 1)
    
    # Draw small RED dots at main pose landmark centers (shoulder, elbow, wrist)
    landmark_radius = 4
    
    cv2.circle(img_with_overlay, shoulder_coords, landmark_radius, red_color, -1)
    cv2.circle(img_with_overlay, shoulder_coords, landmark_radius + 1, (0, 0, 0), 1)
    
    cv2.circle(img_with_overlay, elbow_coords, landmark_radius, red_color, -1)
    cv2.circle(img_with_overlay, elbow_coords, landmark_radius + 1, (0, 0, 0), 1)
    
    cv2.circle(img_with_overlay, wrist_coords, landmark_radius, red_color, -1)
    cv2.circle(img_with_overlay, wrist_coords, landmark_radius + 1, (0, 0, 0), 1)
    
    # ========================================================================
    # DRAW SEMI-LANDMARKS (SLIDING LANDMARKS)
    # ========================================================================
    
    # Draw upper arm semi-landmarks
    for semi_data in upper_arm_semi_landmarks:
        if semi_data['top'] and semi_data['bottom']:
            # Draw thin perpendicular line
            cv2.line(img_with_overlay, semi_data['top'], semi_data['bottom'], cyan_color, 1)
            
            # Draw small magenta intersection points
            cv2.circle(img_with_overlay, semi_data['top'], 3, magenta_color, -1)
            cv2.circle(img_with_overlay, semi_data['top'], 4, (0, 0, 0), 1)
            
            cv2.circle(img_with_overlay, semi_data['bottom'], 3, magenta_color, -1)
            cv2.circle(img_with_overlay, semi_data['bottom'], 4, (0, 0, 0), 1)
            
            # Draw tiny cyan dot at semi-landmark center
            cv2.circle(img_with_overlay, semi_data['center'], 2, cyan_color, -1)
    
    # Draw forearm semi-landmarks
    for semi_data in forearm_semi_landmarks:
        if semi_data['top'] and semi_data['bottom']:
            # Draw thin perpendicular line
            cv2.line(img_with_overlay, semi_data['top'], semi_data['bottom'], cyan_color, 1)
            
            # Draw small magenta intersection points
            cv2.circle(img_with_overlay, semi_data['top'], 3, magenta_color, -1)
            cv2.circle(img_with_overlay, semi_data['top'], 4, (0, 0, 0), 1)
            
            cv2.circle(img_with_overlay, semi_data['bottom'], 3, magenta_color, -1)
            cv2.circle(img_with_overlay, semi_data['bottom'], 4, (0, 0, 0), 1)
            
            # Draw tiny cyan dot at semi-landmark center
            cv2.circle(img_with_overlay, semi_data['center'], 2, cyan_color, -1)
    
    # Display single image with overlay
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Arm Segmentation with Perpendicular Lines at Landmarks', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save results
    cv2.imwrite('arm_mask_overlay.jpg', img_with_overlay)
    print("\n[OK] Result saved: arm_mask_overlay.jpg")
    
    # ========================================================================
    # EXPORT LANDMARKS TO CSV
    # ========================================================================
    
    print("\n[OK] Exporting landmarks to CSV...")
    
    import csv
    
    # Prepare landmark data for CSV export
    landmark_data = []
    landmark_id = 1
    
    # Add main anatomical landmarks (RED)
    # Shoulder top
    if shoulder_top:
        landmark_data.append({
            'Landmark_ID': landmark_id,
            'Type': 'Anatomical',
            'Category': 'Main',
            'Name': 'Shoulder_Top',
            'X_Pixel': shoulder_top[0],
            'Y_Pixel': shoulder_top[1],
            'Width_Pixel': 'N/A',
            'Color': 'red'
        })
        landmark_id += 1
    
    # Shoulder center
    landmark_data.append({
        'Landmark_ID': landmark_id,
        'Type': 'Anatomical',
        'Category': 'Main',
        'Name': 'Shoulder_Center',
        'X_Pixel': shoulder_coords[0],
        'Y_Pixel': shoulder_coords[1],
        'Width_Pixel': 'N/A',
        'Color': 'red'
    })
    landmark_id += 1
    
    # Elbow top
    if elbow_top:
        landmark_data.append({
            'Landmark_ID': landmark_id,
            'Type': 'Anatomical',
            'Category': 'Main',
            'Name': 'Elbow_Top',
            'X_Pixel': elbow_top[0],
            'Y_Pixel': elbow_top[1],
            'Width_Pixel': 'N/A',
            'Color': 'magenta'
        })
        landmark_id += 1
    
    # Elbow center
    landmark_data.append({
        'Landmark_ID': landmark_id,
        'Type': 'Anatomical',
        'Category': 'Main',
        'Name': 'Elbow_Center',
        'X_Pixel': elbow_coords[0],
        'Y_Pixel': elbow_coords[1],
        'Width_Pixel': 'N/A',
        'Color': 'red'
    })
    landmark_id += 1
    
    # Elbow bottom
    if elbow_bottom:
        landmark_data.append({
            'Landmark_ID': landmark_id,
            'Type': 'Anatomical',
            'Category': 'Main',
            'Name': 'Elbow_Bottom',
            'X_Pixel': elbow_bottom[0],
            'Y_Pixel': elbow_bottom[1],
            'Width_Pixel': 'N/A',
            'Color': 'magenta'
        })
        landmark_id += 1
    
    # Wrist top
    if wrist_top:
        landmark_data.append({
            'Landmark_ID': landmark_id,
            'Type': 'Anatomical',
            'Category': 'Main',
            'Name': 'Wrist_Top',
            'X_Pixel': wrist_top[0],
            'Y_Pixel': wrist_top[1],
            'Width_Pixel': 'N/A',
            'Color': 'magenta'
        })
        landmark_id += 1
    
    # Wrist center
    landmark_data.append({
        'Landmark_ID': landmark_id,
        'Type': 'Anatomical',
        'Category': 'Main',
        'Name': 'Wrist_Center',
        'X_Pixel': wrist_coords[0],
        'Y_Pixel': wrist_coords[1],
        'Width_Pixel': 'N/A',
        'Color': 'red'
    })
    landmark_id += 1
    
    # Wrist bottom
    if wrist_bottom:
        landmark_data.append({
            'Landmark_ID': landmark_id,
            'Type': 'Anatomical',
            'Category': 'Main',
            'Name': 'Wrist_Bottom',
            'X_Pixel': wrist_bottom[0],
            'Y_Pixel': wrist_bottom[1],
            'Width_Pixel': 'N/A',
            'Color': 'magenta'
        })
        landmark_id += 1
    
    # Add upper arm semi-landmarks (MAGENTA)
    for i, semi_data in enumerate(upper_arm_semi_landmarks, 1):
        if semi_data['top']:
            landmark_data.append({
                'Landmark_ID': landmark_id,
                'Type': 'Semilandmark',
                'Category': 'UpperArm',
                'Name': f'UpperArm_Semi{i}_Top',
                'X_Pixel': semi_data['top'][0],
                'Y_Pixel': semi_data['top'][1],
                'Width_Pixel': 'N/A',
                'Color': 'magenta'
            })
            landmark_id += 1
        
        if semi_data['bottom']:
            landmark_data.append({
                'Landmark_ID': landmark_id,
                'Type': 'Semilandmark',
                'Category': 'UpperArm',
                'Name': f'UpperArm_Semi{i}_Bottom',
                'X_Pixel': semi_data['bottom'][0],
                'Y_Pixel': semi_data['bottom'][1],
                'Width_Pixel': 'N/A',
                'Color': 'magenta'
            })
            landmark_id += 1
    
    # Add forearm semi-landmarks (MAGENTA)
    for i, semi_data in enumerate(forearm_semi_landmarks, 1):
        if semi_data['top']:
            landmark_data.append({
                'Landmark_ID': landmark_id,
                'Type': 'Semilandmark',
                'Category': 'Forearm',
                'Name': f'Forearm_Semi{i}_Top',
                'X_Pixel': semi_data['top'][0],
                'Y_Pixel': semi_data['top'][1],
                'Width_Pixel': 'N/A',
                'Color': 'magenta'
            })
            landmark_id += 1
        
        if semi_data['bottom']:
            landmark_data.append({
                'Landmark_ID': landmark_id,
                'Type': 'Semilandmark',
                'Category': 'Forearm',
                'Name': f'Forearm_Semi{i}_Bottom',
                'X_Pixel': semi_data['bottom'][0],
                'Y_Pixel': semi_data['bottom'][1],
                'Width_Pixel': 'N/A',
                'Color': 'magenta'
            })
            landmark_id += 1
    
    # Write to CSV file
    csv_filename = 'arm_landmarks_output.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Landmark_ID', 'Type', 'Category', 'Name', 'X_Pixel', 'Y_Pixel', 'Width_Pixel', 'Color']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in landmark_data:
            writer.writerow(row)
    
    print(f"[OK] Landmarks exported to: {csv_filename}")
    print(f"  Total landmarks: {len(landmark_data)}")
    print(f"  Main anatomical landmarks: {len([l for l in landmark_data if l['Type'] == 'Anatomical'])}")
    print(f"  Semi-landmarks: {len([l for l in landmark_data if l['Type'] == 'Semilandmark'])}")
    
    # ========================================================================
    # VISUALIZE LANDMARK LINKAGE (WIREFRAME)
    # ========================================================================
    
    def visualize_landmark_linkage(csv_file, output_image='arm_landmark_linkage.jpg'):
        """
        Create a wireframe visualization of landmarks by linking all points
        Shows only the landmark connections without the actual arm image
        
        Args:
            csv_file: Path to the CSV file containing landmark coordinates
            output_image: Output filename for the linkage visualization
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        print(f"\n[OK] Creating landmark linkage visualization from {csv_file}...")
        
        # Create blank canvas
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        
        # Get image dimensions from landmark coordinates
        x_coords = df['X_Pixel'].values
        y_coords = df['Y_Pixel'].values
        
        x_min, x_max = x_coords.min() - 50, x_coords.max() + 50
        y_min, y_max = y_coords.min() - 50, y_coords.max() + 50
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invert y-axis to match image coordinates
        
        # Define color mapping
        color_map = {
            'red': 'red',
            'magenta': 'magenta',
            'cyan': 'cyan',
            'yellow': 'yellow'
        }
        
        # Group landmarks by category for linking
        main_landmarks = df[df['Type'] == 'Anatomical'].copy()
        upper_arm_semis = df[df['Category'] == 'UpperArm'].copy()
        forearm_semis = df[df['Category'] == 'Forearm'].copy()
        
        # Link main anatomical landmarks (shoulder -> elbow -> wrist)
        main_centers = main_landmarks[main_landmarks['Name'].str.contains('Center')]
        if len(main_centers) >= 2:
            for i in range(len(main_centers) - 1):
                x1, y1 = main_centers.iloc[i]['X_Pixel'], main_centers.iloc[i]['Y_Pixel']
                x2, y2 = main_centers.iloc[i+1]['X_Pixel'], main_centers.iloc[i+1]['Y_Pixel']
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6, label='Main axis' if i == 0 else '')
        
        # Collect all top edge points and bottom edge points in order
        top_points = []
        bottom_points = []
        
        # Shoulder top
        shoulder_top_data = df[df['Name'] == 'Shoulder_Top']
        if len(shoulder_top_data) > 0:
            top_points.append((shoulder_top_data.iloc[0]['X_Pixel'], shoulder_top_data.iloc[0]['Y_Pixel']))
        
        # Upper arm semi-landmarks (top and bottom)
        for i in range(1, 4):  # 3 semi-landmarks
            top_data = df[df['Name'] == f'UpperArm_Semi{i}_Top']
            bottom_data = df[df['Name'] == f'UpperArm_Semi{i}_Bottom']
            
            if len(top_data) > 0:
                top_points.append((top_data.iloc[0]['X_Pixel'], top_data.iloc[0]['Y_Pixel']))
            if len(bottom_data) > 0:
                bottom_points.append((bottom_data.iloc[0]['X_Pixel'], bottom_data.iloc[0]['Y_Pixel']))
        
        # Elbow top and bottom
        elbow_top_data = df[df['Name'] == 'Elbow_Top']
        elbow_bottom_data = df[df['Name'] == 'Elbow_Bottom']
        if len(elbow_top_data) > 0:
            top_points.append((elbow_top_data.iloc[0]['X_Pixel'], elbow_top_data.iloc[0]['Y_Pixel']))
        if len(elbow_bottom_data) > 0:
            bottom_points.append((elbow_bottom_data.iloc[0]['X_Pixel'], elbow_bottom_data.iloc[0]['Y_Pixel']))
        
        # Forearm semi-landmarks (top and bottom)
        for i in range(1, 4):  # 3 semi-landmarks
            top_data = df[df['Name'] == f'Forearm_Semi{i}_Top']
            bottom_data = df[df['Name'] == f'Forearm_Semi{i}_Bottom']
            
            if len(top_data) > 0:
                top_points.append((top_data.iloc[0]['X_Pixel'], top_data.iloc[0]['Y_Pixel']))
            if len(bottom_data) > 0:
                bottom_points.append((bottom_data.iloc[0]['X_Pixel'], bottom_data.iloc[0]['Y_Pixel']))
        
        # Wrist top and bottom
        wrist_top_data = df[df['Name'] == 'Wrist_Top']
        wrist_bottom_data = df[df['Name'] == 'Wrist_Bottom']
        if len(wrist_top_data) > 0:
            top_points.append((wrist_top_data.iloc[0]['X_Pixel'], wrist_top_data.iloc[0]['Y_Pixel']))
        if len(wrist_bottom_data) > 0:
            bottom_points.append((wrist_bottom_data.iloc[0]['X_Pixel'], wrist_bottom_data.iloc[0]['Y_Pixel']))
        
        # Draw continuous line connecting all top edge points
        if len(top_points) > 1:
            top_x = [p[0] for p in top_points]
            top_y = [p[1] for p in top_points]
            ax.plot(top_x, top_y, 'r-', linewidth=2, alpha=0.7, label='Top edge')
        
        # Draw continuous line connecting all bottom edge points
        if len(bottom_points) > 1:
            bottom_x = [p[0] for p in bottom_points]
            bottom_y = [p[1] for p in bottom_points]
            ax.plot(bottom_x, bottom_y, 'r-', linewidth=2, alpha=0.7, label='Bottom edge')
        
        # Draw perpendicular lines connecting top to bottom at each position
        # Link shoulder top to first bottom point if exists
        if len(top_points) > 0 and len(bottom_points) > 0:
            # Connect first top to first bottom (shoulder area)
            ax.plot([top_points[0][0], bottom_points[0][0]], 
                   [top_points[0][1], bottom_points[0][1]], 
                   'gray', linewidth=1, alpha=0.3, linestyle='--')
            
            # Connect remaining pairs
            for i in range(1, min(len(top_points), len(bottom_points))):
                ax.plot([top_points[i][0], bottom_points[i][0]], 
                       [top_points[i][1], bottom_points[i][1]], 
                       'gray', linewidth=1, alpha=0.3, linestyle='--')
        
        # Draw all landmark points
        for idx, row in df.iterrows():
            x, y = row['X_Pixel'], row['Y_Pixel']
            color = color_map.get(row['Color'], 'gray')
            
            if row['Type'] == 'Anatomical' and 'Center' in row['Name']:
                # Main landmarks - larger red dots
                ax.plot(x, y, 'o', color='red', markersize=8, markeredgecolor='black', markeredgewidth=1)
                # Add landmark number
                ax.text(x, y-10, str(row['Landmark_ID']), fontsize=8, ha='center', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            else:
                # Edge intersection points - smaller magenta dots
                ax.plot(x, y, 'o', color=color, markersize=5, markeredgecolor='black', markeredgewidth=0.5)
        
        ax.set_title('Arm Landmark Linkage Visualization (Wireframe)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate (pixels)', fontsize=10)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"[OK] Linkage visualization saved: {output_image}")
        print(f"  Total points visualized: {len(df)}")
    
    # Call the visualization function
    visualize_landmark_linkage(csv_filename)
    
else:
    print("No pose detected!")
