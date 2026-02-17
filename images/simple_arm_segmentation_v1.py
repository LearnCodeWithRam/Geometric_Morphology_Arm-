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
    
    print("\n✓ Using MediaPipe Selfie Segmentation for accurate arm edges")
    
    # Define arm boundaries (min/max x coordinates from shoulder to wrist)
    x_min_boundary = min(shoulder_coords[0], wrist_coords[0])
    x_max_boundary = max(shoulder_coords[0], wrist_coords[0])
    
    print(f"✓ Arm boundaries: x from {x_min_boundary} to {x_max_boundary}")
    
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
    
    # Generous thickness for ROI
    upper_arm_thickness = int(upper_arm_length * 0.55)
    forearm_thickness = int(forearm_length * 0.50)
    
    # Draw ROI along arm path
    cv2.line(mask_roi, shoulder_coords, elbow_coords, 255, upper_arm_thickness)
    cv2.line(mask_roi, elbow_coords, wrist_coords, 255, forearm_thickness)
    cv2.circle(mask_roi, shoulder_coords, upper_arm_thickness // 2, 255, -1)
    cv2.circle(mask_roi, elbow_coords, max(upper_arm_thickness, forearm_thickness) // 2, 255, -1)
    cv2.circle(mask_roi, wrist_coords, forearm_thickness // 2, 255, -1)
    
    # Expand ROI slightly
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
    
    print(f"✓ Mask created with MediaPipe segmentation + boundary constraints")
    print(f"  Mask coverage: {np.count_nonzero(mask)} pixels")
    print(f"  Constrained to x-range: {x_min_boundary} to {x_max_boundary}")
    
    # Segment the arm area
    arm_segmented = cv2.bitwise_and(image, image, mask=mask)
    
    # Display results - Overlay mask on original image
    import matplotlib.pyplot as plt
    
    # Create visualization with mask overlay
    img_with_overlay = image.copy()
    
    # Create colored mask overlay (semi-transparent green)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 1] = mask  # Green channel
    
    # Blend the mask with original image
    alpha = 0.4  # Transparency factor
    img_with_overlay = cv2.addWeighted(img_with_overlay, 1, mask_colored, alpha, 0)
    
    # Draw smaller landmark points (standard size)
    landmark_radius = 5  # Smaller radius
    landmark_thickness = 2
    
    cv2.circle(img_with_overlay, shoulder_coords, landmark_radius, (255, 0, 0), -1)
    cv2.circle(img_with_overlay, shoulder_coords, landmark_radius + 1, (0, 0, 0), landmark_thickness)
    
    cv2.circle(img_with_overlay, elbow_coords, landmark_radius, (0, 255, 0), -1)
    cv2.circle(img_with_overlay, elbow_coords, landmark_radius + 1, (0, 0, 0), landmark_thickness)
    
    cv2.circle(img_with_overlay, wrist_coords, landmark_radius, (0, 0, 255), -1)
    cv2.circle(img_with_overlay, wrist_coords, landmark_radius + 1, (0, 0, 0), landmark_thickness)
    
    # Draw thinner connecting lines
    cv2.line(img_with_overlay, shoulder_coords, elbow_coords, (255, 255, 0), 2)
    cv2.line(img_with_overlay, elbow_coords, wrist_coords, (255, 255, 0), 2)
    
    # Display single image with overlay
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Arm Segmentation with MediaPipe (Accurate Edge Detection)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save results
    cv2.imwrite('arm_mask_overlay.jpg', img_with_overlay)
    print("\n✓ Result saved: arm_mask_overlay.jpg")
    
else:
    print("No pose detected!")
