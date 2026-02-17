"""
ARM MORPHOLOGY ANALYSIS - Complete Implementation
Creates 20 equidistant landmark points along arm contour with measurements
Integrates SAM (Segment Anything Model) for automatic arm segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class ArmMorphologyAnalyzer:
    """Complete arm morphology analysis system with SAM integration"""
    
    def __init__(self, pose_model_path='images/pose_landmarker_heavy.task',
                 sam_checkpoint='images/sam_vit_h_4b8939.pth',
                 sam_model_type='vit_h'):
        """Initialize the analyzer with MediaPipe pose detector and SAM"""
        self.pose_model_path = pose_model_path
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.detector = None
        self.sam = None
        self.mask_generator = None
        self.image = None
        self.image_rgb = None
        self.landmarks = None
        self.shoulder_px = None
        self.elbow_px = None
        self.wrist_px = None
        self.upper_arm_length = None
        self.forearm_length = None
        self.total_arm_length = None
        self.arm_contour = None
        self.arm_mask = None
        self.cropped_arm = None
        self.crop_bbox = None
        self.landmark_points = []
        
    def initialize_detector(self):
        """Initialize MediaPipe Pose Landmarker"""
        base_options = python.BaseOptions(model_asset_path=self.pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("✓ MediaPipe Pose Detector initialized")
        
    def initialize_sam(self):
        """Initialize SAM (Segment Anything Model)"""
        try:
            import torch
            print(f"Loading SAM model: {self.sam_model_type}...")
            self.sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
            
            # Check for CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sam.to(device=device)
            print(f"✓ SAM model loaded on {device}")
            
            # Create mask generator
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            print("✓ SAM Automatic Mask Generator initialized")
        except Exception as e:
            print(f"Error initializing SAM: {e}")
            raise
        
    def load_image(self, image_path):
        """Load image for analysis"""
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"✓ Image loaded: {self.image_rgb.shape}")
        return self.image_rgb
        
    def detect_pose_landmarks(self, image_path):
        """Detect pose landmarks using MediaPipe"""
        mp_image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            raise ValueError("No pose landmarks detected!")
            
        self.landmarks = detection_result.pose_landmarks[0]
        print(f"✓ Detected {len(self.landmarks)} pose landmarks")
        return detection_result
        
    def extract_arm_landmarks(self):
        """Extract shoulder, elbow, wrist landmarks"""
        h, w = self.image_rgb.shape[:2]
        
        # Landmark indices
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15
        
        shoulder = self.landmarks[LEFT_SHOULDER]
        elbow = self.landmarks[LEFT_ELBOW]
        wrist = self.landmarks[LEFT_WRIST]
        
        # Convert to pixel coordinates
        self.shoulder_px = (int(shoulder.x * w), int(shoulder.y * h))
        self.elbow_px = (int(elbow.x * w), int(elbow.y * h))
        self.wrist_px = (int(wrist.x * w), int(wrist.y * h))
        
        print(f"✓ Shoulder: {self.shoulder_px}")
        print(f"✓ Elbow: {self.elbow_px}")
        print(f"✓ Wrist: {self.wrist_px}")
        
        return self.shoulder_px, self.elbow_px, self.wrist_px
        
    def calculate_arm_lengths(self):
        """Calculate Euclidean distances for arm segments"""
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        self.upper_arm_length = euclidean_distance(self.shoulder_px, self.elbow_px)
        self.forearm_length = euclidean_distance(self.elbow_px, self.wrist_px)
        self.total_arm_length = self.upper_arm_length + self.forearm_length
        
        print(f"\n{'='*60}")
        print("ARM LENGTH MEASUREMENTS")
        print(f"{'='*60}")
        print(f"Upper arm (shoulder→elbow): {self.upper_arm_length:.2f} px")
        print(f"Forearm (elbow→wrist): {self.forearm_length:.2f} px")
        print(f"Total arm length: {self.total_arm_length:.2f} px")
        print(f"{'='*60}\n")
        
        return self.upper_arm_length, self.forearm_length, self.total_arm_length
        
    def generate_sam_masks(self):
        """Generate segmentation masks using SAM"""
        print("Generating SAM masks (this may take a moment)...")
        masks = self.mask_generator.generate(self.image_rgb)
        print(f"✓ Generated {len(masks)} masks")
        return masks
        
    def extract_arm_contour_from_sam(self, sam_masks=None):
        """Extract arm contour from SAM segmentation masks"""
        if sam_masks is None:
            sam_masks = self.generate_sam_masks()
            
        print("Searching for arm mask containing landmarks...")
        
        # Strategy 1: Find mask containing all three landmarks
        for i, mask_data in enumerate(sam_masks):
            mask = mask_data['segmentation']
            
            # Check if all landmarks are inside this mask
            if (mask[self.shoulder_px[1], self.shoulder_px[0]] and
                mask[self.elbow_px[1], self.elbow_px[0]] and
                mask[self.wrist_px[1], self.wrist_px[0]]):
                
                print(f"✓ Found arm mask containing all 3 landmarks (mask #{i})")
                self.arm_mask = mask
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)
                
                if contours:
                    self.arm_contour = max(contours, key=cv2.contourArea)
                    print(f"✓ Arm contour extracted: {len(self.arm_contour)} points")
                    return self.arm_contour
        
        # Strategy 2: Find masks containing at least 2 landmarks and merge them
        print("No single mask contains all landmarks. Trying to merge masks...")
        candidate_masks = []
        
        for i, mask_data in enumerate(sam_masks):
            mask = mask_data['segmentation']
            landmark_count = 0
            
            if mask[self.shoulder_px[1], self.shoulder_px[0]]:
                landmark_count += 1
            if mask[self.elbow_px[1], self.elbow_px[0]]:
                landmark_count += 1
            if mask[self.wrist_px[1], self.wrist_px[0]]:
                landmark_count += 1
            
            if landmark_count >= 1:  # At least one landmark
                candidate_masks.append({
                    'mask': mask,
                    'count': landmark_count,
                    'area': mask_data['area'],
                    'index': i
                })
                print(f"  Mask #{i}: {landmark_count} landmarks, area={mask_data['area']:.0f}")
        
        if not candidate_masks:
            raise ValueError("Could not find any mask containing arm landmarks")
        
        # Sort by landmark count (descending) and area (descending)
        candidate_masks.sort(key=lambda x: (x['count'], x['area']), reverse=True)
        
        # Merge top candidate masks
        print(f"Merging top {min(3, len(candidate_masks))} candidate masks...")
        merged_mask = np.zeros_like(candidate_masks[0]['mask'], dtype=bool)
        
        for i in range(min(3, len(candidate_masks))):
            merged_mask = np.logical_or(merged_mask, candidate_masks[i]['mask'])
            print(f"  Added mask #{candidate_masks[i]['index']}")
        
        self.arm_mask = merged_mask
        
        # Extract contour from merged mask
        mask_uint8 = (merged_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)
        
        if contours:
            self.arm_contour = max(contours, key=cv2.contourArea)
            print(f"✓ Merged arm contour extracted: {len(self.arm_contour)} points")
            return self.arm_contour
        
        raise ValueError("Could not extract contour from merged masks")
        
    def filter_contour_by_landmarks(self):
        """Filter contour points between shoulder and wrist"""
        x_min = min(self.shoulder_px[0], self.wrist_px[0])
        x_max = max(self.shoulder_px[0], self.wrist_px[0])
        
        filtered_points = []
        for point in self.arm_contour:
            x, y = point[0]
            if x_min <= x <= x_max:
                filtered_points.append(point)
        
        self.arm_contour = np.array(filtered_points)
        print(f"✓ Filtered contour: {len(self.arm_contour)} points")
        return self.arm_contour
        
    def crop_arm_region(self):
        """Crop arm region from shoulder to wrist"""
        x_coords = self.arm_contour[:, 0, 0]
        y_coords = self.arm_contour[:, 0, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add small margin for width
        margin = 10
        y_min = max(0, y_min - margin)
        y_max = min(self.image_rgb.shape[0], y_max + margin)
        
        # Store bbox for reference
        self.crop_bbox = (x_min, y_min, x_max, y_max)
        
        # Crop the image
        self.cropped_arm = self.image_rgb[y_min:y_max, x_min:x_max].copy()
        
        # Adjust contour coordinates to cropped image space
        adjusted_contour = self.arm_contour.copy()
        adjusted_contour[:, 0, 0] -= x_min
        adjusted_contour[:, 0, 1] -= y_min
        self.arm_contour = adjusted_contour
        
        print(f"✓ Cropped arm: {self.cropped_arm.shape}")
        print(f"  Bounding box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        
        return self.cropped_arm, self.crop_bbox
        
    def save_intermediate_images(self):
        """Save intermediate processing images"""
        if self.arm_mask is not None and self.crop_bbox is not None:
            x_min, y_min, x_max, y_max = self.crop_bbox
            
            # Reconstruct original contour coordinates for visualization
            original_contour = self.arm_contour.copy()
            original_contour[:, 0, 0] += x_min
            original_contour[:, 0, 1] += y_min
            
            # Save original with contour
            img_with_contour = self.image_rgb.copy()
            cv2.drawContours(img_with_contour, [original_contour], -1, (0, 255, 0), 3)
            cv2.circle(img_with_contour, self.shoulder_px, 8, (255, 0, 0), -1)
            cv2.circle(img_with_contour, self.elbow_px, 8, (255, 0, 0), -1)
            cv2.circle(img_with_contour, self.wrist_px, 8, (255, 0, 0), -1)
            cv2.imwrite('original-image-with-arm-contour.jpg', 
                       cv2.cvtColor(img_with_contour, cv2.COLOR_RGB2BGR))
            
            # Save segmentation mask
            mask_vis = (self.arm_mask * 255).astype(np.uint8)
            cv2.imwrite('arm-segmentation-mask.jpg', mask_vis)
            
            print("✓ Saved: original-image-with-arm-contour.jpg")
            print("✓ Saved: arm-segmentation-mask.jpg")
            
        if self.cropped_arm is not None:
            # Save cropped arm
            cv2.imwrite('cropped-arm-contour-based.jpg',
                       cv2.cvtColor(self.cropped_arm, cv2.COLOR_RGB2BGR))
            
            # Save cropped with mask overlay
            if self.arm_mask is not None and self.crop_bbox is not None:
                x_min, y_min, x_max, y_max = self.crop_bbox
                cropped_with_mask = self.cropped_arm.copy()
                mask_overlay = np.zeros_like(cropped_with_mask)
                mask_overlay[:, :, 1] = 100  # Green overlay
                
                # Apply mask
                cropped_mask = self.arm_mask[y_min:y_max, x_min:x_max]
                for c in range(3):
                    cropped_with_mask[:, :, c] = np.where(
                        cropped_mask,
                        cropped_with_mask[:, :, c] * 0.7 + mask_overlay[:, :, c] * 0.3,
                        cropped_with_mask[:, :, c]
                    )
                
                cv2.imwrite('cropped-arm-with-mask-overlay.jpg',
                           cv2.cvtColor(cropped_with_mask.astype(np.uint8), cv2.COLOR_RGB2BGR))
                print("✓ Saved: cropped-arm-with-mask-overlay.jpg")
            
            print("✓ Saved: cropped-arm-contour-based.jpg")
        
    def create_20_landmark_points(self, num_landmarks=20):
        """
        Create 20 equidistant landmark points along arm length
        Returns list of landmark data with top/bottom intersection points
        """
        x_coords = self.arm_contour[:, 0, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        arm_width = x_max - x_min
        
        # Calculate spacing for landmarks (19 intervals for 20 points)
        spacing = arm_width / (num_landmarks - 1)
        
        print(f"\n{'='*60}")
        print(f"CREATING {num_landmarks} LANDMARK POINTS")
        print(f"{'='*60}")
        print(f"Arm width: {arm_width:.2f} px")
        print(f"Spacing between landmarks: {spacing:.2f} px")
        print(f"{'='*60}\n")
        
        self.landmark_points = []
        
        for i in range(num_landmarks):
            x_pos = x_min + (i * spacing)
            
            # Find intersection points at this x position
            intersections = self._find_contour_intersections(int(x_pos))
            
            if len(intersections) >= 2:
                top_point = intersections[0]
                bottom_point = intersections[-1]
                
                # Calculate width at this landmark
                width = bottom_point[1] - top_point[1]
                
                # Calculate distance from shoulder (in original scale)
                distance_from_shoulder = i * spacing
                
                landmark_data = {
                    'id': i + 1,
                    'x_position': int(x_pos),
                    'top_point': top_point,
                    'bottom_point': bottom_point,
                    'width': width,
                    'distance_from_shoulder': distance_from_shoulder
                }
                
                self.landmark_points.append(landmark_data)
                
                print(f"Landmark {i+1:2d}: x={int(x_pos):3d}, "
                      f"top=({top_point[0]},{top_point[1]}), "
                      f"bottom=({bottom_point[0]},{bottom_point[1]}), "
                      f"width={width:.1f}px")
        
        print(f"\n✓ Created {len(self.landmark_points)} landmark points")
        return self.landmark_points
        
    def _find_contour_intersections(self, x_line, tolerance=2):
        """Find intersection points where vertical line crosses contour"""
        intersections = []
        
        for point in self.arm_contour:
            x, y = point[0]
            if abs(x - x_line) <= tolerance:
                intersections.append((x, y))
        
        # Remove duplicates and sort by y
        intersections = sorted(list(set(intersections)), key=lambda p: p[1])
        
        return intersections
        
    def visualize_landmarks_on_arm(self, save_path='arm_with_20_landmarks.jpg'):
        """Visualize all 20 landmarks on arm contour"""
        vis_img = self.cropped_arm.copy()
        
        # Draw contour
        cv2.drawContours(vis_img, [self.arm_contour], -1, (0, 255, 0), 2)
        
        # Draw each landmark
        for landmark in self.landmark_points:
            x_pos = landmark['x_position']
            top = landmark['top_point']
            bottom = landmark['bottom_point']
            landmark_id = landmark['id']
            
            # Draw vertical line (dashed effect)
            for y in range(0, vis_img.shape[0], 8):
                cv2.line(vis_img, (x_pos, y), (x_pos, min(y+4, vis_img.shape[0])), 
                        (255, 255, 0), 1)
            
            # Draw intersection points
            cv2.circle(vis_img, top, 4, (255, 0, 0), -1)  # Red for top
            cv2.circle(vis_img, bottom, 4, (0, 0, 255), -1)  # Blue for bottom
            
            # Add landmark number
            mid_y = (top[1] + bottom[1]) // 2
            cv2.putText(vis_img, str(landmark_id), (x_pos - 10, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save visualization
        vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, vis_bgr)
        print(f"✓ Saved visualization: {save_path}")
        
        return vis_img
        
    def create_measurement_table(self, real_arm_length_cm=None):
        """Create detailed measurement table for all landmarks"""
        print(f"\n{'='*80}")
        print(f"{'ARM MORPHOLOGY MEASUREMENT TABLE':^80}")
        print(f"{'='*80}")
        
        if real_arm_length_cm:
            pixels_per_cm = self.total_arm_length / real_arm_length_cm
            print(f"Calibration: {pixels_per_cm:.2f} pixels/cm")
            print(f"Real arm length: {real_arm_length_cm} cm")
        else:
            print(f"Total arm length: {self.total_arm_length:.2f} pixels")
            print("(Provide real_arm_length_cm for cm measurements)")
        
        print(f"{'='*80}")
        print(f"{'ID':<4} {'X':<6} {'Top(x,y)':<15} {'Bottom(x,y)':<15} {'Width(px)':<12} {'Dist(px)':<10}")
        print(f"{'-'*80}")
        
        for lm in self.landmark_points:
            top_str = f"({lm['top_point'][0]},{lm['top_point'][1]})"
            bottom_str = f"({lm['bottom_point'][0]},{lm['bottom_point'][1]})"
            
            print(f"{lm['id']:<4} {lm['x_position']:<6} {top_str:<15} {bottom_str:<15} "
                  f"{lm['width']:<12.1f} {lm['distance_from_shoulder']:<10.1f}")
        
        print(f"{'='*80}\n")
        
        # If real measurements provided, show cm values
        if real_arm_length_cm:
            print(f"\n{'='*80}")
            print(f"{'MEASUREMENTS IN CENTIMETERS':^80}")
            print(f"{'='*80}")
            print(f"{'ID':<4} {'Distance(cm)':<15} {'Width(cm)':<12}")
            print(f"{'-'*80}")
            
            for lm in self.landmark_points:
                dist_cm = lm['distance_from_shoulder'] / pixels_per_cm
                width_cm = lm['width'] / pixels_per_cm
                print(f"{lm['id']:<4} {dist_cm:<15.2f} {width_cm:<12.2f}")
            
            print(f"{'='*80}\n")
        
    def load_cropped_arm_and_extract_contour(self, cropped_image_path):
        """Load pre-cropped arm image and extract contour"""
        cropped_img = cv2.imread(cropped_image_path)
        self.cropped_arm = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale and create binary mask
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            self.arm_contour = max(contours, key=cv2.contourArea)
            print(f"✓ Loaded cropped arm and extracted contour: {len(self.arm_contour)} points")
            return self.arm_contour
        else:
            raise ValueError("Could not find contour in cropped image")
    
    def run_complete_analysis(self, image_path, cropped_arm_path=None, 
                             num_landmarks=20, real_arm_length_cm=None,
                             use_sam=True):
        """Run complete arm morphology analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING ARM MORPHOLOGY ANALYSIS")
        print("="*80 + "\n")
        
        # Step 1: Initialize detectors
        self.initialize_detector()
        if use_sam and cropped_arm_path is None:
            self.initialize_sam()
        
        # Step 2: Load image
        self.load_image(image_path)
        
        # Step 3: Detect pose landmarks
        detection_result = self.detect_pose_landmarks(image_path)
        
        # Step 4: Extract arm landmarks
        self.extract_arm_landmarks()
        
        # Step 5: Calculate arm lengths
        self.calculate_arm_lengths()
        
        # Step 6: Get arm contour
        if cropped_arm_path:
            # Use pre-cropped image
            print("\nUsing pre-cropped arm image...")
            self.load_cropped_arm_and_extract_contour(cropped_arm_path)
        elif use_sam:
            # Use SAM for segmentation
            print("\nUsing SAM for arm segmentation...")
            self.extract_arm_contour_from_sam()
            self.filter_contour_by_landmarks()
            self.crop_arm_region()
            self.save_intermediate_images()
        else:
            raise ValueError("Must provide either cropped_arm_path or set use_sam=True")
        
        # Step 7: Create landmark points
        self.create_20_landmark_points(num_landmarks)
        
        # Step 8: Visualize
        vis_img = self.visualize_landmarks_on_arm()
        
        # Step 9: Create measurement table
        self.create_measurement_table(real_arm_length_cm)
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80 + "\n")
        
        return {
            'landmarks': self.landmark_points,
            'visualization': vis_img,
            'arm_lengths': {
                'upper_arm': self.upper_arm_length,
                'forearm': self.forearm_length,
                'total': self.total_arm_length
            }
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ArmMorphologyAnalyzer(
        pose_model_path='images/pose_landmarker_heavy.task',
        sam_checkpoint='images/sam_vit_h_4b8939.pth',
        sam_model_type='vit_h'
    )
    
    # Option 1: Run complete analysis with SAM (automatic segmentation)
    print("Running complete analysis with SAM integration...")
    results = analyzer.run_complete_analysis(
        image_path='images/real.png',
        use_sam=True,  # Use SAM for automatic arm segmentation
        num_landmarks=20,
        real_arm_length_cm=25.0  # Optional: provide real arm length in cm
    )
    
    # Option 2: Use pre-cropped image (faster, if you already have it)
    # results = analyzer.run_complete_analysis(
    #     image_path='images/real.png',
    #     cropped_arm_path='cropped-arm-with-mask-overlay.jpg',
    #     num_landmarks=20,
    #     real_arm_length_cm=25.0
    # )
    
    print("Analysis results available in 'results' dictionary")
    print(f"Total landmarks created: {len(results['landmarks'])}")
    
    # Display the visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(results['visualization'])
    plt.title('Arm with 20 Landmark Points', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('final_arm_analysis.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Saved final visualization: final_arm_analysis.jpg")
