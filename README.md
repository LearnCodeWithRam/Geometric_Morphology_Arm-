# Clinical Arm Morphology Analysis

Automated arm morphology analysis using MediaPipe for pose detection and segmentation.

## 🚀 Quick Start

### Installation
```bash
pip install opencv-python numpy matplotlib mediapipe
```

### Download Models
```bash
# Create images directory
mkdir images

# Download pose model (~30 MB)
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task -O images/pose_landmarker_heavy.task

# Download segmentation model (~1 MB)
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite -O images/selfie_multiclass_256x256.tflite
```

### Run Analysis
```bash
python arm_morphology_analysis_v4.py
```

## 📊 Features

- ✅ **18 Landmarks**: 4 anatomical (red) + 14 semilandmarks (yellow)
- ✅ **Fast Processing**: ~3 seconds per image
- ✅ **Lightweight**: Only 31 MB total model size
- ✅ **Accurate**: Clinical-grade landmark detection
- ✅ **CSV Export**: Machine-readable landmark data
- ✅ **No GPU Required**: Runs on CPU efficiently

## 🎯 Output

### Files Generated
1. `clinical_arm_analysis_18_landmarks.jpg` - Visualization
2. `arm_landmarks.csv` - Landmark coordinates
3. `left_arm_cropped.jpg` - Cropped arm region

### Landmarks
- **Landmark 1**: Shoulder (lateral acromion)
- **Landmark 2**: Forearm (elbow-wrist midpoint)
- **Landmark 3**: Wrist (palmar midcarpal joint)
- **Landmark 4**: Armpit (trunk-arm intersection)
- **Landmarks 5-18**: Semilandmarks (equidistant along arm)

## 📈 Version Comparison

| Version | Model | Speed | Size | Status |
|---------|-------|-------|------|--------|
| **V4** ⭐ | MediaPipe Selfie | ~3s | 31 MB | **Recommended** |
| V3 | SAM | ~20s | 2.4 GB | Legacy |
| V2 | SAM | ~20s | 2.4 GB | Legacy |
| V1 | SAM | ~20s | 2.4 GB | Legacy |

## 💻 Usage

### Python API
```python
from arm_morphology_analysis_v4 import ClinicalArmAnalyzerV4

# Initialize
analyzer = ClinicalArmAnalyzerV4(
    pose_model_path='images/pose_landmarker_heavy.task',
    segmentation_model_path='images/selfie_multiclass_256x256.tflite'
)

# Run analysis
results = analyzer.run_complete_analysis(
    image_path='images/real.png',
    export_data=True
)

# Access results
print(f"Total landmarks: {results['total_landmarks']}")
print(f"CSV file: {results['csv_file']}")
```

### Batch Processing
```python
import os

analyzer = ClinicalArmAnalyzerV4()

for image_file in os.listdir('patient_images/'):
    if image_file.endswith('.png'):
        results = analyzer.run_complete_analysis(
            image_path=f'patient_images/{image_file}',
            export_data=True
        )
        print(f"Processed: {image_file}")
```

## 📁 Project Structure

```
.
├── arm_morphology_analysis_v4.py    # Main script (RECOMMENDED)
├── arm_morphology_analysis_v3.py    # SAM version
├── arm_morphology_analysis_v2.py    # SAM with smoothing
├── arm_morphology_analysis_v1.py    # Basic SAM
├── README.md                         # This file
├── V4_IMPROVEMENTS.md               # V4 details
├── VERSION_SUMMARY.md               # Version comparison
├── OUTPUT_FORMAT_GUIDE.md           # Output format specs
└── images/
    ├── pose_landmarker_heavy.task
    ├── selfie_multiclass_256x256.tflite
    └── real.png                     # Your test image
```

## 🔬 Clinical Applications

- Anthropometric studies
- Growth and development tracking
- Nutritional assessment (MUAC)
- Medical documentation
- Arm shape analysis
- Clinical research

## 📊 CSV Output Format

```csv
Landmark_ID,Type,Category,Name,X_Pixel,Y_Pixel,Width_Pixel,Color
1,Anatomical,Main,Shoulder,113,202,N/A,red
2,Anatomical,Main,Forearm,286,221,N/A,red
3,Anatomical,Main,Wrist,456,202,N/A,red
4,Anatomical,Main,Armpit,110,185,N/A,red
5,Semilandmark,Top,Semilandmark_5,150,190,45.50,yellow
...
```

## 🛠️ Troubleshooting

### No pose landmarks detected
- Ensure person is clearly visible
- Check image quality and lighting
- Verify model file exists

### Segmentation quality issues
- Use higher resolution images
- Ensure good contrast
- Check lighting conditions

### Landmarks outside contour
- Verify pose detection accuracy
- Check segmentation threshold
- Ensure arm is fully visible

## 📚 Documentation

- `V4_IMPROVEMENTS.md` - Detailed V4 improvements
- `VERSION_SUMMARY.md` - All version comparisons
- `OUTPUT_FORMAT_GUIDE.md` - Output file specifications

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project uses:
- MediaPipe (Apache 2.0)
- OpenCV (Apache 2.0)

## 🔗 References

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Image Segmentation](https://developers.google.com/mediapipe/solutions/vision/image_segmenter)
- SAM Photo Diagnosis App methodology

## 📞 Support

For issues or questions:
1. Check documentation files
2. Review troubleshooting section
3. Open an issue with details

---

**Version**: 4.0  
**Status**: Production Ready ✅  
**Last Updated**: 2026-02-13
