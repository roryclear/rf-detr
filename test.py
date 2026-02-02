import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano, RFDETRSmall,  RFDETRMedium, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
import numpy as np

excepted_xyxys = [[[61.86511,247.66312,652.24835,930.8383],
 [1.3346028,361.53345,648.7615,1264.4553],
 [622.7613,720.39746,698.4292,787.9133,]],

 [[67.79058,248.21106,629.7574,928.6994, ],
[2.8710365,360.7552,576.2327,1256.1041, ],
[1.8651867,659.653,455.1194,1267.9921, ]],

[[67.15638,251.10565,634.4118,929.71875, ],
[2.1100616,654.9501,457.13464,1268.6119, ]],

[[69.10416,250.3548,629.37036,926.8027, ],
[1.3762093,659.6178,447.79425,1272.3656, ],
[625.56854,719.8756,696.6235,787.5058, ]],
]

models = [RFDETRNano(), RFDETRSmall(), RFDETRMedium(), RFDETRLarge()]
image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
for i, model in enumerate(models):
  detections = model.predict(image, threshold=0.5)
  labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
  annotated_image = sv.BoxAnnotator().annotate(image, detections)
  annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
  np.testing.assert_allclose(detections.xyxy, excepted_xyxys[i])
  annotated_image.save("annotated_image.jpg")