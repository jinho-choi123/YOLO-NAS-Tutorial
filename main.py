from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_M, pretrained_weights = "coco")

results = model.predict("./test-img/test1.jpg")

results.show(box_thickness=2, show_confidence=True)