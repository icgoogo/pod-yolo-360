---
library_name: transformers
license: apache-2.0
language:
  - en
pipeline_tag: object-detection
tags:
  - object-detection
  - vision
datasets:
  - coco
widget:
  - src: >-
      https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg
    example_title: Savanna
  - src: >-
      https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg
    example_title: Football Match
  - src: >-
      https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg
    example_title: Airport
---
## RT-DETRv2

### **Overview**

The RT-DETRv2 model was proposed in [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140) by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu. RT-DETRv2 refines RT-DETR by introducing selective multi-scale feature extraction, a discrete sampling operator for broader deployment compatibility, and improved training strategies like dynamic data augmentation and scale-adaptive hyperparameters. 
These changes enhance flexibility and practicality while maintaining real-time performance.

This model was contributed by [@jadechoghari](https://x.com/jadechoghari) with the help of [@cyrilvallez](https://huggingface.co/cyrilvallez) and [@qubvel-hf](https://huggingface.co/qubvel-hf)

This is 
### **Performance**

RT-DETRv2 consistently outperforms its predecessor across all model sizes while maintaining the same real-time speeds.

![rt-detr-v2-graph.png](https://huggingface.co/datasets/jadechoghari/images/resolve/main/rt-detr-v2-graph.png)

### **How to use**

```python
import torch
import requests

from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
     outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

for result in results:
     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
         score, label = score.item(), label_id.item()
         box = [round(i, 2) for i in box.tolist()]
         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
```

```
cat: 0.97 [341.14, 25.11, 639.98, 372.89]
cat: 0.96 [12.78, 56.35, 317.67, 471.34]
remote: 0.95 [39.96, 73.12, 175.65, 117.44]
sofa: 0.86 [-0.11, 2.97, 639.89, 473.62]
sofa: 0.82 [-0.12, 1.78, 639.87, 473.52]
remote: 0.79 [333.65, 76.38, 370.69, 187.48]
```

### **Training**

RT-DETRv2 is trained on COCO (Lin et al. [2014]) train2017 and validated on COCO val2017 dataset. We report the standard AP metrics (averaged over uniformly sampled IoU thresholds ranging from 0.50 − 0.95 with a step size of 0.05), and APval50 commonly used in real scenarios.

### **Applications**

RT-DETRv2 is ideal for real-time object detection in diverse applications such as **autonomous driving**, **surveillance systems**, **robotics**, and **retail analytics**. Its enhanced flexibility and deployment-friendly design make it suitable for both edge devices and large-scale systems + ensures high accuracy and speed in dynamic, real-world environments.