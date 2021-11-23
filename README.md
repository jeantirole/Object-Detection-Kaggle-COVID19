# Object_Detection_Kaggle_COVID19

### 실험 point

1. DICOM to png/jpg (np.array)
    - [https://www.kaggle.com/h053473666/siimcovid19-512-img-png-600-study-png?select=image](https://www.kaggle.com/h053473666/siimcovid19-512-img-png-600-study-png?select=image)
        1. 512 x 512 png image
        
    - [https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way](https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way)
        1. most of DICOM's store pixel values in exponential scale
        2. need to apply some transformations. DICOM metadata stores information how to make such "human-friendly" transformations.
        3. **fix_monochrome 사용**
        
    - [https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image](https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image)
        1. image resize 
        2. Filter : Lanczos 사용
            - **Filters**
                
                [https://ponyozzang.tistory.com/600](https://ponyozzang.tistory.com/600)
                
                - NEAREST
                - BOX
                - BILINEAR
                - HAMMING
                - BICUBIC
                - LANCZOS
        
2. Augmentation
    - Horizontal Flip
        1. bbox 위치 조정 필요
    - rare class augmentation
    - random_brightness
    
3. Loss 
    - auxiliary loss
        - [https://www.kaggle.com/c/siim-covid19-detection/discussion/240233](https://www.kaggle.com/c/siim-covid19-detection/discussion/240233)
        - 0.617인 친구가 auxiliary loss + meta pseudo labeling 사용중
            - 다른 competition pseuo labeling [https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling](https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling)
            
             
            
            ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/013c0fbd-c8fa-484f-a1f6-db7ba2ce8917/Screen_Shot_2021-06-20_at_2.17.48_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/013c0fbd-c8fa-484f-a1f6-db7ba2ce8917/Screen_Shot_2021-06-20_at_2.17.48_PM.png)
            
            ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ff944ae-7997-4c9d-884c-1692818f9ad2/Screen_Shot_2021-06-20_at_2.18.03_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ff944ae-7997-4c9d-884c-1692818f9ad2/Screen_Shot_2021-06-20_at_2.18.03_PM.png)
            
    
4. Layer 
    - **EfficientNet** + **(GlobalAveragePooling, Dense w/ softmax)**

    ```python
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(n_labels, activation='softmax')
    ```
    
5. Ensemble 

5개 EfficientNet Ensemble 사용

⇒ 5개 sum 후 /5 수행

with strategy.scope():
    
    models = []
    
    models0 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model0.h5'
    )
    models1 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model1.h5'
    )
    models2 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model2.h5'
    )
    models3 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model3.h5'
    )
    models4 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model4.h5'
    )
    
    models.append(models0)
    models.append(models1)
    models.append(models2)
    models.append(models3)
    models.append(models4)

# 5개 모델 probs sum 후 평균으로 나눔 
sub_df[label_cols] = sum([model.predict(dtest, verbose=1) for model in models]) / len(models)
6. Metric
    - AUC
        - [https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229637](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229637)
        - AUC 사용해서 bbox confidence score 계산
    - mAP 설명
        - [https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/212287](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/212287)
7. Models
    - Models
        - Detectron2 (by facebook)
            - [https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
            - [https://www.kaggle.com/corochann/vinbigdata-detectron2-prediction](https://www.kaggle.com/corochann/vinbigdata-detectron2-prediction)
        - Yolo_v5
            - [https://www.kaggle.com/h053473666/siim-cov19-yolov5-train](https://www.kaggle.com/h053473666/siim-cov19-yolov5-train)
        - EfficientDet
            - [https://www.kaggle.com/shonenkov/training-efficientdet](https://www.kaggle.com/shonenkov/training-efficientdet)
        - MMDetection
            - [https://www.kaggle.com/sreevishnudamodaran/siim-mmdetection-cascadercnn-weight-bias](https://www.kaggle.com/sreevishnudamodaran/siim-mmdetection-cascadercnn-weight-bias)
    - Architectures could be used
        - Retinanet (RsNet50, ResNet101, ResNet152)
        - Faster RCNN
        - EfficientNet
        - CenterNet (HourGlass)
    - About 2class model
        - [https://www.kaggle.com/corochann/vinbigdata-2-class-classifier-complete-pipeline](https://www.kaggle.com/corochann/vinbigdata-2-class-classifier-complete-pipeline)
        - 2class model 사용 이유 - 정상과 비정상을 가리는 것 자체가 모델 성능을 늘려줌
    - About yolov14 model
        - [https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-infer](https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-infer)
8. Image preprocssings
    - Findings for this competition
        - images are monochrome, and the intensity scale is different, possibly non-linear
        - abnormalities don’t have clear edges (unlike ribs!!!) so detection is different
        - an opacity is a shape defined by fuzzy changes within a narrow intensity band
        - unlike visible light x-rays penetrate depth-wise, there are no clear cuts where opaque objects occlude the deeper ones
    
9. BBox Ensemble Method
    - WBF
        - [https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset](https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset)
        - [https://www.kaggle.com/shonenkov/bayesian-optimization-wbf-efficientdet](https://www.kaggle.com/shonenkov/bayesian-optimization-wbf-efficientdet)
        - 

10. Test Time Augmentation

- 하나의 모델을 갖고 원본 이미지와 수직 대칭 이미지에 대해서 예측을 추가로 수행 ([https://lv99.tistory.com/74](https://lv99.tistory.com/74))


       
        

[VinBigdata 1등 discussion ([https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229724](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229724))](https://www.notion.so/VinBigdata-1-discussion-https-www-kaggle-com-c-vinbigdata-chest-xray-abnormalities-detection-dis-9d88d284fe4c45dc8ed91113ba78ee78)
