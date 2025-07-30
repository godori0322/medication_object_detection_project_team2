# CapsuleNet(2팀)

## 협업일지 링크
- [고인범](https://www.notion.so/_-23fbc32ff87180ddb1c0ea8614c7bbb9?source=copy_link)

- [김도영]() 

- [김재용](https://www.notion.so/2314e8731dd980a8984ed33e4e5faa1f)

- [김지영](https://www.notion.so/240cf974f5f580bd86f8df0939717058?v=240cf974f5f581318eeb000cf99f19d7)

- [임준혁](https://www.notion.so/_-2314f145016780d48776f603f821d241?source=copy_link)


## AI 3기 초급 프로젝트
<경구약제 이미지 객체 검출 프로젝트>

본 GitHub Repository는 CapsuleNet 팀이 초급 프로젝트를 진행한 과정을 포함합니다. 
주요 코드는 Project 디렉토리에 있으며, 원본 데이터는 data 디렉토리에서 별도로 관리합니다. 

## 프로젝트 세팅
1. 제공된 학습 데이터를 [다운](https://www.kaggle.com/competitions/ai03-level1-project/data)받아 `data/ai03-level1-project` 폴더 하위에 압축 풀기
    ```
    완성 경로:
    data/ai03-level1-project/train_images/*.png
    data/ai03-level1-project/test_images/*.png
    ```
2. YOLO 학습 전처리 데이터로 변환
    - python
        ```
        python Project/src/utils/yolo_image_setting.py
        ```
    - poetry
        ```
        poetry run python Project/src/utils/yolo_image_setting.py
        ```
3. 모델 학습
    - python
        ```
        python Project/main.py --hyp_path \
        Project/hyperparameter/yolov11/last_hyperparameters.yaml
        ```
    - poetry
        ```
        poetry run python Project/main.py --hyp_path \
        Project/hyperparameter/yolov11/last_hyperparameters.yaml
        ```

    #### `Project/outputs` 하위경로에 학습 모델이 생성 됩니다.