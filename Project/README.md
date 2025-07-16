상일님이 올려주신 코드 기반으로 어떤 형식으로 모듈화 하면 좋을지 구상한 Project입니다. -> 기능 별로 응집도 최대화, 결합도 최소화 하여 개발 및 협업 용이하게!


# 현재 구현된 부분


데이터 로딩 및 전처리 담당하는 data 패키지에는 raw 데이터 및 preprocessing 된 데이터만 포함, .gitignore를 통해 repository로 직접 데이터는 push, pull하지 않도록 했습니다. 

src(source) 패키지에 dataset.py, dataloader.py 포함해 raw 데이터를 pre-processing, transform 적용하여 data augmentation 진행 후 (image, label) return -> model 패키지에서 호출해 사용

src(source) 패키지에 모델 관련된 model, train, evaluate 포함하여 모델 학습 및 성능 평가, config 통해 하이퍼 파라미터 관리, visualize 통해 시각화


# 추가로 구현 필요한 부분?

notebooks와 같이 별도의 폴더에서 데이터 EDA, 모델 간 비교 등의 실험 결과를 Colab, Jupyter Notebook 등으로 저장해 주세요. (결과 추적 원하지 않을 경우 .gitignore 사용)


experiments와 같이 별도의 폴더에서 각 모델에 대한 테스트 결과(model.pt 모델 state dict, results.csv 결과, 그외 README.md 등으로 주석이나 기재 사항)를 저장해 주세요. 


docs, result와 같이 별도의 폴더에서 성능 및 시각화 결과 정리해서 주세요. (이 부분은 꼭 repository가 아닌, 노션이나 기타 문서에서 진행하셔도 됩니다.)


utils에는 src 내부에서 사용하는 function 모아놓았고(클래스 id 와 약재 이름을 mapping하는 class_mapping.py), 추가로 logger.py를 통해 train 과정에서 기록을 일단은 experiments에 저장하도록 했습니다. (저장 위치 추후 조정 가능, logger 구현 여부?)



현재 src/models/ 내부에 각 모델을 생성하고, 필요한 model을 call하는 방식으로 진행하고자 합니다. 혹시 다른 의견 있으신 분 말씀해 주세요. 