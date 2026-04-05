# Category-Lineage-Correction

Finetuning Based Model:
It creates some training pairs based on a coorectly classified dataset and finetunes an embedding transformer. We then use this finetuned transformer to create better embeddings.

Signal Based Model:
It creates some signals based on a correctly classified dataset and a machine learning model is trained on embeddings and these signals.

Hybrid Model:
Uses finetune approach so we can get better embeddings and signal approach so we can get better features. We use finetuned model's embeddings in order to create better signals.

Steps to run the Code:
--> Create all the folders in mentioned in config.py

--> Keep the necessary data files in data folder to avoid error

--> Install all the requirements and run the command given below in terminal 
    cmd: pip install -r requirements.txt 
    
--> Run the following command to train finetuning_based_detection model
    cmd: 
    python finetuning_based_detection\src\train_embedding_model.py;
    python finetuning_based_detection\src\embedding_inference.py;
    python finetuning_based_detection\src\bp_prediction.py;
    python finetuning_based_detection\src\inference.py;

--> Run the following command to train signal_based_detection model
    cmd:
    python signal_based_detection\src\feature.py;
    python signal_based_detection\src\train_meta_model.py;
    python signal_based_detection\src\predict_pipeline.py;
    python signal_based_detection\src\inference.py;

--> Run the following command to train hybrid model
    cmd:
    python hybrid_approach\src\train_embedding_model.py;
    python hybrid_approach\src\embedding_and_centroid.py;
    python hybrid_approach\src\feature.py;
    python hybrid_approach\src\train_meta_model.py;
    python hybrid_approach\src\inference.py;

  
