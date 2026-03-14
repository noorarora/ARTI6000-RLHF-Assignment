# ARTI6000 – Advanced Topics in AI and ML  
## Assignment 1 – RLHF Implementation

Student: Noor Arora  
Student id: A1963789
University: University of Adelaide  


## Components

1. Supervised Fine-Tuning (SFT)
   - DistilGPT2 fine-tuned on instruction data.

2. Reward Model
   - DistilBERT trained to score preferred responses.

3. Direct Preference Optimization (DPO)
   - Aligns the SFT model using preference pairs.

4. Evaluation
   - Compares SFT and DPO responses using the reward model.

## Files

01_sft_baseline.ipynb
02_reward_model.ipynb
03_rl_training.ipynb
04_evaluation.ipynb
evaluation_results.csv
