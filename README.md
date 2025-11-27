## PCR-LLM
This is the source code and data for paper "Privacy-preserving Cross-domain Recommendation Enhanced with Large Language Model".

The following steps are to run the model.
1. python all_pq.py --gpu_id 
2. python fed_pretrain
3. python single_train.py --d= --p=......pth
4. python prompt_finetune.py --d= --p=your_pretrained_model.pth

**To run this model, you need to modify the three configuration files—pretrain.yaml, finetune.yaml, and prompt.yaml—along with their associated parameters. You also need to enter the api key in the code(openai or chatglm).**

The document also provides multi-card parallel code, employing a data-parallel strategy.

The datasets used are the Amazon datasets and the Pantry public datasets.

The code references VQRec and Recbolel2.0.