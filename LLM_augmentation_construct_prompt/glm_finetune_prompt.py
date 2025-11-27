import os
from pydantic.v1.schema import model_type_schema
from zhipuai import ZhipuAI
import pickle
import time
import requests
import pickle
from LLM_augmentation_construct_prompt.gpt_finetune_prompt import file_path
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import re
model_type=""
def construct_prompting(item_attribute, item_list, candidate_list):
    # make history string
    history_string = "User history:\n"
    for index in item_list:
        title = item_attribute['title'][index]
        genre = item_attribute['genre'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += title + ", "
        history_string += genre + "\n"
    # make candidates
    candidate_string = "Candidates:\n"
    for index in candidate_list:
        title = item_attribute['title'][index.item()]
        genre = item_attribute['genre'][index.item()]
        candidate_string += "["
        candidate_string += str(index.item())
        candidate_string += "] "
        candidate_string += title + ", "
        candidate_string += genre + "\n"
    # output format
    output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    # make prompt
    prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt
### read candidate
candidate_indices = pickle.load(open(file_path + 'candidate_indices', 'rb'))
candidate_indices_dict = {}
for index in range(candidate_indices.shape[0]):
    candidate_indices_dict[index] = candidate_indices[index]
### read adjacency_list
adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + 'train_mat', 'rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y
### read item_attribute
toy_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id', 'title', 'genre'])
### write augmented dict
augmented_sample_dict = {}
if os.path.exists(file_path + "augmented_sample_dict"):
    print(f"The file augmented_sample_dict exists.")
    augmented_sample_dict = pickle.load(open(file_path + 'augmented_sample_dict', 'rb'))
else:
    print(f"The file augmented_sample_dict does not exist.")
    pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict', 'wb'))


def file_reading():
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_sample_dict', 'rb'))
    return augmented_attribute_dict

def _extract_content(resp: Any) -> str:
    """
    兼容多种返回结构的 content 提取：
    - SDK 对象：resp.choices[0].message.content
    - 字典：resp['choices'][0]['message']['content']
    - 少数情况下（不建议）：resp.json()['choices'][0]['message']['content']
    """

    try:
        return resp.choices[0].message.content
    except Exception:
        pass

    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise KeyError("content（choices[0].message.content）。")


def _parse_pos_neg(text: str) -> Tuple[int, int]:
    m = re.search(r'(-?\d+)\s*::\s*(-?\d+)', text)
    if not m:
        raise ValueError(f"format is wrong 'pos::neg'，actually：{text!r}")
    return int(m.group(1)), int(m.group(2))


def GLM_request(
    prompt: str,
    index: int,
    augmented_sample_dict: Dict[int, Dict[int, int]],
    file_dir: str,
    *,
    model: str = "glm-4-air",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> Tuple[int, int]:

    if api_key is None:
        api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("loss API Key")

    client = ZhipuAI(api_key=api_key)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = _extract_content(resp)
            pos_sample, neg_sample = _parse_pos_neg(content)

            if index not in augmented_sample_dict:
                augmented_sample_dict[index] = {}
            augmented_sample_dict[index][0] = pos_sample
            augmented_sample_dict[index][1] = neg_sample

            os.makedirs(file_dir, exist_ok=True)
            out_path = os.path.join(file_dir, "augmented_sample_dict.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(augmented_sample_dict, f)

            print(f"[OK] index={index} => {pos_sample}::{neg_sample}")
            return pos_sample, neg_sample

        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            last_err = e
            print(f"[WARN] 第 {attempt}/{max_retries} 次尝试失败：{e}")
        except Exception as e:
            last_err = e
            print(f"[WARN] 第 {attempt}/{max_retries} 次未知错误：{e}")

        if attempt < max_retries:
            time.sleep(retry_backoff_sec * attempt)

    raise RuntimeError(f"GLM_request 调用失败：{last_err}")

if __name__ == "__main__":
    prompt = construct_prompting("")
    GLM_request(prompt)