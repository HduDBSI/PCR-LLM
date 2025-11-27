from zhipuai import ZhipuAI


def construct_prompting(item_attribute, item_list):
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
    # output format
    output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    # make prompt
    prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    prompt += history_string
    prompt += output_format
    return prompt

def GLMrequest(prompt):
    client = ZhipuAI(api_key="627bf62970c3a23f2e5bcfac178ba0c9.qbmdB3XZUZPa7hTb") # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-air",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    #print(response.choices[0].message.content)
    return(response.choices[0].message.content)

if __name__ == "__main__":
    prompt = construct_prompting("She's beautiful, seductive, intelligent...and her charms could spell doom for the entire human race! Ben Kingsley, Forrest Whitaker and Natasha Henstridge star in this adrenaline-charged sci-fi hit.")
    GLMrequest(prompt)