from zhipuai import ZhipuAI

def construct_prompting(item_attribute):
    # pre string
    pre_string = "You are a semantic expert. You need to understand the following text and remove the domain-specific words, replacing them with more general terms.\n"
    # make item list
    item_list_string=item_attribute
    # item_list_string = ""
    # for index in indices:
    #     year = item_attribute['year'][index]
    #     title = item_attribute['title'][index]
    #     item_list_string += "["
    #     item_list_string += str(index)
    #     item_list_string += "] "
    #     item_list_string += str(year) + ", "
    #     item_list_string += title + "\n"
    # output format
    # output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    output_format="The output is in English and no longer than 512 tokens."
    # make prompt
    prompt = pre_string + item_list_string + output_format
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