import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flickr8k
from torchvision import transforms
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image

#加载数据集
image_folder = "/Flickr8k/images/"
caption_file = "/Flickr8k/captions.txt"

#加载模型和对应的processor
path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

#处理数据集中的图像以及拼接caption
def load_flickr8k_data(image_folder, caption_file):
    candidate_examples = {}

    with open(caption_file, 'r') as f:
        for line in f:
            image_name, caption = line.strip().split(",", 1)
            image_path = os.path.join(image_folder, image_name)

            if image_path not in candidate_examples:
                candidate_examples[image_path] = []
            candidate_examples[image_path].append(caption)

    return candidate_examples

#获取所有示例
candidate_examples = load_flickr8k_data(image_folder, caption_file)

#加载clip模型用于将数据embedding化
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#将图像embedding化
def compute_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    return image_embedding.cpu().numpy()

#将文字embedding化
def compute_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding.cpu().numpy()

#储存embedding
image_embeddings = []
text_embeddings = []
image_paths = []

for image_path, captions in candidate_examples.items():
    image_embedding = compute_image_embedding(image_path)
    combined_caption = " ".join(captions)
    text_embedding = compute_text_embedding(combined_caption)

    image_embeddings.append(image_embedding)
    text_embeddings.append(text_embedding)
    image_paths.append(image_path)

#将图像和文字embedding numpy化
image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)

#将获取的embedding存入faiss并进行索引
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

image_index = build_faiss_index(image_embeddings)
text_index = build_faiss_index(text_embeddings)

#利用clip进行双向的相似度召回，得到k个示例
def search_with_clip(test_image_path, test_caption, image_index, text_index, k):
    test_image = Image.open(test_image_path)
    image_embedding = compute_image_embedding(test_image)
    text_embedding = compute_text_embedding(test_caption)

    D_text, I_text = text_index.search(image_embedding, k)
    text_results = [(i, d) for i, d in zip(I_text[0], D_text[0])]

    D_image, I_image = image_index.search(text_embedding, k)
    image_results = [(i, d) for i, d in zip(I_image[0], D_image[0])]

    result_ids = list(set([i for i, _ in text_results] + [i for i, _ in image_results]))
    return [list(candidate_examples.items())[i] for i in result_ids[:k]]

#计算联合似然函数P(answer|prompt)，用于评价召回的示例对大模型推理能力的帮助
def calculate_joint_likelihood(test_image_path, test_question, test_answer, top_candidates, internvl_model,
                               internvl_processor):
    test_image = Image.open(test_image_path)
    likelihood_scores = {}

    for i, (example_image_path, example_captions) in enumerate(top_candidates):
        example_image = Image.open(example_image_path)
        combined_caption = example_captions
        prompt = (
            f"Please provide a detailed answer based on the image provided. "
            f"When answering, consider all the details visible in the picture and be as specific as possible. "
            f"Use the following examples as a guide for the level of detail expected:\n\n"
            f"Example Image: {example_image_path}\n"
            f"Example Question: {test_question}\n"
            f"Example Answer: {combined_caption}\n"
            f"Test image: {test_image_path}\n\n"
            f"Your Turn:\n"
            f"Question: {test_question}\n"
            f"Please follow the example to give the answer to the question.\n"
            f"Answer: {test_answer}\n"
        )#设计prompt
        inputs = internvl_processor(text=[prompt], images=[test_image, example_image], return_tensors="pt")

        with torch.no_grad():
            outputs = internvl_model(**inputs)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            correct_answer_ids = internvl_processor.tokenizer.encode(test_answer, return_tensors="pt").to(logits.device)
            joint_likelihood = 0.0
            num_tokens = correct_answer_ids.shape[1]
            #计算联合似然得分
            for j in range(num_tokens - 1):
                token_id = correct_answer_ids[0, j + 1]
                token_log_prob = log_probs[0, -1 - j, token_id].item()
                joint_likelihood += token_log_prob
        likelihood_scores[example_image_path] = joint_likelihood

    #根据似然得分进行排序
    sorted_candidates = sorted(likelihood_scores.items(), key=lambda x: x[1], reverse=False)
    return sorted_candidates

#用排序完后的n个示例重新构建prompt用于大模型推理
def build_detailed_prompt(selected_n_results, test_question, test_answer, test_image_path):
    context = "\n".join([
        f"example{i + 1}:\n"
        f"Example Image: {img_path}\n"
        f"Example Question: {test_question}\n"
        f"Example Answer: {caption}"
        for i, (img_path, caption) in enumerate(selected_n_results)
    ])
    prompt = (
        f"Please provide a detailed answer based on the image provided. "
        f"When answering, consider all the details visible in the picture and be as specific as possible. "
        f"Use the following examples as a guide for the level of detail expected:\n\n"
        f"{context}\n\n"
        f"Test Image: {test_image_path}\n\n"
        f"Your Turn:\n"
        f"Question: {test_question}\n"
        f"Please follow the example to give the answer to the question.\n"
        f"Answer: {test_answer}\n"
    )
    return prompt

#测试的image和qa对
test_image_path = "/test_image.jpg"
test_question = "What is the cat doing?"
test_answer = "The cat is playing on the green grass with a yellow and black butterfly."

n = int(input('How many examples to use? '))

#从2n个示例中根据相似度召回前n个，再利用联合似然得分进行排序，构建prompt
top_2n_candidates = search_with_clip(test_image_path, test_question, image_index, text_index, 2*n)

sorted_2n_results = calculate_joint_likelihood(test_image_path, test_question, test_answer, top_2n_candidates,
                                               model, processor)

selected_n_results = sorted_2n_results[:n]
new_prompt = build_detailed_prompt(selected_n_results, test_question, test_answer, test_image_path)


