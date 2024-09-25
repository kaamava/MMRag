import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flickr8k
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForVision2Seq, CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image

# Step 1: 定义数据集的路径
image_folder = "Flickr8k/images/"  # Flickr8k 图像文件夹
caption_file = "Flickr8k/captions.txt"  # Flickr8k caption 文件

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

flickr8k_dataset = Flickr8k(root=image_folder, ann_file=caption_file, transform=transform)
flickr8k_dataloader = DataLoader(flickr8k_dataset, batch_size=1, shuffle=False, num_workers=2)

# Step 3: 构建候选集
candidate_examples = {}
for idx, (image, captions) in enumerate(flickr8k_dataloader):
    # 获取每个图像及其对应的所有 captions
    image_path = flickr8k_dataset.ids[idx]
    captions_list = [caption for caption in captions]
    candidate_examples[image_path] = captions_list

# Step 4: 使用 CLIP 模型初始化
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Step 5: 使用 CLIP 模型计算嵌入，并确保文本不会超长
def compute_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    return image_embedding.cpu().numpy()


def compute_text_embedding(text):
    # 添加 truncation=True 以确保文本自动截断为 77 个 token
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding.cpu().numpy()


def get_combined_caption(captions):
    return " ".join(captions)


# 使用拼接的 caption 计算文本嵌入
image_embeddings = np.vstack([compute_image_embedding(image) for image, _ in flickr8k_dataset])
text_embeddings = np.vstack(
    [compute_text_embedding(get_combined_caption(captions)) for _, captions in flickr8k_dataset])


# Step 6: 使用 FAISS 构建索引
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# 构建图像和文本的 FAISS 索引
image_index = build_faiss_index(image_embeddings)
text_index = build_faiss_index(text_embeddings)

# Step 7: 使用测试样本进行检索
test_image_path = "path/to/Flickr8k/test_image.jpg"
test_question = "What is in the picture?"
test_answer = "A cat is sitting on the couch."  # 假设这是测试样本的正确答案


def search_with_clip(vqa_image_path, vqa_question, image_index, text_index, k=3):
    test_image = Image.open(vqa_image_path)
    image_embedding = compute_image_embedding(test_image)
    text_embedding = compute_text_embedding(vqa_question)

    # 图像到文本检索
    D_text, I_text = text_index.search(image_embedding, k)
    text_results = [(i, d) for i, d in zip(I_text[0], D_text[0])]

    # 文本到图像检索
    D_image, I_image = image_index.search(text_embedding, k)
    image_results = [(i, d) for i, d in zip(I_image[0], D_image[0])]

    # 返回两个检索结果的交集（取出最相关的图文对），限制为 k 个结果
    result_ids = list(set([i for i, _ in text_results] + [i for i, _ in image_results]))
    return [list(candidate_examples.items())[i] for i in result_ids[:k]]  # 只返回 k 个示例


# 使用 FAISS 检索最匹配的三个图文对
top_3_candidates = search_with_clip(test_image_path, test_question, image_index, text_index, k=3)

internvl_model = AutoModelForVision2Seq.from_pretrained("OpenGVLab/InternVL2-8B")
internvl_processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2-8B")

def calculate_joint_likelihood(test_image_path, test_question, test_answer, top_candidates, internvl_model,
                               internvl_processor):
    test_image = Image.open(test_image_path)
    likelihood_scores = {}

    for i, (example_image_path, example_captions) in enumerate(top_candidates):
        example_image = Image.open(example_image_path)
        combined_caption = get_combined_caption(example_captions)

        # 构建新的 prompt
        prompt = (
            f"Please provide a detailed answer based on the image provided. "
            f"When answering, consider all the details visible in the picture and be as specific as possible."
            f"Use the following examples as a guide for the level of detail expected:\n\n"
            f"Example Image: {example_image_path}\n"
            f"Example Question: {test_question}\n"
            f"Example Answer: {combined_caption}\n"
            f"Test image: {test_image_path}\n\n"
            f"Your Turn:\n"
            f"Question: {test_question}\n"
            f"Please follow the example to give the answer to the question.\n"
            f"Answer: {test_answer}\n"
        )

        inputs = internvl_processor(text=[prompt], images=[test_image, example_image], return_tensors="pt")

        with torch.no_grad():
            outputs = internvl_model(**inputs, output_hidden_states=False, return_dict=True)
            logits = outputs.logits  # 获取生成的 logits
            log_probs = torch.log_softmax(logits, dim=-1)  # 计算生成每个 token 的 log 概率

            # 计算联合 log 概率
            correct_answer_ids = internvl_processor.tokenizer.encode(test_answer, return_tensors="pt").to(logits.device)
            joint_likelihood = 0.0

            # 倒序遍历正确答案的 token，匹配倒数第 j 个 token 的 log_probs
            num_tokens = correct_answer_ids.shape[1]
            for j in range(num_tokens - 1):
                token_id = correct_answer_ids[0, j + 1]
                token_log_prob = log_probs[0, -1 - j, token_id].item()
                joint_likelihood += token_log_prob

        likelihood_scores[example_image_path] = joint_likelihood

    sorted_candidates = sorted(likelihood_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates


sorted_results = calculate_joint_likelihood(test_image_path, test_question, test_answer, top_3_candidates,
                                            internvl_model, internvl_processor)

print("Sorted Top-3 Examples by Joint Likelihood Scores:")
for i, (image_path, score) in enumerate(sorted_results):
    print(f"Example {i}: Image File = {image_path}, Joint Likelihood Score = {score}")



