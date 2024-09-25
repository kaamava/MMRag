import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")

test_image_path = "path/to/test_image.jpg"
test_question = "What is in the picture?"

retrieved_examples = []

def calculate_joint_likelihood(test_image_path, test_question, test_answer, retrieved_examples, blip_model, blip_processor):
    test_image = Image.open(test_image_path)
    likelihood_scores = {}

    for i, (example_image_path, example_text) in enumerate(retrieved_examples):
        example_image = Image.open(example_image_path)

        prompt = (
            f"Please provide a detailed answer based on the image provided. "
            f"When answering, consider all the details visible in the picture and be as specific as possible."
            f"Use the following examples as a guide for the level of detail expected:\n\n"
            f"Example Image: {example_image}\n"
            f"Example Question:{test_question}\n"
            f"Example Answer:{example_text}\n"
            f"Test image: {test_image}\n\n"
            f"Your Turn:\n"
            f"Question: {test_question}\n"
            f"Please follow the example to give the answer to the question.\n:"
            f"Answer: {test_answer}\n"
        )


        inputs = blip_processor(text=[prompt], images=[test_image, example_image], return_tensors="pt")

        with torch.no_grad():
            outputs = blip_model(**inputs, output_hidden_states=False, return_dict=True)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            gold_ids = blip_processor.tokenizer.encode(test_answer, return_tensors="pt").to(logits.device)

            # 计算生成正确答案的联合似然概率
            joint_likelihood = 0.0
            for j in range(gold_ids.shape[1] - 1):
                token_id = gold_ids[0, j + 1]
                token_log_prob = log_probs[0, -j, token_id].item()
                joint_likelihood += token_log_prob

        likelihood_scores[i] = joint_likelihood

    sorted_examples = sorted(likelihood_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_examples


sorted_results = calculate_joint_likelihood(test_image_path, test_question, retrieved_examples, blip_model,
                                            blip_processor)

print("Sorted Examples:")
for idx, score in sorted_results:
    print(f"Example {idx}: Likelihood Score = {score}")
