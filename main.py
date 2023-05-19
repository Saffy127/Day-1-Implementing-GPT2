API_Token = "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"

from huggingface_hub.inference_api import InferenceApi
import json

try:
  # Initialize the Hugging Face Inference API
  inference = InferenceApi(repo_id="gpt2-large", token=API_Token)

  # defining our parameters
  top_p_params = {
    "max_length": 128,
    "top_p": 0.95,
    "do_sample": True
  }  # top-p (nucleus sampling)
  top_k_params = {"max_length": 128, "top_k": 4, "do_sample": True}  # top-k
  contrastive_params = {"max_length": 128, "penalty_alpha": 0.6, "top_k": 4}

  input_string = "Learn about AGI because"

  # Perform inference with the default parameters
  result = inference(input_string)
  print(result)

  # Perform inference with the top_p_params
  result = inference(input_string, top_p_params)
  print("Top P: {}".format(result))

  # Perform inference with the top_k_params
  result = inference(input_string, top_k_params)
  print("Top K: {}".format(result))

  # Perform inference with the contrastive_params
  result = inference(input_string, contrastive_params)
  print("Contrastive: {}".format(result))

  # Save results to a file
  with open('results.txt', 'w') as f:
    f.write(json.dumps(result))

except Exception as e:
  print(f"An error occurred: {e}")
