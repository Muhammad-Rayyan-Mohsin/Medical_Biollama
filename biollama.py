import transformers
import torch

model_id = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Initialize conversation with system message
messages = [
    {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
]

print("BioMedical Chatbot initialized. Type 'exit' or 'quit' to end the conversation.")

while True:
    # Get user input
    user_question = input("\nEnter your medical question (or 'exit' to quit): ")
    
    # Check if user wants to exit
    if user_question.lower() in ['exit', 'quit']:
        print("Exiting chatbot. Goodbye!")
        break
    
    # Add user message to conversation history
    messages.append({"role": "user", "content": user_question})
    
    # Generate response
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract the response
    response = outputs[0]["generated_text"][len(prompt):]
    print("\nChatbot:", response)
    
    # Add assistant's response to conversation history
    messages.append({"role": "assistant", "content": response})
