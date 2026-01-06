from transformers import pipeline

text_gen = pipeline(task = 'text-generation', model="openai-community/gpt2")

text = "In 2030, AI systems will"

text_generated = text_gen(text_inputs=text, num_return_sequences = 2, max_new_tokens = 50)

print('=================================================')
print('Generated Texts:')
print('-------------------------------------------------')
print(text_generated[0]['generated_text'])
print('-------------------------------------------------')
print(text_generated[1]['generated_text'])