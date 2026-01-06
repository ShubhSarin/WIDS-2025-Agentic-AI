from transformers import pipeline

original_text = """Machine learning is the art of teaching computers to learn from data instead of yelling instructions at them line by line (which, sadly, does not work). At its core, it's about building models that spot patterns, make predictions, and improve with experience—whether that's recommending your next binge-watch, flagging spam emails, or helping doctors spot diseases earlier. What makes machine learning powerful (and slightly magical) is its ability to generalize: it doesn't just memorize answers, it learns rules from examples. Of course, it's not actual intelligence—it's more like a very fast, very literal intern who's amazing with numbers and terrible without data—but when guided well, it can solve problems at a scale and speed humans simply can't."""

summarise = pipeline(task = 'summarization', model = 'facebook/bart-large-cnn')
summary = summarise(original_text, max_length = 100, min_length = 50)


print(f"Length of original text in characters = {len(original_text)}")
print(summary[0]['summary_text'])