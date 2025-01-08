from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preset and intended answers
preset_answer = "I'm feeling fine right now, thanks" 
intended_answer = "I'm feeling fine right now"

# Encode sentences to get embeddings
preset_embedding = model.encode(preset_answer, convert_to_tensor=True)
intended_embedding = model.encode(intended_answer, convert_to_tensor=True)

# Compute cosine similarity
cosine_similarity = util.cos_sim(preset_embedding, intended_embedding).item()

# Convert to percentage
percentage_match = cosine_similarity * 100
print(f"Percentage Match: {percentage_match:.2f}%")
