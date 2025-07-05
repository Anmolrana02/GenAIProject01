import os

# Define base path
base_dir = "src"

# Define folder structure and files with their starter content
structure = {
    "embeddings": {
        "generate_embeddings.py": '''def generate():
    print("Generating embeddings...")''',
        "embedding_model.py": '''def get_embedding_model():
    print("Loading embedding model...")
    return None'''
    },
    "metadata": {
        "extract_metadata.py": '''def extract_metadata():
    print("Extracting metadata...")'''
    },
    "queries": {
        "query_interface.py": '''def get_query():
    print("Fetching user query...")
    return "Sample query"'''
    },
    "prompts": {
        "few_shot_prompt.py": '''def build_prompt(results):
    print("Building few-shot prompt...")
    return "Prompt with examples"''',
        "prompt_templates.json": '''{
  "default": "Provide a response based on the following examples: {examples}"
}'''
    },
    "vector_db": {
        "vector_store.py": '''def search_similar(query):
    print("Searching vector DB for similar queries...")
    return ["example1", "example2"]'''
    },
    "llm_app": {
        "llm_interface.py": '''def query_llm(prompt):
    print("Sending prompt to LLM...")
    return "LLM response"'''
    },
    "utils": {
        "logger.py": '''def log(message):
    print(f"[LOG]: {message}")''',
        "preprocessing.py": '''def preprocess(text):
    print("Preprocessing text...")
    return text.lower()'''
    },
    "config": {
        "config.yaml": '''# Configuration file for GenAIProject01
embedding_model: "openai"
vector_db: "faiss"
llm_provider: "openai"'''
    }
}

# Create folders and files
for folder, files in structure.items():
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    for file_name, content in files.items():
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as f:
            f.write(content)

print("✅ src/ folder structure created with all boilerplate files.")
