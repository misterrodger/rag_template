import json
import numpy as np
from src.chunker import chunk_file
from src.embedder import create_embeddings
from src.vector_store import create_index, save_index, load_index
from src.rag import query_rag


def build_knowledge_base():
    print("Building knowledge base...\n")

    chunks = []
    embeddings_list = []

    for filename in ["climate_change", "ai_overview"]:
        filepath = f"data/{filename}.txt"
        print(f"Processing {filepath}...")

        doc_chunks = chunk_file(filepath)
        doc_embeddings = create_embeddings(doc_chunks)

        chunks.extend(doc_chunks)
        embeddings_list.extend(doc_embeddings.tolist())

        with open(f"storage/{filename}_chunks.json", "w") as f:
            json.dump(doc_chunks, f)

        with open(f"storage/{filename}_embeddings.json", "w") as f:
            json.dump(doc_embeddings.tolist(), f)

        print(f"  Created {len(doc_chunks)} chunks\n")

    embeddings = np.array(embeddings_list)
    index = create_index(embeddings)
    save_index(index, chunks)

    print(f"Knowledge base complete: {len(chunks)} total chunks indexed\n")

    return index, chunks


def main():
    try:
        loaded_index, loaded_chunks = load_index()
        print(f"Loaded existing index with {len(loaded_chunks)} chunks\n")
    except:
        loaded_index, loaded_chunks = build_knowledge_base()

    queries = [
        "What is climate change?",
        "What is artificial intelligence?",
        "Tell me about greenhouse gases and their effect on Earth",
        "How does machine learning work?"
    ]

    for query in queries:
        print(f"{'='*70}")
        print(f"Q: {query}")
        print(f"{'='*70}")

        result = query_rag(query, loaded_index, loaded_chunks, k=3)

        print(f"\nA: {result['answer']}")

        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n  [{i}] Distance: {source['distance']:.4f}")
            print(f"      {source['chunk']}")

        print("\n")


if __name__ == "__main__":
    main()


# Loaded existing index with 14 chunks

# ======================================================================
# Q: What is climate change?
# ======================================================================

# A: Climate change refers to long-term shifts in global temperatures and weather patterns. While natural variations have occurred throughout Earth's history, current changes are primarily driven by human activities, especially the burning of fossil fuels, which releases greenhouse gases into the atmosphere. The scientific consensus confirms that human activity is the dominant cause of the observed warming since the mid-20th century.

# Sources:

#   [1] Distance: 0.7437
#       Climate Change: Understanding Our Changing Planet

# What is Climate Change?

# Climate change refers to...

#   [2] Distance: 1.0157
#       Impact on Ecosystems and Biodiversity

# Rising temperatures and changing precipitation patterns are a...

#   [3] Distance: 1.0712
#       Melting Ice and Rising Seas

# Arctic sea ice is declining at a rate of about 13% per decade, and glac...


# ======================================================================
# Q: What is artificial intelligence?
# ======================================================================

# A: Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.

# Sources:

#   [1] Distance: 0.7452
#       Artificial Intelligence: An Overview

# Introduction to AI

# Artificial intelligence (AI) refers to the...

#   [2] Distance: 0.9916
#       The Future of AI

# The future of AI holds tremendous potential and uncertainty. Researchers are worki...

#   [3] Distance: 1.0787
#       Robotics and Automation

# Robotics combines AI with mechanical engineering to create machines capable...


# ======================================================================
# Q: Tell me about greenhouse gases and their effect on Earth
# ======================================================================

# A: Greenhouse gases are certain gases in Earth's atmosphere that trap heat from the sun, contributing to the greenhouse effect, which is a natural process that keeps our planet warm enough to sustain life. Key greenhouse gases include carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). 

# Human activities, such as the burning of fossil fuels and deforestation, have intensified the greenhouse effect by significantly increasing the concentrations of these gases. For instance, carbon dioxide levels have risen by over 50% since pre-industrial times. This increase in greenhouse gas concentrations has led to a rise in global average temperatures, which have increased by approximately 1.1 degrees Celsius since the late 19th century.

# The consequences of this temperature rise are profound, as even modest increases can have significant impacts on Earth's climate system. Without substantial action, temperatures could rise by 2-4 degrees Celsius by the end of the century, leading to severe and potentially irreversible consequences, including melting ice, rising sea levels, and increased extreme weather events.

# Sources:

#   [1] Distance: 0.7215
#       The Greenhouse Effect

# The greenhouse effect is a natural process where certain gases in Earth's atm...

#   [2] Distance: 0.9603
#       Climate Change: Understanding Our Changing Planet

# What is Climate Change?

# Climate change refers to...

#   [3] Distance: 1.1656
#       Melting Ice and Rising Seas

# Arctic sea ice is declining at a rate of about 13% per decade, and glac...


# =====
# 2. **Model Training**: In supervised learning, the model is trained using labeled data, where the input data is paired with the correct output. The model learns to map inputs to outputs by finding patterns in the training data. In unsupervised learning, the model explores unlabeled data to identify patterns or groupings without predefined labels.

# 3. **Learning Process**: The model uses algorithms to adjust its parameters based on the data it processes. This involves techniques such as optimization to minimize errors in predictions.

# 4. **Evaluation**: After training, the model is evaluated on a separate set of data (validation or test data) to assess its performance and generalization ability.

# 5. **Iteration**: The training process may be repeated with adjustments to improve accuracy, involving tuning hyperparameters or using different algorithms.

# 6. **Deployment**: Once the model is trained and validated, it can be deployed for practical applications, where it can make predictions or decisions based on new input data.

# Overall, machine learning allows systems to improve their performance on specific tasks over time as they are exposed to more data.

# Sources:

#   [1] Distance: 0.9969
#       Artificial Intelligence: An Overview

# Introduction to AI

# Artificial intelligence (AI) refers to the...

#   [2] Distance: 1.0727
#       Deep Learning Revolution

# Deep learning, a specialized form of machine learning, uses artificial neu...

#   [3] Distance: 1.2198
#       Computer Vision Applications

# Computer vision enables machines to interpret and understand visual in...=================================================================
# Q: How does machine learning work?
# ======================================================================

# A: Machine learning works by enabling computers to learn from data without being explicitly programmed. It involves the following key processes:

# 1. **Data Collection**: Machine learning begins with the collection of data, which can be labeled or unlabeled depending on the type of learning.

# 2. **Model Training**: In supervised learning, the model is trained using labeled data, where the input data is paired with the correct output. The model learns to map inputs to outputs by finding patterns in the training data. In unsupervised learning, the model explores unlabeled data to identify patterns or groupings without predefined labels.

# 3. **Learning Process**: The model uses algorithms to adjust its parameters based on the data it processes. This involves techniques such as optimization to minimize errors in predictions.

# 4. **Evaluation**: After training, the model is evaluated on a separate set of data (validation or test data) to assess its performance and generalization ability.

# 5. **Iteration**: The training process may be repeated with adjustments to improve accuracy, involving tuning hyperparameters or using different algorithms.

# 6. **Deployment**: Once the model is trained and validated, it can be deployed for practical applications, where it can make predictions or decisions based on new input data.

# Overall, machine learning allows systems to improve their performance on specific tasks over time as they are exposed to more data.

# Sources:

#   [1] Distance: 0.9969
#       Artificial Intelligence: An Overview

# Introduction to AI

# Artificial intelligence (AI) refers to the...

#   [2] Distance: 1.0727
#       Deep Learning Revolution

# Deep learning, a specialized form of machine learning, uses artificial neu...

#   [3] Distance: 1.2198
#       Computer Vision Applications

# Computer vision enables machines to interpret and understand visual in...