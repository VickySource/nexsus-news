from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader=TextLoader("learning\day2\Promting\Chain-of_thought.txt")
docs=loader.load()
print(f"Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(
chunk_size=500,
chunk_overlap=50
)
chunks= splitter.split_documents(docs)
print("Split into (len(chunks)) chunks")
for i, chunk in enumerate (chunks[:2]):
    print("Chunk (1): (chunk.page_content[:100])...")