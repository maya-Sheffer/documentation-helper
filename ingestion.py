import os
import asyncio
import ssl
import certifi
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_info, log_success, log_error, log_warning, log_header

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 10) -> int:
    """Index small batches sequentially to limit load on the local Ollama model."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if not documents:
        raise ValueError("No documents to index")

    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    indexed_documents = 0
    for batch_num, batch in enumerate(batches, start=1):
        try:
            await asyncio.wait_for(
                vectorstore.aadd_documents(
                    batch,
                    embedding_chunk_size=batch_size,
                    batch_size=batch_size,
                ),
                timeout=180,
            )
        except Exception as e:
            reason = "request timed out after 180 seconds" if isinstance(e, TimeoutError) else str(e)
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {reason}")
            raise RuntimeError(
                f"Indexing stopped after {indexed_documents}/{len(documents)} documents; "
                "check the Ollama service and the error above before running again"
            ) from e

        indexed_documents += len(batch)
        log_success(
            f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} "
            f"({indexed_documents}/{len(documents)} documents indexed)"
        )

    log_success(
        f"VectorStore Indexing: All batches processed successfully! ({len(batches)}/{len(batches)})"
    )
    return indexed_documents


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("Checking the local Ollama embedding model before crawling")
    try:
        await asyncio.wait_for(
            embeddings.aembed_query("Embedding service health check."), timeout=60
        )
    except Exception as e:
        raise RuntimeError(
            "Ollama embedding check failed. Open Ollama and confirm that "
            "nomic-embed-text is available before running the pipeline"
        ) from e
    log_success("Ollama embedding check passed")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        raw_content = tavily_crawl_result_item.get("raw_content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            log_warning(
                f"TavilyCrawl: Skipping {tavily_crawl_result_item['url']} because no text content was returned"
            )
            continue

        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=raw_content,
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    if not all_docs:
        raise RuntimeError("TavilyCrawl returned no pages with text content; nothing can be indexed")

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    indexed_documents = await index_documents_async(splitted_docs, batch_size=10)

    log_header("PIPELINE COMPLETE")
    log_info(f"Documents indexed in Pinecone: {indexed_documents}")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

