import json
import logging
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# Configure log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiProcessor:
    def __init__(self, model_name, project):
        self.model = VertexAI(model_name=model_name, project=project)

    def generate_document_summary(self, documents: list, **args):
        chain_type = "map_reduce" if len(documents) > 10 else "stuff"

        chain = load_summarize_chain(
            llm=self.model,
            chain_type=chain_type,
            **args
        )

        return chain.run(documents)

    def count_total_tokens(self, docs: list):
        temp_model = GenerativeModel("gemini-1.0-pro")
        total = 0
        logger.info("Counting total billable characters...")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        return total

    def get_model(self):
        return self.model

class YoutubeProcessor:
    # Retrieve the full transcript

    def __init__(self, genai_processor: GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        self.GeminiProcessor = genai_processor

    def retrieve_youtube_documents(self, video_url: str, verbose=False):
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)

        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            logger.info(f"{author}\n{length}\n{title}\n{total_size}\n{total_billable_characters}")

        return result

    def find_key_concepts(self, documents: list, sample_size: int = 0, verbose=False):
        # iterate through all documents of group size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")

        # Optimize sample size given no input
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose:
                logging.info(f"No sample size specified. Setting number of documents per sample as 5. Sample size: {sample_size}")

        # Find number of documents in each group
        num_docs_per_group = len(documents) // sample_size + (len(documents) % sample_size > 0)

        # Check thresholds for response quality
        if num_docs_per_group > 10:
            raise ValueError("Each group has more than 10 documents and output quality will be degraded significantly. Increase the sample_size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more than 5 documents and output quality is likely to be degraded. Consider increasing the sample size.")

        # Split the document in chunks of size num_docs_per_group
        groups = [documents[i:i + num_docs_per_group] for i in range(0, len(documents), num_docs_per_group)]

        batch_concepts = []
        batch_cost = 0

        logger.info("Finding key concepts...")
        for group in tqdm(groups):
            # Combine content of documents per group
            group_content = ""

            for doc in group:
                group_content += doc.page_content

            # Prompt for finding concepts
            prompt = PromptTemplate(
                template="""
                Find and define key concepts or terms found in the text:
                {text}

                Respond with the key concepts and definitions in the following format:
                Concept1: Definition1
                Concept2: Definition2
                ...
                """,
                input_variables=["text"]
            )

            # Create chain
            chain = prompt | self.GeminiProcessor.model

            # Run chain
            output_concept = chain.invoke({"text": group_content})

            # Log the raw output for debugging
            if verbose:
                logger.info(f"Raw output from chain: {output_concept}")

            # Convert string output to dictionary format
            concept_dict = {}
            try:
                for line in output_concept.strip().split("\n"):
                    if ": " in line:
                        concept, definition = line.split(": ", 1)
                        concept_dict[concept.strip()] = definition.strip()
                if concept_dict:
                    batch_concepts.append(concept_dict)
            except Exception as e:
                logger.error(f"Error processing output: {e}")
                logger.error(f"Output that caused error: {output_concept}")

            # Post Processing Observation
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char / 1000) * 0.000125

                logging.info(f"Running chain on {len(group)} documents")
                logging.info(f"Total input characters: {total_input_char}")
                logging.info(f"Total cost: {total_input_cost}")

                total_output_char = len(output_concept)
                total_output_cost = (total_output_char / 1000) * 0.000375

                logging.info(f"Total output characters: {total_output_char}")
                logging.info(f"Total cost: {total_output_cost}")

                batch_cost += total_input_cost + total_output_cost
                logging.info(f"Total group cost: {total_input_cost + total_output_cost}\n")

        logging.info(f"Total Analysis Cost: ${batch_cost}")
        return {"key_concepts": batch_concepts}
