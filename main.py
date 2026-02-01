import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from src.document_processor import PolicyProcessor
from src.vector_storage import VectorManager
from src.agent_core import MedicalPolicyAgent
from src.report_generator import ReportTool

# Load environment variables
load_dotenv()

# Configure logging to both file and console
log_filename = f"logs/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Step 1: Document ingestion and chunking
    logger.info("Step 1: Starting PDF document processing...")
    processor = PolicyProcessor(data_path="data")
    chunks = processor.get_chunks()
    logger.info(f"Successfully processed documents into {len(chunks)} chunks.")

    # Step 2: RAG setup and vector indexing
    logger.info("Step 2: Building vector memory (FAISS index)...")
    vector_mgr = VectorManager(chunks)
    vector_mgr.create_and_save_index()
    retriever = vector_mgr.get_retriever()

    # Step 3: Agent and tools initialization
    logger.info("Step 3: Initializing MedicalPolicyAgent and tools...")
    policy_agent = MedicalPolicyAgent(retriever)
    executor = policy_agent.create_agent_executor()

    # Step 4: Mission definition and logic constraints
    mission_prompt = """
        1. Scan all PDF files for billing rules.
        2. For each rule, search industry standards and write a SQL violation query using Appendix fields.
        3. make sure that when you extract the data from the pdfs you get all the rules not just some
        4. very very important!!! rules must contain real codes from the file dont use example codes and dont ignore the numbers or generic. use the real codes in the query
        5. Categorize rules into Mutual Exclusion, Overutilization, or Service Not Covered.
        6. Call 'FinalReportGenerator' with ALL rules found. and only after all rules found
        7. Identify a billing rule (e.g., Drug Wastage or Endoscopy).
        8. Once a rule is identified, use 'PolicyRetrieval' AGAIN specifically to find the numeric CPT codes associated with it. 
        IMPORTANT: Your final answer MUST be the output of the 'FinalReportGenerator' tool.
        """

    # Step 5: Execution and error handling
    logger.info("Step 4: Agent execution started. Analyzing policies...")
    try:
        # Execution of the agent mission
        response = executor.invoke({"input": mission_prompt, "chat_history": []})
        logger.info("Agent execution completed successfully.")

        print("\nFinal Agent Response:")
        print(response["output"])

    except Exception as e:
        logger.error(f"Critical error during agent execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()