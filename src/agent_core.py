from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
import os
import logging

# Internal imports
from src.report_generator import ReportTool
from utils.search_utils import web_search_validation

# Initialize logger for this module
logger = logging.getLogger(__name__)


class MedicalPolicyAgent:
    def __init__(self, retriever):
        """
        Initialize the AI Agent with LLM, tools, and document retriever.
        """
        logger.info("Initializing MedicalPolicyAgent components...")

        # Set temperature to 0 for deterministic SQL and rule extraction
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.retriever = retriever
        self.report_writer = ReportTool()

        # Definition of tools available to the agent
        # Logging added via tool descriptions or internal logic
        self.tools = [
            Tool(
                name="PolicyRetrieval",
                func=lambda q: self.retriever.get_relevant_documents(q),
                description="Search internal medical policy documents for rules and specific text quotes."
            ),
            Tool(
                name="IndustryValidation",
                func=web_search_validation,
                description="Check if a billing rule is standard in the US healthcare market to determine Logic Confidence."
            ),
            Tool(
                name="FinalReportGenerator",
                func=self.report_writer.write_html_report,
                description="Generates the final HTML report. Call this ONLY once after analyzing ALL rules."
            )
        ]
        logger.info(f"Agent tools initialized: {[tool.name for tool in self.tools]}")

    def create_agent_executor(self):
        """
        Create the agent executor with specific system instructions and SQL constraints.
        """
        logger.info("Building Agent Executor with System Message and Prompt Template...")

        # Allowed schema fields for SQL generation
        appendix_fields = "patient_id, dob, gender, tin, npi, claim_number, dos, pos, diagnosis_code, cpt_code, units, billed_amount, modifiers"

        # System instructions defining the agent's persona and logic boundaries
        system_message = f"""
                You are a Senior Payment Integrity AI Engineer. 
                STRICT SQL FIELDS: {appendix_fields}

                WORKFLOW:
                1. Search documents for rules using 'PolicyRetrieval'.
                2. Validate each rule using 'IndustryValidation'.
                3. Write SQL based ONLY on the Appendix fields.
                4. Categorize as: 'Mutual Exclusion', 'Overutilization', or 'Service Not Covered'.
                5. OUTPUT: Use 'FinalReportGenerator' with a JSON list.

                JSON STRUCTURE EXAMPLE (YOU MUST USE THIS FORMAT):
                [
                  ((
                    "name": "Rule Name",
                    "classification": "Mutual Exclusion",
                    "description": "...",
                    "sql": "...",
                    "confidence": "High",
                    "quote": "..."
                  ))
                ]

                STRICT RULES:
                - Extract strings and numbers literally. Do not summarize code lists.
                - Use ONLY the 'claims' table.
                - SQL JOINs must be on patient_id and dos.
                - Never finish without calling 'FinalReportGenerator'.
                - Important!!!: Do not use the word 'endoscopy' inside the SQL. Instead, find the 5-digit numeric CPT codes in the PDF and list them inside the IN() clause of your query."
                - Use only the specific CPT codes found in the text. dont do example codes
                - make sure you get all!!!! the rules and than generate report
                """

        # Construct the prompt template with placeholders for history and reasoning steps
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        try:
            # Initialize the OpenAI functions agent
            agent = create_openai_functions_agent(self.llm, self.tools, prompt)
            logger.info("OpenAI Functions Agent created successfully.")

            # Return the executor responsible for the agent's runtime loop
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            logger.error(f"Failed to create Agent Executor: {e}")
            raise